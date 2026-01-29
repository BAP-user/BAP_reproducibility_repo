import os
import gc
import csv
import copy
import numpy as np
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import Image, ImageFile, ImageFilter
from tqdm import tqdm

# Specific imports for COCO
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

import open_clip

# =============================================================================
# 0. ENV & CONFIGURATION
# =============================================================================

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

MEAN_TENSOR = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD_TENSOR  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

# =============================================================================
# 1. UTILS & HELPERS
# =============================================================================

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.mode = mode 

    def check(self, current_score):
        improved = False
        stop = False
        if self.mode == 'min':
            if current_score < (self.best_score - self.min_delta):
                self.best_score = current_score
                self.counter = 0
                improved = True
            else:
                self.counter += 1
        else:
            if current_score > (self.best_score + self.min_delta):
                self.best_score = current_score
                self.counter = 0
                improved = True
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            stop = True
        return stop, improved

def standalone_get_mask(ann, height, width):
    seg = ann['segmentation']
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg['counts'], list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    m = mask_utils.decode(rle)
    return Image.fromarray((m * 255).astype(np.uint8), mode='L')

# =============================================================================
# 2. DATA MANAGERS (MEMORY OPTIMIZED)
# =============================================================================

class BackgroundManager:
    def __init__(self, places_root, dtd_root, max_memory_items=100000):
        print("Indexing Backgrounds (Places + DTD)...")
        self.bg_files = []
        if os.path.exists(places_root):
            self.bg_files.extend(glob.glob(os.path.join(places_root, "**", "*.jpg"), recursive=True))
        if os.path.exists(dtd_root):
            self.bg_files.extend(glob.glob(os.path.join(dtd_root, "**", "*.jpg"), recursive=True))
            
        total_found = len(self.bg_files)
        print(f"Found {total_found} total background images.")
        
        if total_found > max_memory_items:
            print(f"Sampling {max_memory_items} random backgrounds to save RAM...")
            random.shuffle(self.bg_files)
            self.bg_files = self.bg_files[:max_memory_items]
        gc.collect() 
        
    def get_random_pool(self, size=None):
        if size: return random.sample(self.bg_files, min(size, len(self.bg_files)))
        return self.bg_files
    
    def preload_to_gpu(self, size, device, target_size=(224, 224)):
        """
        Stores images as UINT8 (0-255) on GPU to save 4x Memory.
        We will normalize them on-the-fly in the Trainer.
        """
        print(f"Preloading {size} backgrounds to GPU VRAM...")
        pool_files = self.get_random_pool(size)
        tensors = []
        for f in tqdm(pool_files, desc="Caching BGs"):
            try:
                with Image.open(f) as img:
                    img = img.convert("RGB").resize(target_size)
                    arr = np.array(img).transpose(2, 0, 1) # HWC -> CHW
                    tensors.append(arr)
            except: continue
        
        # Keep as uint8 to save VRAM
        bg_tensor = torch.tensor(np.array(tensors), dtype=torch.uint8).to(device)
        bg_tensor = bg_tensor.to(memory_format=torch.channels_last)
        
        mem_usage = (bg_tensor.nelement() * 1) / 1e9 # 1 byte per element
        print(f"Backgrounds cached. VRAM Usage: {mem_usage:.2f} GB (Optimized)")
        return bg_tensor

class CocoForegroundManager:
    def __init__(self, ann_path, img_root, target_class_names, min_res=28):
        print(f"Loading COCO annotations from {ann_path}...")
        self.coco = COCO(ann_path)
        self.img_root = img_root
        self.min_res = min_res
        
        cats = self.coco.loadCats(self.coco.getCatIds())
        name_to_id = {cat['name']: cat['id'] for cat in cats}
        self.valid_instances = [] 
        self.class_map = {} 
        
        for idx, cls_name in enumerate(target_class_names):
            if cls_name not in name_to_id: continue
            coco_cat_id = name_to_id[cls_name]
            self.class_map[cls_name] = idx 
            
            ann_ids = self.coco.getAnnIds(catIds=[coco_cat_id], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                bbox = ann['bbox']
                if bbox[2] >= self.min_res and bbox[3] >= self.min_res:
                    img_info = self.coco.loadImgs(ann['image_id'])[0]
                    self.valid_instances.append({
                        'img_path': os.path.join(self.img_root, img_info['file_name']),
                        'ann': ann, 
                        'label_idx': idx,
                        'class_name': cls_name,
                        'height': img_info['height'],
                        'width': img_info['width']
                    })
        print(f"Total Valid Foreground Instances: {len(self.valid_instances)}")

# =============================================================================
# 3. UNIFIED RAM DATASET
# =============================================================================

class UnifiedRAMDataset(Dataset):
    def __init__(self, valid_instances, expansions=1):
        self.expansions = expansions
        self.images = [] 
        self.masks = []  
        self.targets = [] 
        self.keys = []   
        
        initial_targets = []
        
        print(f">> Caching {len(valid_instances)} images to RAM as UINT8 (One-time load)...")
        resize_pil = T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR)

        for item in tqdm(valid_instances, desc="Caching"):
            try:
                with Image.open(item['img_path']) as fg_img:
                    fg_img = fg_img.convert("RGB")
                    mask_img = standalone_get_mask(item['ann'], item['height'], item['width'])
                    
                    bbox = item['ann']['bbox']
                    x, y, w, h = [int(b) for b in bbox]
                    fg_crop = fg_img.crop((x, y, x+w, y+h))
                    mask_crop = mask_img.crop((x, y, x+w, y+h))
                    
                    fg_resized = resize_pil(fg_crop)
                    mask_resized = resize_pil(mask_crop)
                    
                    fg_byte = torch.from_numpy(np.array(fg_resized)).permute(2, 0, 1) 
                    mask_byte = torch.from_numpy(np.array(mask_resized)).unsqueeze(0)
                    
                    self.images.append(fg_byte)
                    self.masks.append(mask_byte)
                    
                    initial_targets.append(torch.tensor(item['label_idx'], dtype=torch.long))
                    self.keys.append(f"{item['ann']['image_id']}_{item['ann']['id']}")
            except:
                continue

        self.images = torch.stack(self.images) # uint8
        self.masks = torch.stack(self.masks)   # uint8
        self.targets = torch.stack(initial_targets) 
        
        mem_gb = (self.images.nelement() + self.masks.nelement()) / 1e9
        print(f"Dataset Cached. Size: {self.images.shape} | RAM Usage: {mem_gb:.2f} GB")

    def switch_to_align_mode(self, anchor_lookup):
        print(">> Switching Dataset to Alignment Mode (Updating Targets)...")
        new_targets = []
        valid_indices = []
        
        # Detect vector dimension from first item
        sample_dim = next(iter(anchor_lookup.values())).shape[-1]
        
        for idx, key in enumerate(self.keys):
            if key in anchor_lookup:
                new_targets.append(anchor_lookup[key].squeeze())
                valid_indices.append(idx)
            else:
                new_targets.append(torch.zeros(sample_dim))
                
        self.targets = torch.stack(new_targets)
        return valid_indices

    def switch_to_erm_mode(self):
        # Only needed if you run align THEN erm, but logic is implicit via main structure
        pass

    def __len__(self):
        return len(self.images) * self.expansions

    def __getitem__(self, idx):
        real_idx = idx % len(self.images)
        fg_byte = self.images[real_idx]
        mask_byte = self.masks[real_idx]
        target = self.targets[real_idx]
        return fg_byte, mask_byte, target

# =============================================================================
# 4. DISTILLATION
# =============================================================================

def distill_coco_vectors(model, coco_manager, bg_manager, n_bg_samples=10, device='cuda'):
    model.eval()
    anchor_lookup = {} 
    
    # Load smaller chunk for distillation
    bg_gpu_cache = bg_manager.preload_to_gpu(size=2000, device=device)
    num_cached_bgs = bg_gpu_cache.size(0)
    
    print(">> Phase 1: Distilling COCO Anchors...")
    pbar = tqdm(coco_manager.valid_instances, desc="Distilling")
    mean_dev = MEAN_TENSOR.to(device)
    std_dev = STD_TENSOR.to(device)
    
    for i, item in enumerate(pbar):
        if i % 1000 == 0: gc.collect()
        unique_key = f"{item['ann']['image_id']}_{item['ann']['id']}"
        try:
            with Image.open(item['img_path']) as fg_img:
                fg_img = fg_img.convert("RGB")
                mask_img = standalone_get_mask(item['ann'], item['height'], item['width'])
                bbox = item['ann']['bbox']
                x, y, w, h = [int(b) for b in bbox]
                fg_crop = fg_img.crop((x, y, x+w, y+h))
                mask_crop = mask_img.crop((x, y, x+w, y+h))
                
                target_size = (224, 224)
                scale = 0.75
                new_w, new_h = int(target_size[0] * scale), int(target_size[1] * scale)
                fg_crop.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
                mask_crop.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
                mask_crop = mask_crop.point(lambda p: 255 if p > 100 else 0)
                mask_crop = mask_crop.filter(ImageFilter.GaussianBlur(radius=1))
                
                fg_canvas = Image.new("RGB", target_size, (0,0,0))
                mask_canvas = Image.new("L", target_size, 0)
                offset = ((target_size[0] - fg_crop.size[0]) // 2, (target_size[1] - fg_crop.size[1]) // 2)
                fg_canvas.paste(fg_crop, offset)
                mask_canvas.paste(mask_crop, offset)
                
                fg_tensor = torch.from_numpy(np.array(fg_canvas)).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
                mask_tensor = torch.from_numpy(np.array(mask_canvas)).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32) / 255.0

            indices = torch.randint(0, num_cached_bgs, (n_bg_samples,), device=device)
            # Convert BG (uint8) to Float on the fly
            bg_batch = bg_gpu_cache[indices].to(dtype=torch.float32) / 255.0
            
            composites = (fg_tensor * mask_tensor) + (bg_batch * (1.0 - mask_tensor))
            composites = (composites - mean_dev) / std_dev
            composites = composites.to(memory_format=torch.channels_last)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16): 
                    feats = model.visual(composites)
                    if len(feats.shape) > 2: feats = torch.flatten(feats, 1)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    mean_vec = torch.mean(feats, dim=0, keepdim=True)
                    purified_vec = mean_vec / mean_vec.norm(dim=-1, keepdim=True)
                anchor_lookup[unique_key] = purified_vec.cpu()
        except: pass
    
    del bg_gpu_cache
    torch.cuda.empty_cache()
    return anchor_lookup

# =============================================================================
# 5. MODEL & TRAINER (GPU MEMORY FIX)
# =============================================================================

class UnifiedCLIP(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        self.device = device
        if 'ViT' in model_name: arch, pretrained = 'ViT-B-16', 'laion2b_s34b_b88k'
        elif 'convnext' in model_name: arch, pretrained = 'convnext_base_w', 'laion2b_s13b_b82k'
        print(f"Loading {arch}...")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.visual = self.clip_model.visual
        self.visual = self.visual.to(memory_format=torch.channels_last)
        
        if hasattr(self.visual, 'output_dim'): self.embed_dim = self.visual.output_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.visual(dummy)
                self.embed_dim = out.shape[1]
        self.head = nn.Linear(self.embed_dim, num_classes)
        # try: self.visual = torch.compile(self.visual)
        # except: pass

    def forward(self, x):
        features = self.visual(x)
        if len(features.shape) > 2: features = torch.flatten(features, 1)
        return self.head(features)
    
    def get_features(self, x):
        features = self.visual(x)
        if len(features.shape) > 2: features = torch.flatten(features, 1)
        return features / features.norm(dim=-1, keepdim=True)

class Trainer:
    def __init__(self, device):
        self.device = device
        self.mean = MEAN_TENSOR.to(device)
        self.std = STD_TENSOR.to(device)
    
    def _prepare_batch(self, fg, mask, bg_gpu_tensor, num_bgs):
        """
        Handles the UINT8 -> BF16 conversion on the fly to save VRAM.
        """
        fg = fg.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        
        # 1. Convert FG/Mask to BF16 and normalize
        fg = fg.to(dtype=torch.bfloat16).div(255.0)
        mask = mask.to(dtype=torch.bfloat16).div(255.0)
        
        # 2. Sample BG (It is UINT8 in VRAM)
        bg_indices = torch.randint(0, num_bgs, (fg.size(0),), device=self.device)
        
        # 3. Convert BG to BF16 and normalize
        bg_batch = bg_gpu_tensor[bg_indices].to(dtype=torch.bfloat16).div(255.0)
        
        # 4. Composite
        comp = fg * mask + bg_batch * (1.0 - mask)
        comp = (comp - self.mean.to(dtype=torch.bfloat16)) / self.std.to(dtype=torch.bfloat16)
        comp = comp.to(memory_format=torch.channels_last)
        return comp

    def run_alignment(self, model, train_loader, val_loader, bg_gpu_tensor, max_epochs=40):
        print("\n=== Starting Alignment Phase ===")
        for p in model.clip_model.parameters(): p.requires_grad = False
        for p in model.visual.parameters(): p.requires_grad = True
        
        optimizer = optim.AdamW(model.visual.parameters(), lr=7e-6, weight_decay=0.01)
        warmup_steps = 400

        scheduler = SequentialLR(optimizer, schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max_epochs * len(train_loader), eta_min=5e-7)
        ], milestones=[warmup_steps])
        
        loss_fn = nn.CosineEmbeddingLoss()
        early_stopper = EarlyStopper(patience=50, mode='min')
        num_bgs = bg_gpu_tensor.size(0)
        epochs_run = 0
        
        for ep in range(max_epochs):
            epochs_run += 1
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Align Ep {ep+1}/{max_epochs}")
            for fg, mask, target_vec in pbar:
                target_vec = target_vec.to(self.device, non_blocking=True)
                comp = self._prepare_batch(fg, mask, bg_gpu_tensor, num_bgs)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    preds = model.get_features(comp)
                    loss = loss_fn(preds, target_vec, torch.ones(preds.size(0), device=self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for fg, mask, target_vec in val_loader:
                    target_vec = target_vec.to(self.device, non_blocking=True)
                    comp = self._prepare_batch(fg, mask, bg_gpu_tensor, num_bgs)
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        preds = model.get_features(comp)
                        val_loss += loss_fn(preds, target_vec, torch.ones(preds.size(0), device=self.device)).item()
            val_loss /= len(val_loader)
            print(f"Epoch {ep+1} Result | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
            stop, improved = early_stopper.check(val_loss)
            if improved: torch.save(model.state_dict(), "temp_best_align.pt")
            if stop: break
        if os.path.exists("temp_best_align.pt"):
            model.load_state_dict(torch.load("temp_best_align.pt"))
            os.remove("temp_best_align.pt")
        return epochs_run

    def run_erm(self, model, train_loader, bg_gpu_tensor, fine_tune_epochs):
        print(f"\n=== Starting ERM Phase (5 Probe + {fine_tune_epochs} FT) ===")
        loss_fn = nn.CrossEntropyLoss()
        num_bgs = bg_gpu_tensor.size(0)

        # 1. LINEAR PROBE
        print(">> Linear Probe...")
        for p in model.visual.parameters(): p.requires_grad = False
        for p in model.head.parameters(): p.requires_grad = True

        opt_probe = optim.AdamW(model.head.parameters(), lr=1e-3)

        for ep in range(5):
            model.train()
            total_acc = 0
            total_count = 0
            pbar = tqdm(train_loader, desc=f"Probe Ep {ep+1}/5")

            for fg, mask, target_cls in pbar:
                target_cls = target_cls.to(self.device, non_blocking=True)
                
                # Use memory efficient batch prep
                comp = self._prepare_batch(fg, mask, bg_gpu_tensor, num_bgs)

                opt_probe.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(comp)
                    loss = loss_fn(logits, target_cls)

                loss.backward()
                opt_probe.step()

                acc = (logits.argmax(1) == target_cls).float().sum()
                total_acc += acc
                total_count += target_cls.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            print(f"Probe Ep {ep+1} Summary | Avg Acc: {total_acc/total_count:.4f}")

        # 2. FINE TUNING
        print(f">> Full Fine-Tuning for {fine_tune_epochs} epochs...")
        for p in model.visual.parameters(): p.requires_grad = True

        optimizer = optim.AdamW([
            {'params': model.visual.parameters(), 'lr': 7e-6},
            {'params': model.head.parameters(), 'lr': 5e-4}
        ], weight_decay=0.01)

        warmup_steps = 200
        total_steps = fine_tune_epochs * len(train_loader)

        scheduler = SequentialLR(optimizer, schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
        ], milestones=[warmup_steps])

        for ep in range(fine_tune_epochs):
            model.train()
            ep_loss = 0
            pbar = tqdm(train_loader, desc=f"FT Ep {ep+1}/{fine_tune_epochs}")

            for fg, mask, target_cls in pbar:
                target_cls = target_cls.to(self.device, non_blocking=True)
                
                comp = self._prepare_batch(fg, mask, bg_gpu_tensor, num_bgs)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(comp)
                    loss = loss_fn(logits, target_cls)

                loss.backward()
                optimizer.step()
                scheduler.step()

                ep_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            print(f"FT Epoch {ep+1} Result | Avg Loss: {ep_loss/len(train_loader):.4f}")

# =============================================================================
# 6. MAIN 
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    bg_mgr = BackgroundManager(CONFIG['places_root'], CONFIG['dtd_root'], max_memory_items=100000)
    coco_mgr = CocoForegroundManager(CONFIG['coco_ann_path'], CONFIG['coco_img_root'], CONFIG['target_classes'], min_res=CONFIG['min_resolution'])
    trainer = Trainer(device)
    num_classes = len(CONFIG['target_classes'])
    FIXED_EPOCHS =  160
    
    # Preload BGs (UINT8)
    bg_gpu_tensor = bg_mgr.preload_to_gpu(size=15000, device=device)

    print("\n>> Initializing Unified RAM Dataset...")
    full_ds = UnifiedRAMDataset(coco_mgr.valid_instances, expansions=1)

    for arch_name in CONFIG['model_archs']:
        print(f"\n\n{'='*40}\nProcessing Architecture: {arch_name}\n{'='*40}")

        # --- PHASE 1: ERM ---
        print(f"\n>> Starting ERM Phase (Fixed {FIXED_EPOCHS} Epochs)...")
        model_erm = UnifiedCLIP(arch_name, num_classes, device).to(device)
        
        # Use full_ds in its default state (ERM targets)
        val_size = int(len(full_ds) * 0.1)
        train_size = len(full_ds) - val_size
        train_ds_erm, _ = random_split(full_ds, [train_size, val_size])
        
        train_loader_erm = DataLoader(train_ds_erm, batch_size=CONFIG['batch_size'], shuffle=True, 
                                      num_workers=2, pin_memory=True, drop_last=True, prefetch_factor=2)
        
        trainer.run_erm(model_erm, train_loader_erm, bg_gpu_tensor, fine_tune_epochs=FIXED_EPOCHS)
        
        save_path_erm = os.path.join(CONFIG['output_dir'], f"{arch_name}_erm.pt")
        torch.save(model_erm.state_dict(), save_path_erm)
        
        del model_erm, train_loader_erm, train_ds_erm
        gc.collect()
        torch.cuda.empty_cache()

        # --- PHASE 2: ALIGNMENT ---
        print(f"\n>> Starting Alignment Phase (Fixed {FIXED_EPOCHS} Epochs)...")
        model_align = UnifiedCLIP(arch_name, num_classes, device).to(device)
        
        anchor_lookup = distill_coco_vectors(model_align, coco_mgr, bg_mgr, n_bg_samples=10, device=device)
        
        # Switch dataset targets to Vectors
        valid_indices = full_ds.switch_to_align_mode(anchor_lookup)
        align_subset = Subset(full_ds, valid_indices)
        
        val_size = int(len(align_subset) * 0.1)
        train_size = len(align_subset) - val_size
        train_ds, val_ds = random_split(align_subset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                                  num_workers=2, pin_memory=True, drop_last=True, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                                num_workers=2, pin_memory=True)
        
        trainer.run_alignment(model_align, train_loader, val_loader, bg_gpu_tensor, max_epochs=FIXED_EPOCHS)
        
        save_path = os.path.join(CONFIG['output_dir'], f"{arch_name}_alignment.pt")
        torch.save(model_align.state_dict(), save_path)
        
        del model_align, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()


CONFIG = {
    'coco_ann_path': '/root/data/COCO/coco2017/annotations/instances_train2017.json',
    'coco_img_root': '/root/data/COCO/coco2017/train2017',
    'places_root': '/root/data/places',
    'dtd_root': '/root/data/DTD/dtd/images',
    'output_dir': '/root/coco_robustness_align',
    'target_classes': ["car", "bus", "Airplane", "Boat", "Train",'Bicycle'], 
    'min_resolution': 28,
    'model_archs': ['convnext_base_w'],
    # Reduced batch size to 512 to prevent OOM on ConvNext activations
    'batch_size': 512, 
    'num_workers': 2,
}
if not os.path.exists(CONFIG['output_dir']):
    os.makedirs(CONFIG['output_dir'])

main()
