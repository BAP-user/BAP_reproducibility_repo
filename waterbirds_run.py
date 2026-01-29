import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "0" 

from sklearn.linear_model import LogisticRegression
import time
import sys
import gc
import csv
import copy
import numpy as np
import pandas as pd
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as T
from PIL import Image, ImageFile, ImageFilter
import open_clip
from wilds import get_dataset
from tqdm import tqdm
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from functools import lru_cache

# 0. CRITICAL ENV SETUP
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True 

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================

# CLIP Templates
TEMPLATES = [
    'a photo of a {}.', 'a photo of the {}.', 'a photo of many {}.', 'a photo of the hard to see {}.',
    'a photo of a hard to see {}.', 'a low resolution photo of the {}.', 'a low resolution photo of a {}.',
    'a bad photo of the {}.', 'a bad photo of a {}.', 'a cropped photo of the {}.', 'a cropped photo of a {}.',
    'a bright photo of a {}.', 'a bright photo of the {}.', 'a dark photo of a {}.', 'a dark photo of a {}.',
    'a photo of a clean {}.', 'a photo of the clean {}.', 'a photo of a dirty {}.', 'a photo of the dirty {}.',
    'a close-up photo of a {}.', 'a close-up photo of the {}.', 'a black and white photo of the {}.',
    'a black and white photo of a {}.', 'a jpeg photo of a {}.', 'a jpeg photo of the {}.', 'a blurry photo of the {}.',
    'a blurry photo of a {}.', 'a good photo of the {}.', 'a good photo of a {}.', 'a photo of one {}.',
    'a photo of a large {}.', 'a photo of the large {}.', 'a photo of a nice {}.', 'a photo of the nice {}.',
    'a photo of a small {}.', 'a photo of the small {}.', 'a photo of a weird {}.', 'a photo of the weird {}.',
    'a photo of a cool {}.', 'a photo of the cool {}.'
]

# Standard ImageNet/CLIP Normalization
NORMALIZE = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

WATER_BIRDS_LIST = [
    'Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Crested_Auklet',
    'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brandt_Cormorant',
    'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Frigatebird', 'Northern_Fulmar',
    'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull',
    'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull',
    'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Red_legged_Kittiwake', 'Brown_Pelican',
    'White_Pelican', 'Horned_Puffin', 'Artic_Tern', 'Black_Tern',
    'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern',
    'Least_Tern', 'Gadwall', 'Eared_Grebe', 'Horned_Grebe',
    'Pied_billed_Grebe', 'Western_Grebe', 'Mallard', 'Hooded_Merganser',
    'Red_breasted_Merganser', 'Pigeon_Guillemot', 'Pacific_Loon'
]

ALIGN_PLACES = [
    "coast", "swamp", "river", "pond", "iceberg",
    "closet", "chalet", "castle", "cemetery", "alley", 
    "attic", "bar", "driveway", "orchard", 'engine_room', 'tree_farm',
    "classroom", "auditorium", "bridge", "windmill", "igloo", 
    "sky", "snowfield",'ocean', 'rainforest', 'castle', 'abbey','canyon',
    'market/outdoor', 'martial_arts_gym','shopfront', 'crevasse', 'ocean', 'rainforest'
]

# =============================================================================
# 2. UTILS
# =============================================================================

@lru_cache(maxsize=20000)
def cached_image_open(path, mode='RGB'):
    try:
        with Image.open(path) as img:
            return img.convert(mode)
    except Exception:
        return None

def fixed_composite(bird_path, mask_path, bg_path, target_size=(224,224), bg_is_array=False, bg_array=None):
    try:
        bird_img = cached_image_open(bird_path, "RGB")
        if bird_img is None: return np.zeros(target_size + (3,), dtype=np.float32)
        bird_img = bird_img.copy()

        if os.path.exists(mask_path):
            mask_img = cached_image_open(mask_path, "L")
            if mask_img is not None: mask_img = mask_img.copy()
            else: mask_img = Image.new("L", bird_img.size, 255)
        else:
            mask_img = Image.new("L", bird_img.size, 255)
        
        if bird_img.size != mask_img.size: mask_img = mask_img.resize(bird_img.size)
        bbox = mask_img.getbbox()
        if bbox: 
            bird_crop, mask_crop = bird_img.crop(bbox), mask_img.crop(bbox)
        else: 
            bird_crop, mask_crop = bird_img, mask_img

        if bg_is_array and bg_array is not None:
            bg_img = Image.fromarray(bg_array).convert("RGB")
            if bg_img.size != target_size:
                bg_img = bg_img.resize(target_size)
        else:
            bg_img = cached_image_open(bg_path, "RGB")
            if bg_img is None: return np.zeros(target_size + (3,), dtype=np.float32)
            bg_img = bg_img.copy().resize(target_size)

        scale = 0.6
        max_w, max_h = int(target_size[0] * scale), int(target_size[1] * scale)
        
        bird_crop.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        mask_crop.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        mask_crop = mask_crop.point(lambda p: 255 if p > 100 else 0)
        mask_crop = mask_crop.filter(ImageFilter.GaussianBlur(radius=1))

        offset = ((target_size[0] - bird_crop.size[0]) // 2, (target_size[1] - bird_crop.size[1]) // 2)
        bg_img.paste(bird_crop, offset, mask_crop)
        
        return np.array(bg_img).astype(np.float32) / 255.0

    except Exception as e:
        return np.zeros(target_size + (3,), dtype=np.float32)

def get_image_embeddings(model, image_list, device):
    tensor_list = []
    for img_np in image_list:
        t = torch.from_numpy(img_np).permute(2,0,1).float()
        t = NORMALIZE(t)
        tensor_list.append(t)
    
    batch = torch.stack(tensor_list).to(device)
    with torch.no_grad():
        feats = model.visual(batch)
        if len(feats.shape) > 2: feats = torch.flatten(feats, 1)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

# =============================================================================
# 3. DATA CLASSES & MANAGERS
# =============================================================================

class DistillationDataset(Dataset):
    def __init__(self, align_df, manager, bg_places_list, n_samples=30):
        self.align_df = align_df
        self.manager = manager
        self.bg_places_list = bg_places_list
        self.n_samples = n_samples
        self.normalize = NORMALIZE

    def __len__(self):
        return len(self.align_df)

    def __getitem__(self, idx):
        row = self.align_df.iloc[idx]
        bird_path = row['full_img_path']
        mask_path = row['full_mask_path']
        
        bird_img = cached_image_open(bird_path, "RGB")
        if bird_img is None: 
            return torch.zeros(self.n_samples, 3, 224, 224), row['filepath'], False

        bird_img = bird_img.copy()
        if os.path.exists(mask_path):
            mask_img = cached_image_open(mask_path, "L")
            if mask_img is not None: mask_img = mask_img.copy()
            else: mask_img = Image.new("L", bird_img.size, 255)
        else:
            mask_img = Image.new("L", bird_img.size, 255)

        if bird_img.size != mask_img.size: mask_img = mask_img.resize(bird_img.size)
        bbox = mask_img.getbbox()
        if bbox: bird_img, mask_img = bird_img.crop(bbox), mask_img.crop(bbox)

        valid_tensors = []
        attempts = 0
        
        valid_bgs = []
        for _ in range(3): 
            bg_cat = random.choice(self.bg_places_list)
            pool = self.manager.get_background_pool(bg_cat, split='train')
            if pool: valid_bgs.extend(pool)
            
        if not valid_bgs:
             return torch.zeros(self.n_samples, 3, 224, 224), row['filepath'], False

        while len(valid_tensors) < self.n_samples and attempts < self.n_samples * 5:
            attempts += 1
            bg_path = random.choice(valid_bgs)
            try:
                bg_pil = cached_image_open(bg_path, "RGB")
                if bg_pil is None: continue
                bg_pil = bg_pil.copy().resize((224, 224))
                
                target_size = (224, 224)
                scale = 0.75
                new_w, new_h = int(target_size[0] * scale), int(target_size[1] * scale)
                
                fg_r = bird_img.copy()
                mask_r = mask_img.copy()
                fg_r.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
                mask_r.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
                mask_r = mask_r.point(lambda p: 255 if p > 100 else 0)
                mask_r = mask_r.filter(ImageFilter.GaussianBlur(radius=1))
                
                offset = ((target_size[0] - fg_r.size[0]) // 2, (target_size[1] - fg_r.size[1]) // 2)
                bg_pil.paste(fg_r, offset, mask_r)
                
                t = self.normalize(T.ToTensor()(bg_pil))
                valid_tensors.append(t)
                
            except Exception:
                continue
        
        if len(valid_tensors) < self.n_samples:
             return torch.zeros(self.n_samples, 3, 224, 224), row['filepath'], False
             
        return torch.stack(valid_tensors), row['filepath'], True

class InstanceAlignedDataset(Dataset):
    def __init__(self, metadata, bg_pool, target_lookup, expansions=10):
        self.bg_pool = bg_pool
        self.target_lookup = target_lookup
        self.expansions = expansions
        self.normalize = NORMALIZE
        
        self.valid_birds = []
        for _, row in metadata.iterrows():
            if row['filepath'] in self.target_lookup:
                self.valid_birds.append({
                    'img_path': row['full_img_path'],
                    'mask_path': row['full_mask_path'],
                    'target': self.target_lookup[row['filepath']],
                    'label': row['label']
                })
        
        print(f"Dynamic Dataset Initialized: {len(self.valid_birds)} unique birds.")
        print(f"Virtual Epoch Size: {len(self.valid_birds) * expansions} samples.")

    def __len__(self):
        return len(self.valid_birds) * self.expansions

    def __getitem__(self, idx):
        bird_idx = idx % len(self.valid_birds)
        bird_data = self.valid_birds[bird_idx]

        bg_path = random.choice(self.bg_pool)
        
        img_np = fixed_composite(bird_data['img_path'], bird_data['mask_path'], bg_path)
        tensor = self.normalize(torch.from_numpy(img_np).permute(2, 0, 1).float())
        
        return tensor, bird_data['target'].squeeze(), torch.tensor(bird_data['label'], dtype=torch.long)

class DataSplitManager:
    def __init__(self, cub_root, places_root, waterbirds_meta_path=None, bg_split_ratio=0.8):
        self.cub_root = cub_root
        self.places_root = places_root
        self.waterbirds_meta_path = waterbirds_meta_path 
        self.cub_df = self._load_cub_metadata()
        self.places_cache = {}
        self.bg_split_ratio = bg_split_ratio

    def _load_cub_metadata(self):
        try:
            images_df = pd.read_csv(os.path.join(self.cub_root, 'images.txt'), sep=r'\s+', names=['img_id', 'filepath'])
            labels_df = pd.read_csv(os.path.join(self.cub_root, 'image_class_labels.txt'), sep=r'\s+', names=['img_id', 'class_id'])
            classes_df = pd.read_csv(os.path.join(self.cub_root, 'classes.txt'), sep=r'\s+', names=['class_id', 'class_name'])
            data = images_df.merge(labels_df, on='img_id').merge(classes_df, on='class_id')
            data['full_img_path'] = data['filepath'].apply(lambda x: os.path.join(self.cub_root, 'images', x))
            data['full_mask_path'] = data['filepath'].apply(lambda x: os.path.join(self.cub_root, 'segmentations', x.replace('.jpg', '.png')))
            data['is_waterbird'] = data['class_name'].apply(lambda x: any(wb in x for wb in WATER_BIRDS_LIST))
            data['super_class_name'] = data['is_waterbird'].apply(lambda x: 'waterbird' if x else 'landbird')
            data['label'] = data['is_waterbird'].astype(int)
            
            if self.waterbirds_meta_path and os.path.exists(self.waterbirds_meta_path):
                print(f"[DataSplitManager] checking for leakage against: {self.waterbirds_meta_path}")
                wb_df = pd.read_csv(self.waterbirds_meta_path)
                test_images = set(wb_df[wb_df['split'] == 2]['img_filename'].values)
                initial_count = len(data)
                data = data[~data['filepath'].isin(test_images)]
                removed_count = initial_count - len(data)
                print(f"[DataSplitManager] LEAKAGE CHECK: Removed {removed_count} CUB images that overlap with Waterbirds Test Set.")
            else:
                print("[DataSplitManager] Warning: No Waterbirds metadata found. Skipping leakage check.")

            return data
        except Exception as e:
            print(f"Error loading CUB: {e}")
            return pd.DataFrame()

    def get_background_pool(self, bg_types, split='train'):
        if isinstance(bg_types, str): bg_types = [bg_types]
        pool = []
        for bg in bg_types:
            if bg not in self.places_cache:
                pat = os.path.join(self.places_root, "**", bg, "*.jpg")
                files = sorted(glob.glob(pat, recursive=True))
                if not files: 
                    pat = os.path.join(self.places_root, "**", bg + "*.jpg")
                    files = sorted(glob.glob(pat, recursive=True))
                
                if len(files) > 0:
                    rng = random.Random(42)
                    rng.shuffle(files)
                    idx = int(len(files) * self.bg_split_ratio)
                    self.places_cache[bg] = {'train': files[:idx], 'test': files[idx:]}
            
            if bg in self.places_cache:
                pool.extend(self.places_cache[bg][split])
        return pool

class WaterbirdsDataset(Dataset):
    def __init__(self, wilds_subset, transform=None):
        self.subset = wilds_subset
        self.transform = transform
        
    def __len__(self): return len(self.subset)
        
    def __getitem__(self, idx):
        x, y, metadata = self.subset[idx]
        if self.transform: x = self.transform(x)
        return x, y, metadata

def get_group_idx(y, metadata):
    device = y.device if isinstance(y, torch.Tensor) else 'cpu'
    if isinstance(metadata, torch.Tensor):
        place = metadata[:, 0].to(device)
    else: 
        place = metadata[:, 0]
    return (y * 2 + place).long()

class DataManager:
    def __init__(self, root_dir, batch_size=128):
        print(f"Loading Waterbirds from {root_dir}...")
        self.dataset = get_dataset(dataset="waterbirds", root_dir=root_dir, download=False)
        self.batch_size = batch_size
        
        self.eval_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORMALIZE])
        self.train_transform = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.ToTensor(), NORMALIZE])
        self.randaug_transform = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=9), T.ToTensor(), NORMALIZE])

    def _loader_args(self, shuffle=False, workers=8): 
        return {
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'num_workers': workers,       
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2
        }

    def get_loader(self, split, transform_type='standard', workers=8): 
        subset = self.dataset.get_subset(split)
        tf = self.randaug_transform if transform_type == 'randaug' else (self.train_transform if transform_type == 'train' else self.eval_transform)
        ds = WaterbirdsDataset(subset, transform=tf)
        return DataLoader(ds, **self._loader_args(shuffle=(split=='train'), workers=workers))

    def get_stratified_loader(self, split, target_size, transform_type='train'):
        full_subset = self.dataset.get_subset(split)
        y_all = full_subset.y_array
        meta_all = full_subset.metadata_array
        groups = (y_all * 2 + meta_all[:, 0]).long()
        
        total_count = len(full_subset)
        indices_by_group = defaultdict(list)
        for idx, g in enumerate(groups.tolist()):
            indices_by_group[g].append(idx)
            
        selected_indices = []
        for g, indices in indices_by_group.items():
            ratio = len(indices) / total_count
            count_to_select = int(ratio * target_size)
            selected = random.sample(indices, count_to_select)
            selected_indices.extend(selected)
            
        remaining = target_size - len(selected_indices)
        if remaining > 0:
            all_indices = set(range(total_count))
            used_indices = set(selected_indices)
            pool = list(all_indices - used_indices)
            if pool:
                extras = random.sample(pool, min(remaining, len(pool)))
                selected_indices.extend(extras)
                
        print(f"  [Stratified Sampler] Original: {total_count} -> Subsampled: {len(selected_indices)}")
        tf = self.train_transform if transform_type == 'train' else self.eval_transform
        sub_ds = Subset(WaterbirdsDataset(full_subset, transform=tf), selected_indices)
        
        kwargs = self._loader_args(shuffle=True)
        kwargs['batch_size'] = 64
        return DataLoader(sub_ds, **kwargs)

# =============================================================================
# 4. MODEL CLASS
# =============================================================================

class UnifiedCLIP(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        if 'vit' in model_name: arch, pretrained = 'ViT-B-16', 'laion2b_s34b_b88k'
        elif 'resnet' in model_name: arch, pretrained = 'RN50', 'openai'
        elif 'convnext' in model_name: arch, pretrained = 'convnext_base_w', 'laion2b_s13b_b82k'
        else: arch, pretrained = 'ViT-B-32', 'laion2b_s34b_b79k'

        print(f"Loading {arch} with weights: {pretrained}...")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.visual = self.clip_model.visual
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.visual(dummy)
            self.embed_dim = out.shape[1] if len(out.shape) > 2 else out.shape[-1]

        self.head = nn.Linear(self.embed_dim, 2) 

    def setup_training_mode(self, mode, linear_probe=False):
        if mode == 'align':
            for p in self.clip_model.parameters(): p.requires_grad = False
            for p in self.visual.parameters(): p.requires_grad = True
        elif mode == 'probe':
            for p in self.clip_model.parameters(): p.requires_grad = False
            if linear_probe:
                for p in self.visual.parameters(): p.requires_grad = False
                for p in self.head.parameters(): p.requires_grad = True
            else:
                for p in self.visual.parameters(): p.requires_grad = True
                for p in self.head.parameters(): p.requires_grad = True
        elif mode == 'eval':
            self.eval()
    
    def encode_image(self, x):
        features = self.visual(x)
        if len(features.shape) > 2: features = torch.flatten(features, 1)
        return features

    def forward(self, x): return self.head(self.encode_image(x))

# =============================================================================
# 5. EXPERIMENT ORCHESTRATOR
# =============================================================================

class BenchmarkOrchestrator:
    def __init__(self, config):
        self.cfg = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.data_mgr = DataManager(config['waterbirds_root'], batch_size=128)
        self.synthetic_mgr = DataSplitManager(
            config['cub_root'], 
            config['places_root'], 
            waterbirds_meta_path=config['waterbirds_meta_csv']
        )
        
        full_train = self.data_mgr.dataset.get_subset('train')
        self.FULL_TRAIN_SIZE = len(full_train)
        print(f"Total Official Training Budget: {self.FULL_TRAIN_SIZE} images")
        
        # --- LOGGING SETUP ---
        # 1. Trajectory Log (Epoch-wise stats during Stage 3)
        traj_headers = [
            'run_id', 'stage', 'epoch', 
            'acc_avg', 'acc_LB_L', 'acc_LB_W', 'acc_WB_L', 'acc_WB_W', 'worst_group_acc',
            'train_loss'
        ]
        if not os.path.exists(config['trajectory_csv']):
            with open(config['trajectory_csv'], 'w', newline='') as f:
                csv.writer(f).writerow(traj_headers)

        # 2. Alignment Loss Log (Epoch-wise loss during Stage 2)
        align_headers = ['run_id', 'epoch', 'train_loss', 'val_loss']
        if not os.path.exists(config['align_loss_csv']):
            with open(config['align_loss_csv'], 'w', newline='') as f:
                csv.writer(f).writerow(align_headers)

    def _get_loader_kwargs(self, batch_size, shuffle, workers=8):
        return {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': workers,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2
        }

    def _kill_loader(self, loader):
        try:
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                loader._iterator._shutdown_workers()
                del loader._iterator
        except Exception:
            pass
        
        try:
            del loader
        except Exception:
            pass
        
        time.sleep(2)

    def run(self):
        routine = 'Align_Synthetic_Trajectory'
        
        for run_id in range(1, self.cfg['num_runs'] + 1):
            # --- FIX: Update seeds based on run_id ---
            current_seed = 42 + run_id
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            random.seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)
            
            for model_name in self.cfg['model_arch']:
                # === NO SWEEP: Use fixed n_samples ===
                current_n_samples = self.cfg['fixed_n_samples']
                fixed_align_count = self.cfg['fixed_align_count']
                fixed_expansions = self.cfg['fixed_expansion_count']
                
                print(f"\n=== Run {run_id} | n_samples: {current_n_samples} | Expansions: {fixed_expansions} | Align Count: {fixed_align_count} | Model: {model_name} ===")
                model = UnifiedCLIP(model_name, self.device).to(self.device)
                
                df_cub = self.synthetic_mgr.cub_df
                df_wb = df_cub[df_cub['label'] == 1]
                df_lb = df_cub[df_cub['label'] == 0]
                
                half = fixed_align_count // 2
                wb_samp = df_wb.sample(n=min(half, len(df_wb)), replace=False)
                lb_samp = df_lb.sample(n=min(half, len(df_lb)), replace=False)
                align_df = pd.concat([wb_samp, lb_samp]).sample(frac=1).reset_index(drop=True)

                # --- PHASE 1: PARALLEL DISTILLATION ---
                print(f">> Phase 1: Distilling Instance Vectors ({len(align_df)} birds)...")
                target_lookup = {}
                model.eval()
                distill_bgs = self.cfg.get('align_bgs', ALIGN_PLACES) 
                
                distill_ds = DistillationDataset(align_df, self.synthetic_mgr, distill_bgs, n_samples=current_n_samples)
                distill_loader = DataLoader(
                    distill_ds, 
                    batch_size=1,        
                    shuffle=False, 
                    num_workers=8,        
                    pin_memory=True,
                    prefetch_factor=4
                )
                
                with torch.no_grad():
                    for batch_imgs, filepaths, valids in tqdm(distill_loader, desc="Distilling"):
                        if not valids.item(): continue
                        
                        batch_imgs = batch_imgs.squeeze(0).to(self.device, non_blocking=True)
                        
                        feats = model.visual(batch_imgs)
                        if len(feats.shape) > 2: feats = torch.flatten(feats, 1)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        
                        mean_vec = torch.mean(feats, dim=0, keepdim=True)
                        purified_vec = mean_vec / mean_vec.norm(dim=-1, keepdim=True)
                        
                        target_lookup[filepaths[0]] = purified_vec.cpu()

                print(f"Distilled vectors generated for {len(target_lookup)}/{len(align_df)} birds.")
                
                # --- PHASE 2: ALIGNMENT TRAINING & LOSS LOGGING ---
                bg_pool = self.synthetic_mgr.get_background_pool(distill_bgs, split='train')
                inst_ds = InstanceAlignedDataset(align_df, bg_pool, target_lookup, expansions=fixed_expansions)
                
                print(f">> Phase 2: Alignment Training (Instance-Wise with {fixed_expansions} expansions)...")
                self._train_alignment_instance(model, inst_ds, run_id)
                self._cleanup()
                
                # --- PHASE 3: PROBE TRAINING + TRAJECTORY LOGGING (LP + FT) ---
                probe_budget = self.FULL_TRAIN_SIZE 
                probe_loader = self.data_mgr.get_stratified_loader('train', target_size=probe_budget)
                print(f">> Phase 3: Probe Training & Trajectory Logging...")
                
                self._train_probe_trajectory(model, probe_loader, run_id)

                del probe_loader
                self._cleanup()
                del model; gc.collect(); torch.cuda.empty_cache()

    def _train_alignment_instance(self, model, full_dataset, run_id):
        model.setup_training_mode('align')
        
        total_len = len(full_dataset)
        val_size = int(total_len * 0.1)
        train_size = total_len - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        
        BATCH_SIZE = 256
        kwargs = self._get_loader_kwargs(BATCH_SIZE, shuffle=True)
        train_loader = DataLoader(train_ds, **kwargs)
        
        kwargs['shuffle'] = False
        val_loader = DataLoader(val_ds, **kwargs)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=5e-6, weight_decay=0.01)

        max_epochs = 30 
        total_steps = len(train_loader) * max_epochs
        warmup_steps = int(0.1 * total_steps)
        
        scheduler = SequentialLR(optimizer, schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
        ], milestones=[warmup_steps])

        loss_fn = nn.CosineEmbeddingLoss()
        
        for ep in range(max_epochs):
            model.train()
            train_loss = 0
            for imgs, targets, _ in train_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    img_embs = model.visual(imgs)
                    if len(img_embs.shape) > 2: img_embs = torch.flatten(img_embs, 1)
                    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
                    
                    y_ones = torch.ones(imgs.size(0)).to(self.device)
                    loss = loss_fn(img_embs, targets, y_ones)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets, _ in val_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    img_embs = model.visual(imgs)
                    if len(img_embs.shape) > 2: img_embs = torch.flatten(img_embs, 1)
                    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
                    
                    y_ones = torch.ones(imgs.size(0)).to(self.device)
                    val_loss += loss_fn(img_embs, targets, y_ones).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # --- LOGGING ALIGNMENT LOSS ---
            print(f" [Align Ep {ep+1}/{max_epochs}] Train: {avg_train_loss:.4f} Val: {avg_val_loss:.4f}")
            with open(self.cfg['align_loss_csv'], 'a', newline='') as f:
                csv.writer(f).writerow([run_id, ep + 1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}"])

        del train_loader
        del val_loader
        self._cleanup()

    def _cleanup(self):
        cached_image_open.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
        if sys.platform == 'linux':
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass
        print("[System] Memory Cleared & Cache Flushed.")

    def _train_probe_trajectory(self, model, train_loader, run_id):
        # We need a persistent test loader for evaluation at each epoch
        test_loader = self.data_mgr.get_loader('test', 'standard')

        try:
            if hasattr(train_loader.dataset, 'subset'): 
                y_targets = train_loader.dataset.subset.y_array
            else: 
                y_targets = torch.tensor([y for _, y, _ in train_loader.dataset])
                
            counts = torch.bincount(y_targets.cpu())
            counts[counts == 0] = 1 
            weights = len(y_targets) / (len(counts) * counts.float())
            crit = nn.CrossEntropyLoss(weight=weights.to(self.device))
        except: 
            crit = nn.CrossEntropyLoss()

        # ================================
        # STAGE 3.A: LINEAR PROBE (30 Epochs)
        # ================================
        model.setup_training_mode('probe', linear_probe=True)
        optimizer = optim.AdamW(model.head.parameters(), lr=1e-4)
        
        epochs_lp = 30
        print(f"  > Starting Linear Probe Trajectory ({epochs_lp} epochs)...")

        for ep in range(epochs_lp):
            model.train()
            # Explicitly freeze visual, unfreeze head just to be safe in loop
            model.visual.requires_grad_(False)
            model.head.requires_grad_(True)
            
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"LP Ep {ep+1}/{epochs_lp}", leave=False):
                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = crit(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_loss = train_loss / len(train_loader)
            
            # --- EVALUATE & LOG ---
            stats = self._evaluate(model, test_loader)
            self._log_trajectory(run_id, 'LinearProbe', ep + 1, stats, avg_loss)
            print(f"   [LP Ep {ep+1}] Loss: {avg_loss:.4f} | Avg Acc: {stats['acc_avg']:.4f} | WGA: {stats['worst_group_acc']:.4f}")

        # ================================
        # STAGE 3.B: FINE-TUNE (15 Epochs)
        # ================================
        print(f"  > Unfreezing backbone for Fine-Tuning Trajectory (15 epochs, lr=5e-6)...")
        
        # Unfreeze everything
        for p in model.parameters(): p.requires_grad = True
        
        # New optimizer for full model
        optimizer = optim.AdamW(model.parameters(), lr=1e-6)
        
        epochs_ft = 0
        
        for ep in range(epochs_ft):
            model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"FT Ep {ep+1}/{epochs_ft}", leave=False):
                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = crit(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            avg_loss = train_loss / len(train_loader)
            
            # --- EVALUATE & LOG ---
            stats = self._evaluate(model, test_loader)
            # Log epoch as 31...45 to represent continuation
            actual_epoch_log = epochs_lp + ep + 1
            self._log_trajectory(run_id, 'FineTune', actual_epoch_log, stats, avg_loss)
            print(f"   [FT Ep {ep+1}] Loss: {avg_loss:.4f} | Avg Acc: {stats['acc_avg']:.4f} | WGA: {stats['worst_group_acc']:.4f}")

        del test_loader

    def _evaluate(self, model, loader):
        model.eval()
        group_correct, group_total = defaultdict(int), defaultdict(int)
        
        with torch.no_grad():
            for x, y, metadata in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                logits = model(x)
                preds = logits.argmax(1)
                group_ids = get_group_idx(y, metadata).tolist()
                
                for i, gid in enumerate(group_ids):
                    is_correct = (preds[i] == y[i]).item()
                    group_correct[gid] += is_correct
                    group_total[gid] += 1
                
        stats = {}
        total_corr = sum(group_correct.values())
        total_cnt = sum(group_total.values())
        stats['acc_avg'] = total_corr / total_cnt if total_cnt > 0 else 0
        
        group_map = {0: 'acc_LB_L', 1: 'acc_LB_W', 2: 'acc_WB_L', 3: 'acc_WB_W'}
        accuracies = []
        for gid, name in group_map.items():
            acc = group_correct[gid] / group_total[gid] if group_total[gid] > 0 else 0
            stats[name] = acc
            accuracies.append(acc)
        stats['worst_group_acc'] = min(accuracies)
        
        return stats

    def _log_trajectory(self, run_id, stage, epoch, stats, loss):
        row = [
            run_id, stage, epoch,
            f"{stats['acc_avg']:.4f}",
            f"{stats['acc_LB_L']:.4f}", f"{stats['acc_LB_W']:.4f}",
            f"{stats['acc_WB_L']:.4f}", f"{stats['acc_WB_W']:.4f}",
            f"{stats['worst_group_acc']:.4f}",
            f"{loss:.4f}"
        ]
        with open(self.cfg['trajectory_csv'], 'a', newline='') as f:
            csv.writer(f).writerow(row)

def main():
    config = {
        # Required paths
        "waterbirds_root": "/CLIP/data/wilds_data/",
        "cub_root": "/CLIP/data/waterbirds/CUB_200_2011",
        "places_root": "/CLIP/data/places",
        "waterbirds_meta_csv": "/CLIP/data/wilds_data/waterbirds_v1.0/metadata.csv",

        # Output logs
        "trajectory_csv": "/CLIP/data/BAP_trajectory_logs.csv",
        "align_loss_csv": "/CLIP/data/BAP_align_loss_logs.csv",

        # Experiment control
        "num_runs": 5,

        # Fixed Parameters (No Sweep)
        "fixed_n_samples": 16,
        "fixed_expansion_count": 4,
        "fixed_align_count": 4000,

        # Model architectures to evaluate
        "model_arch": ["vit"]
    }

    orchestrator = BenchmarkOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
