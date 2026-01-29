# Code for BAP

This repository contains the implementation of BAP and code to reproduce the main Waterbirds benchmark results reported in the paper.

---

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
````

For reference, we provide:

* `gpu_info.txt` — GPU and driver information
* `cuda_version.txt` — CUDA toolkit version used
* `requirements.txt` — Python package versions

These files describe the environment used to produce the reported results.


## Dataset Preparation

Datasets are downloaded and prepared automatically by running:

```bash
python init_datasets.py
```

This script downloads and structures all required datasets.


## Running Our Method

To reproduce the main Waterbirds results from the paper:

```bash
python waterbirds_run.py
```

Results are written to CSV log files in the configured output directory.

---

## Additional Large-Scale Experiment

The file:

```bash
large_scale_demo.py
```

contains code for the additional large-scale COCO vehicle instances pre-training experiment included for completeness.
This experiment requires very large dataset and high-memory GPUs (e.g., NVIDIA H100) and is **not expected to be runnable on standard hardware**. It is provided for reference only.

---

## Notes

* This release is intended for research reproducibility.
* The code is provided as-is for the purpose of paper review.
