# HB-FCL

The source code for 📄 **HB-FCL (High-fidelity Boundary-aware Federated Continual Learning for Medical Image Diagnosis)**.

## 📑 Table of Contents

- [📋 Repository Layout](#-repository-layout)
- [⚙️ Environment Setup](#-environment-setup)
- [🗂️ Datasets](#-datasets)
- [🚀 Running an Experiment](#-running-an-experiment)
- [📊 Results](#-results)

---

## 📋 Repository Layout

```
HB-FCL/
├── main.py                     # Command-line entry point for sequential federated training
├── methods/                    # Continual and federated learning algorithms
├── dataset/                    # Dataset adapters and preprocessing pipelines
├── convs/                      # Backbone and classifier implementations
├── utils/                      # Data partitioning, logging, CLIP text conditioning, synthetic-data utilities
├── forgetting_calculator/      # Forgetting metrics computed from saved AUC records
└── assist/                     # Optional scripts for TSE training and synthetic-data generation
```

---

## ⚙️ Environment Setup

The code has been tested on **NVIDIA RTX 5880 Ada Generation 48G GPU**.

### 1. Install PyTorch

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pretrained Model Checkpoints

The following pretrained models require local checkpoints (the code does **not** download weights automatically):

- 🧠 Diffusion model: [Roentgen-v2](https://huggingface.co/stanfordmimi/RoentGen-v2) (MINIM)
- 🧠 CLIP model: [kaveh/rclip](https://huggingface.co/kaveh/rclip)

Please download and place them in accessible paths, then specify the path when running experiments (see examples below).

---

## 🗂️ Datasets

The loaders support four public chest X-ray datasets:

1. **CXR**  
   - Source: [CXR Dataset](https://www.heywhale.com/mw/dataset/62c2ac49913a54a66037f872)  
   - Loader: `dataset/CXR_with_split_Dataset.py`

2. **ChestXR**  
   - Source: [CXR COVID-19 Grand Challenge](https://cxr-covid19.grand-challenge.org/)  
   - Alternative download: [Google Drive](https://drive.google.com/file/d/1MRtZkovJS5iEl6W4ctkOsb1nAfTZU1r1/view?usp=drive_link)

3. **CheXpert**  
   - Source: [Kaggle CheXpert](https://www.kaggle.com/datasets/ashery/chexpert)  
   - Preprocessing:  
     - Combine the original `train.csv` and `val.csv` files.  
     - Re-split them into `train.csv` and `val.csv` using an 8:2 ratio.  
     - Use the test set provided by the [CheXlocalize](https://stanford.redivis.com/datasets/efx9-5nspnbb4b) dataset.

4. **NIH Chest X-rays 14**  
   - Source: [Kaggle NIH Chest X-ray 14 (224×224 resized)](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)  
   - Preprocessing:  
     - Split `train_val_list_NIH.txt` into `train.txt` and `val.txt` with an 8:2 ratio.

**Customization**: You can modify dataset names and loading logic in `DataManager.dataset_map` within `utils/data_manager.py`.

---

## 🚀 Running an Experiment

The following command provides a complete example for running HB-FCL across four sequential tasks:

```text
CXR → ChestXR → CheXpert → NIH14
```

### ▶️ Example

```bash
python main.py \
    --illustration "HB-FCL" \
    --group=Ours \
    --exp_name=seed_2026_beta_0_2048to1024 \
    --pre 0.2 \
    --cur 0.8 \
    --r 0.015 \
    --tse_filter_radius 0.015 \
    --dataset_names CXR ChestXR CheXpert NIH14 \
    --classes 2 1 11 8 \
    --data_paths \
        "datasets/CXR/chest_xray" \
        "datasets/CXR/CXR_Covid-19_Challenge-001" \
        "datasets/CXR/CheXpert-v1.0-small" \
        "datasets/CXR/NIH_Chest_X_ray_14" \
    --method=ours \
    --net=densenet121 \
    --num_users 5 \
    --local_bs=64 \
    --beta=0.0 \
    --com_rounds 10 10 10 10 \
    --tse_com_rounds 10 10 10 10 \
    --local_lr 0.0001 \
    --weight_decay 1e-4 \
    --seed 2026 \
    --pre_nums 2048 \
    --nums 1024 \
    --minim_generation_batch_size 16 \
    --minim_pretrained_model_path '/HuggingFace_Model/stanfordmimi/RoentGen-v2' \
    --minim_train_batch_size 8 \
    --minim_gradient_accumulation_steps 4 \
    --minim_use_ema \
    --minim_resolution 256 --minim_center_crop \
    --minim_learning_rate 5e-6 \
    --minim_lr_scheduler constant \
    --minim_lr_warmup_steps 100 \
    --minim_com_rounds 10 \
    --minim_local_epochs 2
```

### 🔧 Important Arguments

| Argument | Description |
|----------|-------------|
| `--dataset_names` | Names of datasets for each task (must match keys in `DataManager.dataset_map`) |
| `--classes` | Number of new classes introduced in each task (order corresponds to `--dataset_names`) |
| `--data_paths` | Paths to each dataset (order corresponds to `--dataset_names`) |

> ⚠️ Ensure the order of the three arguments above is consistent.

For a full list of options, run:

```bash
python main.py --help
```

---

## 📊 Results

By default, experimental results are saved under `result/` and generated artifacts (e.g., synthetic data, logs) under `run/`.