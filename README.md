# Motion Diffusion Model

This repository provides training and sampling scripts for the Motion Diffusion Model (MDM) based on diffusion processes over human motion representations.

## Prerequisites

- **Ubuntu / Debian-based Linux**
- **CUDA-enabled GPU** (optional but recommended for training)
- **Conda** (Miniconda or Anaconda)
- **Git**

## Setup Environment

### 1. Clone the repository

```bash
git clone https://github.com/GuyTevet/motion-diffusion-model.git
cd motion-diffusion-model
```

### 2. Install FFmpeg

```bash
sudo apt update
sudo apt install ffmpeg
```

### 3. Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate mdm
```

### 4. Install additional Python dependencies

```bash
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

## Download Dependencies

Download required external assets and pretrained components:

```bash
bash prepare/download_dependencies.sh
```

## Get Data

Download the HumanML3D dataset and organize it under `dataset/`:

```bash
# Option 1: Using the provided script
bash prepare/download_datasets.sh

# Option 2: Manual clone and rename
git clone https://huggingface.co/datasets/NamYeongCho/HumanML3D_new dataset/HumanML3D
```

## Training

### Train a basic MDM

```bash
python -m train.train_mdm \
  --save_dir save/my_model \
  --dataset humanml
```

### Train with 50 diffusion steps

```bash
python -m train.train_mdm \
  --save_dir save/my_model_50Steps \
  --dataset humanml \
  --diffusion_steps 50 \
  --mask_frames \
  --use_ema
```

### Train MDM + DistilBERT decoder

```bash
python -m train.train_mdm \
  --save_dir save/my_humanml_trans_dec_bert_512 \
  --dataset humanml \
  --diffusion_steps 50 \
  --arch trans_dec \
  --text_encoder_type bert \
  --mask_frames \
  --use_ema
```

## Sampling / Generation

### Download pre-trained models

```bash
git clone https://huggingface.co/NamYeongCho/humanml_mixamo2_trans_enc_512
mv humanml_mixamo2_trans_enc_512 save/humanml_mixamo2_trans_enc_512
```

### Generate motion samples

```bash
python -m sample.generate \
  --model_path ./save/humanml_mixamo2_trans_enc_512/model000400000.pt \
  --num_samples 10 \
  --num_repetitions 3
```

## License