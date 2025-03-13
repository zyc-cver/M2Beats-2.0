<p align="center">
  <h1>M2Beats 2.0: When Motion Meets Beats in Short-form Videos Twice</h1>
</p>

<p align="center">
  <a href="https://github.com/zyc-cver/M2Beats-2.0/tree/main">
    <img src="https://img.shields.io/badge/Project-Repo-orange?logo=git" alt="Project">
  </a>
</p>

## Installation

### 1. Create Conda Environment
```bash
conda create -n m2beats2 python=3.9 -y
conda activate m2beats2
```

### 2. Install Required Packages
```bash
pip install numpy==1.24.0 torch==1.12.1
```

## Model
Download our pretrained model from [Google Drive](https://drive.google.com/your_model_link).

## Dataset
Download our dataset from [Google Drive](https://drive.google.com/your_dataset_link).

The dataset follows the directory structure below:
```bash
AIST_M2B_2/
  ├── data_split/
  ├── test_data_split.csv
  ├── train_data_split.csv
```
Original video and motion data can be downloaded from [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html)

## Evaluation
Run the following command to evaluate the model:
```bash
python eval.py --checkpoint checkpoints/model.pth --test AIST_M2B_2/train_data_split.csv --test_data AIST_M2B_2/data_split
```

## Test
### Testing on Sample Data from the Dataset
```bash
python test.py --checkpoint checkpoints/model.pth --test AIST_M2B_2/train_data_split.csv --test_data AIST_M2B_2/data_split
```

### Testing on Your Own Data
If you want to test on your own data, you need to extract 2D human keypoints from video data beforehand. We recommend using [mmpose](https://github.com/open-mmlab/mmpose).

---

More details will be available once our paper is published!

