<div align="center">
<h1> RoDeCon-Net: Medical Image Segmentation via Robust Decoupling and Contrast-Enhanced Fusion </h1>
</div>

## 🎈 News

[2025.7.11] **We are accepted <-- We are here !**

[2025.6.16] **We are conducting rebuttal <-- We are here !**

[2025.3.1] Training and testing code released.



## ⭐ Abstract

Medical image segmentation is crucial for clinical decision-making, treatment planning, and disease tracking. Nonetheless, it confronts two significant challenges: the presence of "soft boundaries" between the foreground and background exacerbated by poor illumination and low contrast, and the misleading co-occurrence of salient and non-salient objects during the training phase, which complicates the model's accuracy in distinguishing relevant features. To overcome these challenges, we introduce RoDeCon-Net, a novel framework engineered to enhance medical image segmentation. RoDeCon-Net incorporates a Feature Decoupling Unit (FDU) that dynamically separates encoded features into foreground, background, and uncertain regions, using advanced attention mechanisms to refine feature distinction and reduce uncertainty. Additionally, our Contrast-driven Feature Alignment Unit (CFAU) and Cross-layer Feature Cascade Unit (CFCU) synergize to reinforce feature contrasts and promote effective multi-level feature fusion, thus improving the detection of salient objects amidst complex backgrounds and handling various object scales within images. Comprehensive evaluations of RoDeCon-Net on five diverse medical image datasets validate its superior performance and versatility, showcasing its potential to set new benchmarks in medical image segmentation.

## 🚀 Introduction

<div align="center">
    <img width="400" alt="image" src="figures/challenge.png?raw=true">
</div>

The challenges: (a) the effects of fuzzy edges (soft boundaries). (b) The misleading co-occurrence of salient and non-salient objects.

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="figures/network.png?raw=true">
</div>

Illustration of the overall architecture.


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n Net python=3.8
conda activate Net
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

- Download datasets: ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018), Kvasir from this[link](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip), BUSI from this [link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), Moun-Seg from this [link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018), and COVID-19 from this [link](https://drive.usercontent.google.com/download?id=1FHx0Cqkq9iYjEMN3Ldm9FnZ4Vr1u3p-j&export=download&authuser=0).


- Folder organization: put datasets into ./data/datasets folder.

### 3. Train the Net

```
python train.py --datasets ISIC2018
training records is saved to ./log/dataset folder
pre-training file is saved to ./log/dataset folder
concrete information see train.py, please
```

### 3. Test the Net

```
python test.py --datasets ISIC2018
training records is saved to ./log/dataset folder
pre-training file is saved to ./log/dataset folder
concrete information see test.py, please
```

### 3. Code example

```
python Test/test_example.py
```

## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="figures/com_pic.png?raw=true">
</div>

<div align="center">
We compare our method against 13 state-of-the-art methods. The red box indicates the area of incorrect predictions.
</div>

## ✨ Quantitative comparison

<div align="center">
<img width="800" alt="image" src="figures/com_tab.png?raw=true">
</div>

<div align="center">
Performance comparison with ten SOTA methods on ISIC2018, Kvasir, BUSI, COVID-19 and Monu-Seg datasets.
</div>

# ⭐ Rebuttal

## 🖼️ Visualization of Ablation Results

<div align="center">
<img width="800" alt="image" src="figures/aba.png?raw=true">
</div>

**Response:** We conduct ablation experiments and present the intermediate visualizations of foreground, background, and uncertain regions from Stage I. Additionally, we provide the intermediate visualizations of the encoder from Stage I, as well as those from the CFAU and CFCU components.


## 🖼️ Convergence Analysis

<div align="center">
<img width="800" alt="image" src="figures/curve.png?raw=true">
</div>

**Response:** We evaluated the effects of the loss functions $\lambda_f$, $\lambda_b$, and $\lambda_u$ on the stability of the model. F, B, and U denote $\lambda_f$, $\lambda_b$, and $\lambda_u$, respectively.



## 🎫 License

The content of this project itself is licensed under [LICENSE](https://github.com/ILoveACM-MM/RoDeCon-Net?tab=Apache-2.0-1-ov-file).
