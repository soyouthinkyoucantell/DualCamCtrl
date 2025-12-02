
#  📷DualCamCtrl: Dual-Branch Diffusion Model for Geometry-Aware Camera-Controlled Video Generation

[![Page](https://img.shields.io/badge/github-Project_page-blue?logo=github)](https://soyouthinkyoucantell.github.io/dualcamctrl-page/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](#)
[![Ckpt](https://img.shields.io/badge/🤗%20HuggingFace-Checkpoint%20-yellow)](https://huggingface.co/FayeHongfeiZhang/DualCamCtrl/tree/main)


[Hongfei Zhang](#) <sup>1*</sup>, [Kanghao Chen](https://khao123.github.io/) <sup>1,5*</sup>,  [Zixin Zhang](https://scholar.google.com/citations?user=BbZ0mwoAAAAJ&hl=en) <sup>1,5</sup>, [Harold H. Chen](https://haroldchen19.github.io/) <sup>1,5</sup>, 
  [Yuanhuiyi Lyu](https://qc-ly.github.io/) <sup>1</sup>,  [Yuqi Zhang](#) <sup>3</sup>,  [Shuai Yang](https://andysonys.github.io/) <sup>1</sup>,  [Kun Zhou](https://redrock303.github.io/) <sup>4</sup>,  [Ying-Cong Chen](https://www.yingcong.me/) <sup>1,2,✉</sup> 

<sup>1</sup> HKUST(GZ)  <sup>2</sup> HKUST  <sup>3</sup> Fudan University  <sup>4</sup> Shenzhen University  <sup>5</sup> Knowin


\* Equal Contribution. ✉Corresponding author.

---

https://github.com/user-attachments/assets/e6a1ff6c-f74f-4c7f-9ad1-66ea7b65353a

---


## 🧩 Contents

<!-- #### 1. [📰 News](#-news)
####  2. [⚙️ TODO](#⚙️-todo)
####  3. [🎯 Overview](#🎯-overview)
####  4. [🔧 Installation](#🔧-installation)
####  5. [🔮 Inference](#🔮-inference)
####  6. [✨ Gradio](#✨-gradio)
####  7. [🔥 Training](#🔥-training) -->

#### 1. [📰 News](#1-news)
#### 2. [⚙️ TODO](#2-todo)
#### 3. [🎯 Overview](#3-overview)
#### 4. [🔧 Installation](#4-installation)
#### 5. [🔮 Inference](#5-inference)
#### 6. [✨ Gradio](#6-gradio)
#### 7. [🔥 Training](#7-training)

## 📰 News

✅ 2025.11 — Released inference pipeline & demo dataset ✔️

✅ 2025.11 — Uploaded official DualCamCtrl checkpoints to HuggingFace 🔑

## ⚙️ TODO
⬜ Release the training code 🚀

## 🎯 Overview

### Abstract
This paper presents **DualCamCtrl**, a novel end-to-end diffusion model for camera-controlled video generation. Recent works have advanced this field by representing camera poses as ray-based conditions, yet they often lack sufficient scene understanding and geometric awareness.
**DualCamCtrl** specifically targets this limitation by introducing a dual-branch framework that mutually generates camera-consistent RGB and depth sequences.
To harmonize these two modalities, we further propose the **S**emant**I**c **G**uided **M**utual **A**lignment (SIGMA) mechanism, which performs RGB–depth fusion in a semantics-guided and mutually reinforced manner.
These designs collectively enable **DualCamCtrl** to better disentangle appearance and geometry modeling, generating videos that more faithfully adhere to the specified camera trajectories. Extensive experiments demonstrate that **DualCamCtrl** achieves more consistent camera-controlled video generation **with over 40% reduction** on camera motion errors compared with prior methods.


### Results

![I2V Quantitative Comparison](assets/i2vcompare_cropped.jpg)  
*Comparison between our method and other state-of-the-art approaches. Given the same camera pose and input image as generation conditions, our method achieves the best alignment between camera motion and scene dynamics, producing the most visually accurate video. The ’+’ signs marked in the figure serve as anchors for visual comparison.*


![T2V Quantitative Comparison](assets/i2v.png)  
*Quantitative comparisons on **I2V** setting. ↑ / ↓ denotes higher/lower is better. Best and second best results highlighted.*


![I2V/T2V Comparison](assets/t2v.png)  
*Quantitative comparisons on **T2V** setting across REALESTATE10K and DL3DV.*


## 🔧 Installation

#### Clone repo and create an enviroment with Python 3.11:

```
git clone https://github.com/soyouthinkyoucantell/DualCamCtrl.git
conda create -n dualcamctrl python=3.11 -y
conda activate dualcamctrl
```
#### Install DiffSynth-Studio dependencies from source code:

```
cd DualCamCtrl
pip install -e .
```

#### Then install GenFusion dependencies:
```
mkdir dependency
cd dependency
git clone https://github.com/rmbrualla/pycolmap.git 
cd pycolmap
pip install -e .
pip install numpy==1.26.4 peft accelerate==1.9.0 decord==0.6.0 deepspeed diffusers omegaconf  
```




## 🔮 Inference


### Checkpoints

Get the checkpoints from the HuggingFace repo:&nbsp; [DualCamCtrl Checkpoints](https://huggingface.co/FayeHongfeiZhang/DualCamCtrl)

#### Put it the checkpoints dir 
```
cd ../.. # make sure you are at the root dir 
```

Your project structure should be like

```
DualCamCtrl/
├── checkpoints/                 # ← Put downloaded .pt here
│   └── dualcamctrl_diffusion_transformer.pt
├── demo_dataset/                # Small demo dataset strcture
├── demo_pic/                    # Demo images for quick inference
├── diffsynth/                   
├── examples/   
├── ....                 
├── requirements.txt             
├── README.md                    
└── setup.py                    
```

#### Test with our demo pictures and depth:
```
cd .. # make sure you are at the root dir 
export PYTHONPATH=.
python -m test_script.test_demo
```


## ✨ Gradio 

Install gradio dependency (needs large memory GPU)
```
pip install gradio
```

Run app

```
export PYTHONPATH=.
python gradio/app.py # For Large Memory GPU
```

## 🔥 Training

### Training details coming soon… Stay tuned! 🚀