# FPDRANet: Experimental First-Photon Visualization of Quantum Erasure With Hybrid Entanglement

This repository contains the official PyTorch implementation of the paper **"Experimental First-Photon Visualization of Quantum Erasure With Hybrid Entanglement"**, published in *Laser & Photonics Reviews (2025)*.

## 📄 Overview
We propose a **First-Photon Physics-enhanced Dual-branch Residual Attention Network (FPDRANet)** to visualize quantum erasure phenomena using polarization-orbital angular momentum (OAM) two-photon hybrid entanglement. Unlike traditional methods requiring exhaustive mode scanning or high photon flux, our approach reconstructs high-fidelity OAM interference patterns (petal-like structures) using **fewer than one photon per pixel**.

The network leverages 4D data cubes (spatial + time-of-flight histograms) to enforce physical constraints, significantly outperforming conventional first-photon algorithms like SPIRAL-TAP in low-light quantum imaging scenarios.

## 📂 Code Structure
- **`FPDRANet.py`**: The architecture of FPDRANet, including the Residual Attention Block (RAB) and Hybrid Dense Residual Attention Block (HDRAB).
- **`Train.py`**: Main training script for the network.
- **`Test.py`**: Script for testing and evaluating the model with pre-trained weights.
- **`preprocess_image.m` & `spiral/`**: MATLAB scripts used for data preprocessing and baseline comparisons (SPIRAL-TAP).

## 🔗 Citation
If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{Yu2025FPDRANet,
  title={Experimental First-Photon Visualization of Quantum Erasure With Hybrid Entanglement},
  author={Yu, Wen-Kai and Wu, Qing-Yuan and Chen, Xiao-Xiao and Huo, Juan and Li, Jian and Yang, Jia-Zhi and Zhang, An-Ning},
  journal={Laser \& Photonics Reviews},
  year={2025},
  doi={10.1002/lpor.202501816}
}
