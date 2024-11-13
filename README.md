# Machine-Learning
Machine learning examples

## Installation on Windows

0. Install [Python 3.11.8](https://www.python.org/downloads/release/python-3118/). Add an alias script to `C:\scripts\python3.bat`:
```PowerShell
@echo off
C:\Users\{YOUR USERNAME}\AppData\Local\Programs\Python\Python311\python.exe %*
```
Include path to system or user environment variables.
1. Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network). To verify CUDA was successfully installed,
```PowerShell
nvcc --version
```
2. Verify GPU drivers are set up: `nvidia-smi`
3. Install essential packages for development: `pip3 install numpy pandas scikit-learn matplotlib seaborn jupyter tensorflow keras torch torchvision scipy plotly opencv-python notebook`
4. After CUDA installation: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
4. Verify PyTorch can recognize GPU: 
```python
import torch 
torch.cuda.is_available()
```