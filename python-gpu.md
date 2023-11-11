# using gpu for python

## Base installision

### 1. Check GPU Compatibility

- Ensure your NVIDIA GPU is CUDA-compatible. You can check this on NVIDIA's official website.

### 2. Install Anaconda

- Download and install Anaconda from its [official website](https://www.anaconda.com/products/distribution).
- After installation, open Anaconda Navigator.

### 3. Create a New Conda Environment

- In Anaconda Navigator, go to "Environments" and create a new environment.
- Choose Python as the base language. Select the Python version compatible with the PyTorch version you plan to use.

### 4. Install CUDA Toolkit

- Download and install the CUDA Toolkit from NVIDIA's [official site](https://developer.nvidia.com/cuda-downloads).
- Ensure the CUDA version is compatible with the PyTorch version you intend to use.

### 5. Install CuDNN

**Steps to Install cuDNN for CUDA 12.1**

1. **NVIDIA Developer Account**: If you don't already have an NVIDIA Developer account, you need to create one. Go to the [NVIDIA Developer website](https://developer.nvidia.com/) and sign up or log in.
2. **Download cuDNN**:

   - Go to the [cuDNN Download Page](https://developer.nvidia.com/cudnn).
   - Under the section of **cuDNN Archive**, find the version compatible with CUDA 12.1.
   - Click on the link and choose the version for Windows (x64).
3. **Accept the Terms and Download**:

   - Click on the download link after accepting the terms of the license agreement.
   - Usually, you would download the cuDNN Library for Windows (x64).
4. **Extract the cuDNN Package**:

   - Once downloaded, extract the contents of the cuDNN zip file.
5. **Install cuDNN**:

   - Copy the following files from the extracted folder to the CUDA Toolkit directory (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`):
     - `bin\*.dll` to `<CUDA_PATH>\bin`
     - `include\cudnn*.h` to `<CUDA_PATH>\include`
     - `lib\x64\cudnn*.lib` to `<CUDA_PATH>\lib\x64`
   - Replace `<CUDA_PATH>` with your actual CUDA Toolkit installation path.
6. **Set Environment Variables**:

   - Add the path to the CUDA binaries to your system's PATH environment variable, if not already present:
     - Right-click on 'This PC' or 'My Computer', and select 'Properties'.
     - Click on 'Advanced system settings' and then 'Environment Variables'.
     - Under 'System Variables', find and select 'Path', then click 'Edit'.
     - Add the path `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin` and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp`.
   - Click OK to close all dialogs.
7. **Verify Installation**:

   - Open Command Prompt and run the command:
     ```cmd
     nvidia-smi
     ```
   - This command should display your GPU information if everything is installed correctly.

### 6. Install PyTorch with GPU Support

- Activate your new environment in Anaconda.
- Install PyTorch with GPU support. Use the installation command provided on the [PyTorch official website](https://pytorch.org/get-started/locally/), selecting the appropriate CUDA version.

### 7. Install Jupyter Notebook

- Inside your conda environment, install Jupyter Notebook:
  ```bash
  conda install -c anaconda jupyter
  ```

### 8. Launch Jupyter Notebook

- After installation, launch Jupyter Notebook from Anaconda Navigator or by using the command line:
  ```bash
  jupyter notebook
  ```

### 9. Verify GPU Usage

- In your Jupyter Notebook, you can verify that PyTorch is using the GPU:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

### 10. Run Your Python Files

- You can now open your Python files in the Jupyter Notebook and run them. PyTorch will automatically utilize the GPU if your code is written to do so.

### Additional Tips:

- Always keep your GPU drivers up to date.
- Regularly update Anaconda, Python, PyTorch, CUDA, and CuDNN to their latest versions for better performance and compatibility.
- Refer to PyTorch's official documentation for specific GPU operations and optimizations.

By following these steps, you should be able to run your Python files using PyTorch with GPU support in a Jupyter Notebook environment on your Windows system with an NVIDIA GPU.
