# LLaMA2-Interface

## Chapter 1: Download and Install

## Install locally with Text Generation WebUI

* Download and install [Anaconda](https://www.anaconda.com/download)

  * If you have free enviroment, skip this step. But if not,open anaconda prompt and create a new enviroment on each path you want :
    * ```
      (base) D:\> conda create -n textgen python=3.11.5
      ```

      you can replace your own enviroment name with textgen and python version which is installed on your system.
    * Active the enviroment you created with the command that conda give to you :

      * ```
        conda activate textgen
        ```

        However, when you want to deactivate the active enviroment :
      * ```
        conda deactivate
        ```
* Download and install [PyTorch](https://pytorch.org/get-started/locally/).

  * If your system has NVIDIA GPU, you can install version of PyTorch which is compatible with the version of [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) :
    * ```
      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
      ```
  * If your system does not have an NVIDIA GPU or you don't install a CUDA compatible version of PyTorch, PyTorch will still work, but it will run on the CPU, which is significantly slower for deep learning tasks.
* Download and install [WebUI](https://github.com/oobabooga/text-generation-webui)

  * With git which is installed on your system, get the interface:

    ```
    git clone https://github.com/oobabooga/text-generation-webui
    ```
* You have to change the directory in each drive where you downloaded the text-generation-webui, for example :

  * ```
    cd text-generation-webui
    ```
* Install all the required python modules :

  * ```
    pip install -r requirements.txt
    ```
* Now is the time to spin up the server :

  ```
  python server.py
  ```

    Conda should give you the URL, paste that on your browser for loading the interface.
