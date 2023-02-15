#!/bin/bash

# Create conda environment.
conda env create -f environment.yml
conda activate CAE

# Install additional packages.
conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install hydra-core
pip install einops

# Download datasets.
wget https://www.dropbox.com/s/hcmin7jmem7pfn8/datasets.zip
unzip datasets.zip
rm datasets.zip
