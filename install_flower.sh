#!/bin/bash
git clone https://github.com/adap/flower.git
cd flower
git checkout d1eb90f74714a9c10ddbeefb767b56be7b61303d
pip install --upgrade pip
cp ../flower.patch ./flower.patch
git apply --whitespace=fix flower.patch
pip install .
pip install torch torchvision tensorflow scikit-learn tqdm uvicorn fastapi opencv-python-headless pytest
rm -rf ~/.cache/pip
cd ..