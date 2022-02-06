# detectron2-license-plate-detection

Detect car license plates using [detectron2](https://github.com/facebookresearch/detectron2). The model is trained on a custom dataset of license plates created by hand annotating image search results.

This repository contains the full dataset and annotations, along with the source code for training and testing.

## Usage

- [Optional] Create new virtual environment using venv / virtualenv.
- Install required packages using pip.

``` shell
pip install -r requirements.txt
```
- Train

``` shell
cd src
python train.py
```
- Test

``` shell
python test.py
```

