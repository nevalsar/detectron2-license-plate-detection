# detectron2-license-plate-detection

Detect car license plates using [PyTorch](https://pytorch.org/) and [detectron2](https://github.com/facebookresearch/detectron2). The model is trained on a custom dataset of license plates created by hand annotating image search results.

This repository contains the full dataset and annotations, along with the source code for training and testing.

## Usage

- [Optional] Create new virtual environment using [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv).

``` shell
pyenv virtualenv detectron2-license-plate-detection-env
pyenv activate detectron2-license-plate-detection-env
```

- Install required packages using pip.

``` shell
pip install -r requirements.txt
```

- Run training script.
``` shell
cd src
python train.py
```

- Run testing script.
The model is now run on a test image and a test video and  bounding box is drawn around detected license plates along with confidence value. 
``` shell
python test.py
```

