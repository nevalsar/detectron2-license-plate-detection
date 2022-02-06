from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

dataset_root = "../license-plate-dataset"
output_root = "../output"

cfg_save_path = output_root + "/object_detection/od_cfg.pickle"
with open(cfg_save_path, "rb") as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = dataset_root + "/test/6.license-plate.jpg"
video_path = dataset_root + "/test/highway.mp4"

on_image(image_path, predictor)
on_video(video_path, predictor)
