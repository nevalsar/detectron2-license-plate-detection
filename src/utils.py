from curses import meta
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2 import model_zoo

import random
import cv2
import matplotlib.pyplot as plt
import torch


def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        viz = Visualizer(
            img[:, :, ::-1], metadata=dataset_custom_metadata
        )  # BGR to RGB format conversion
        viz = viz.draw_dataset_dict(s)
        plt.figure(figsize=(15, 20))
        plt.imshow(viz.get_image())
        plt.show()


def get_train_cfg(
    config_file_path,
    checkpoint_url,
    train_dataset_name,
    test_dataset_name,
    num_classes,
    device,
    output_dir,
):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg


def on_image(image_path, predictor):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    viz = Visualizer(
        im[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION
    )
    viz = viz.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(14, 10))
    plt.imshow(viz.get_image())
    plt.show()


def on_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() == False:
        print("Error opening file..")
        return

    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        viz = Visualizer(
            image[:, :, ::-1], metadata={}, instance_mode=ColorMode.SEGMENTATION
        )
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:, :, ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()


def check_cuda():
    if torch.cuda.is_available():
        print("Cuda available!")
    else:
        raise Exception("Cuda not available.")
