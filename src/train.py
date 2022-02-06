from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle
from utils import *

dataset_root = "../license-plate-dataset"
output_root = "../output"

config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

output_dir = output_root + "/object_detection"
num_classes = 1

device = "cuda"

train_dataset_name = "LP_train"
train_images_path = dataset_root + "/train"
train_json_annotation_path = dataset_root + "/train.json"

test_dataset_name = "LP_test"
test_images_path = dataset_root + "/test"
test_json_annotation_path = dataset_root + "/test.json"

cfg_save_path = output_dir + "/od_cfg.pickle"

################################################################################

# Register dataset
register_coco_instances(
    name=train_dataset_name,
    metadata={},
    json_file=train_json_annotation_path,
    image_root=train_images_path,
)

register_coco_instances(
    name=test_dataset_name,
    metadata={},
    json_file=test_json_annotation_path,
    image_root=test_images_path,
)

# plot_samples(dataset_name=train_dataset_name, n=2)

################################################################################


def main():
    cfg = get_train_cfg(
        config_file_path,
        checkpoint_url,
        train_dataset_name,
        test_dataset_name,
        num_classes,
        device,
        output_dir,
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
