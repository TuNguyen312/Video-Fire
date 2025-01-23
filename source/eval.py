# GSV team
# Created date: 09/11/2024
# Created by Ryan Truong

import os
import argparse

import tensorflow as tf

from core.utils.datasets import *
from core.utils.tools import *
from core.utils.visualize import *

from core.model.r2plus1d import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="", help=".yaml config")
    parser.add_argument("--model", type=str, default=None, help=".keras file")

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "Please specify .yaml configuration file path"
    assert os.path.exists(opt.model), "Please specify model file path"

    cfg = LoadYaml(opt.yaml)

    output_signature = (
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(len(cfg.class_names)), dtype=tf.int16),
    )

    output_size = (cfg.input_width, cfg.input_height)

    test_ds = tf.data.Dataset.from_generator(
        FrameGenerator(
            cfg.test_path,
            cfg.n_frames,
            cfg.class_names,
            frame_step=cfg.frame_step,
            output_size=output_size,
        ),
        output_signature=output_signature,
    )

    test_ds = test_ds.batch(cfg.batch_size)

    model = tf.keras.models.load_model(
        opt.model,
        custom_objects={
            "Conv2Plus1D": Conv2Plus1D,
            "ResidualMain": ResidualMain,
            "Project": Project,
            "ResizeVideo": ResizeVideo,
        },
    )

    # Evaluate on test set
    eval = model.evaluate(test_ds, return_dict=True)
    print(
        f"Evaluate on the test set:\n accuracy: {eval['accuracy']:.2f}\n loss: {eval['loss']:.2f}"
    )

    actual, predicted = get_actual_predicted_labels(test_ds, model)
    plot_confusion_matrix(actual, predicted, cfg.class_names)
