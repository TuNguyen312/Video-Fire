# GSV team
# Created date: 09/11/2024
# Created by Ryan Truong

import os
import argparse

import tensorflow as tf

from core.utils.datasets import *
from core.utils.tools import *
from core.utils.visualize import *

from core.utils.visualize import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="", help=".yaml config")
    parser.add_argument("--model", type=str, default="", help=".keras file")
    parser.add_argument("--video_path", type=str, default="", help="path to test video")

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "Please specify .yaml configuration file path"
    assert os.path.exists(opt.model), "Please specify model file path"
    assert os.path.exists(opt.video_path), "Please specify video path"

    cfg = LoadYaml(opt.yaml)

    video_frames = frames_from_video_file(
        opt.video_path,
        cfg.n_frames,
        cfg.frame_step,
        (cfg.input_width, cfg.input_height),
    )
    video_frames_ex = np.expand_dims(video_frames, axis=0)
    class_names = cfg.class_names

    model = tf.keras.models.load_model(
        opt.model,
        custom_objects={
            "Conv2Plus1D": Conv2Plus1D,
            "ResidualMain": ResidualMain,
            "Project": Project,
            "ResizeVideo": ResizeVideo,
        },
    )

    predicted = model.predict(video_frames_ex)
    predicted = int(np.round(np.squeeze(predicted)))
    print(f"Predicted: {class_names[predicted]}")

    # show_video_frames(rgba_image)
