import os
import math
import matplotlib as plt
import argparse

import tensorflow as tf
from keras.utils import plot_model

# from core.model.slowfast import SlowFast
from core.model.r2plus1d import *
from core.model.c3d import *
from core.utils.tools import *
from core.utils.datasets import *
from core.utils.visualize import *


class VideoFire:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--yaml", type=str, default="", help=".yaml configs")
        parser.add_argument("--model_type", type=str, default="", help="model to train")
        parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="name of the .keras file to save",
        )

        # Check path
        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "Please specify .yaml configuration file path"
        # models = ["r2plus1d", "slowfast"]
        assert opt.model_type, "Please specify the model to train"

        self.model_type = opt.model_type
        self.model_name = opt.model_name
        # Load configs
        self.cfg = LoadYaml(opt.yaml)

        # Load dataset
        output_signature = (
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(len(self.cfg.class_names)), dtype=tf.int16),
        )

        output_size = (self.cfg.input_width, self.cfg.input_height)

        self.train_ds = tf.data.Dataset.from_generator(
            FrameGenerator(
                self.cfg.train_path,
                self.cfg.n_frames,
                self.cfg.class_names,
                frame_step=self.cfg.frame_step,
                training=True,
                output_size=output_size,
            ),
            output_signature=output_signature,
        )

        self.train_ds = self.train_ds.batch(self.cfg.batch_size)

        self.val_ds = tf.data.Dataset.from_generator(
            FrameGenerator(
                self.cfg.val_path,
                self.cfg.n_frames,
                self.cfg.class_names,
                frame_step=self.cfg.frame_step,
                output_size=output_size,
            ),
            output_signature=output_signature,
        )
        self.val_ds = self.val_ds.batch(self.cfg.batch_size)

        self.test_ds = tf.data.Dataset.from_generator(
            FrameGenerator(
                self.cfg.test_path,
                self.cfg.n_frames,
                self.cfg.class_names,
                frame_step=self.cfg.frame_step,
                output_size=output_size,
            ),
            output_signature=output_signature,
        )

        self.test_ds = self.test_ds.batch(self.cfg.batch_size)

        # Initialize model
        if self.model_type == "c3d":
            self.model = C3D(
                self.cfg.n_frames,
                self.cfg.input_height,
                self.cfg.input_width,
                len(self.cfg.class_names),
            ).model
        if self.model_type == "r2plus1d-18" or self.model_type == "r2plus1d-34":
            self.model = R2Plus1D(
                self.model_type,
                self.cfg.n_frames,
                self.cfg.input_height,
                self.cfg.input_width,
                len(self.cfg.class_names),
            ).model
        # if self.model_name == "slowfast":
        #     self.model = SlowFast()
        # self.model = self.model.build_graph((240, 240, 4))
        self.model.summary()
        plot_model(
            self.model, to_file=f"images/{self.model_name}.png", show_shapes=True
        )

    def train(self):

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)
        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        # Train model
        self.history = self.model.fit(
            x=self.train_ds, epochs=20, validation_data=self.val_ds
        )

        # Plot history
        plot_history(self.history)

        # # Save model
        save_name = self.model_name if self.model_name else self.model_type
        self.model.save(f"models/{save_name}.keras")
        print("Model saved")


if __name__ == "__main__":
    video_fire = VideoFire()
    video_fire.train()
