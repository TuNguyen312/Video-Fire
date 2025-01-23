import yaml
from pathlib import Path
import tensorflow as tf


class LoadYaml:
    def __init__(self, yaml_path):
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.train_path = data["DATASET"]["TRAIN"]
        self.val_path = data["DATASET"]["VAL"]
        self.test_path = data["DATASET"]["TEST"]
        self.class_names = data["DATASET"]["CLASS_NAMES"]

        self.n_frames = data["MODEL"]["N_FRAMES"]
        self.frame_step = data["MODEL"]["FRAME_STEP"]
        self.input_width = data["MODEL"]["INPUT_WIDTH"]
        self.input_height = data["MODEL"]["INPUT_HEIGHT"]

        self.learning_rate = data["TRAIN"]["LR"]
        self.batch_size = data["TRAIN"]["BATCH_SIZE"]
        self.num_epochs = data["TRAIN"]["NUM_EPOCHS"]

        print("Load yaml successfully...")


def get_actual_predicted_labels(dataset, model):
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    predicted = predicted.round()

    return actual, predicted


# def convert_to_tflite(model):
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     tflite_model = converter.convert()
#     return tflite_model


# if __name__ == "__main__":
#     data = LoadYaml("/home/tunguyen/ESPFire/configs/configs.yaml")
#     print(data.classes)
