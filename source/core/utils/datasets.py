import math
import random
import os
import cv2
import tensorflow as tf
import numpy as np
import pathlib


def get_files_and_class_names(path):
    video_paths = []
    types = ("*/*.mp4", "*/*.avi")
    for type in types:
        video_paths.extend(path.glob(type))
    classes = [p.parent.name for p in video_paths]
    return video_paths, classes


def format_frames(path, frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    if frame is None:
        raise ValueError(path)
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames, frame_step, output_size=(224, 224)):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []  # show_video_frames(rgba_image)

    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, int(max_start) + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(video_path, frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(video_path, frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    # note: debug
    # if len(result) < n_frames:
    #     new_result = [] * n_frames
    #     step = len(result) / n_frames
    #     curr_step = step
    #     for i in range(n_frames):
    #         new_result[i] = result[math.floor(curr_step)]
    #         curr_step += step
    #     result = new_result
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


class FrameGenerator:
    def __init__(
        self,
        path,
        n_frames,
        class_names,
        frame_step=15,
        training=False,
        output_size=(224, 224),
    ):
        """Returns a set of frames with their associated label.

        Args:
          path: Video file paths.
          n_frames: Number of frames.
          training: Boolean to determine if training dataset is being created.
        """
        self.path = pathlib.Path(path)
        self.n_frames = n_frames
        self.frame_step = frame_step
        self.training = training
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.output_size = output_size
        self.class_ids_for_name = dict(
            (name, idx) for idx, name in enumerate(self.class_names)
        )

    def __call__(self):
        video_paths, classes = get_files_and_class_names(self.path)

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(
                path, self.n_frames, self.frame_step, self.output_size
            )
            label = self.class_ids_for_name[name]  # Encode labels
            encoding = np.zeros(self.num_classes)
            encoding[label] = 1
            yield video_frames, encoding
