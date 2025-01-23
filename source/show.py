import argparse
import os
import pathlib
from random import randrange

from core.utils.tools import *
from core.utils.datasets import *
from core.utils.visualize import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="", help=".yaml configs")
    parser.add_argument("--path", type=str, default="", help="data path")
    # parser.add_argument("--single", dest="is_single", action="store_true")
    # parser.add_argument("--multiple", dest="is_single", action="store_false")
    # parser.add_argument("--show_frames", action="store_true")
    # parser.set_defaults(is_single=True)
    parser.add_argument(
        "--show_frames",
        type=str,
        default=False,
        help="show input frames extract from video",
    )
    opt = parser.parse_args()

    # Check path
    assert os.path.exists(opt.yaml), "Please specify .yaml configuration file path"
    assert os.path.exists(opt.path), "Please specify data file path"

    # Load configs
    cfg = LoadYaml(opt.yaml)

    path = pathlib.Path(opt.path)
    video_paths, classes = get_files_and_class_names(path)
    # if opt.is_single:
    #     l = 1
    # else:
    #     l = len(video_paths)
    # for i in range(1, l):
    idx = randrange(0, len(video_paths))
    frames = frames_from_video_file(
        video_paths[idx],
        cfg.n_frames,
        cfg.frame_step,
        (cfg.input_width, cfg.input_height),
    )
    show_video_frames(frames, classes[idx])
