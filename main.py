import os
import json
import sys
import tensorflow as tf
import numpy as np
import argparse

print(sys.argv[2])

# Load configuration from the first command-line argument
with open(sys.argv[2], 'r') as fh:
    cfg = json.load(fh)

OUTPUT_DIR = cfg['output_dir']
LOGDIR = os.path.join(OUTPUT_DIR, "log")

os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cfg['gpu'])

from model_HoloGAN import HoloGAN
from tools.utils import pp, show_all_variables

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--input_height", type=int, default=108, help="The size of image to use (will be center cropped). [108] or [128] for celebA and lsun, [400] for chairs. Cats and Cars are already cropped")
    parser.add_argument("--input_width", type=int, default=None, help="The size of image to use (will be center cropped). If None, same value as input_height [None]")
    parser.add_argument("--output_height", type=int, default=64, help="The size of the output images to produce 64 or 128")
    parser.add_argument("--output_width", type=int, default=None, help="The size of the output images to produce. If None, same value as output_height [None]")
    parser.add_argument("--dataset", type=str, default="celebA", help="The name of dataset [celebA, lsun, chairs, shoes, cars, cats]")
    parser.add_argument("--input_fname_pattern", type=str, default="*.jpg", help="Glob pattern of filename of input images [*]")
    parser.add_argument("--train_size", type=float, default=np.inf, help="Number of images to train-Useful when only a subset of the dataset is needed to train the model")
    parser.add_argument("--crop", type=bool, default=True, help="True for training, False for testing [False]")
    parser.add_argument("--train", type=bool, default=True, help="True for training, False for testing [False]")
    parser.add_argument("--rotate_azimuth", type=bool, default=False, help="Sample images with varying azimuth")
    parser.add_argument("--rotate_elevation", type=bool, default=False, help="Sample images with varying elevation")
    args = parser.parse_args()
    return args

def main(args):
    pp.pprint(vars(args))
    if args.input_width is None:
        args.input_width = args.input_height
    if args.output_width is None:
        args.output_width = args.output_height
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True
    print("FLAGS " + str(args.dataset))
    with tf.compat.v1.Session(config=run_config) as sess:
        model = HoloGAN(
            sess,
            input_width=args.input_width,
            input_height=args.input_height,
            output_width=args.output_width,
            output_height=args.output_height,
            dataset_name=args.dataset,
            input_fname_pattern=args.input_fname_pattern,
            crop=args.crop)

        model.build(cfg['build_func'])

        show_all_variables()

        if args.train:
            train_func = eval("model." + (cfg['train_func']))
            train_func(args)
        else:
            if not model.load(LOGDIR)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            model.sample_HoloGAN(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
