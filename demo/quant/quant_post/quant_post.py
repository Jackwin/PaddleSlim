import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np
import paddle.fluid as fluid

sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
from paddleslim.common import get_logger
from paddleslim.quant import quant_post
from utility import add_arguments, print_arguments
#import imagenet_reader as reader
import reader_cv2 as reader
_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  16,                 "Minibatch size.")
add_arg('batch_num',        int,  5,               "Batch number")
add_arg('image_shape',      str, "3,608,608",       "Input image size")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model_path',       str,  "./inference_model/MobileNet/",  "model dir")
add_arg('save_path',        str,  "./quant_model/MobileNet/",  "model dir to save quanted model")
add_arg('model_filename',       str, None,                 "model file name")
add_arg('params_filename',      str, None,                 "params file name")
add_arg('resize_short_size', int, 256,                      "Set resize short size")
add_arg('activation_quantize_type', str,"range_abs_max", "quantize type")
add_arg('weight_quantize_type', str, "channel_wise_abs_max", "quantize type")
## reader
#add_arg('data_dim',       int,  224,                 "data dimension")
#add_arg('data_dir',       str,  "../../data/果蔬23_0910",  "data dir")
#add_arg('image_mean',      str, "0.485,0.456,0.406",       "Input image mean")
#add_arg('image_std',      str, "0.229,0.224,0.225",       "Input image standards")
#add_arg('thread_num',       int,  4,                 "thread number")
#add_arg('buf_size',       int,  224,                 "buffer size")

add_arg('use_mixup',      bool,      False,        "Whether to use mixup or not")
add_arg('mixup_alpha',      float,     0.2,      "Set the mixup_alpha parameter")
add_arg('l2_decay',         float, 1e-4,                 "L2_decay parameter.")
add_arg('momentum_rate',    float, 0.9,                  "momentum_rate.")
add_arg('use_label_smoothing',      bool,      False,        "Whether to use label_smoothing or not")
add_arg('label_smoothing_epsilon',      float,     0.2,      "Set the label_smoothing_epsilon parameter")
add_arg('lower_scale',      float,     0.08,      "Set the lower_scale in ramdom_crop")
add_arg('lower_ratio',      float,     3./4.,      "Set the lower_ratio in ramdom_crop")
add_arg('upper_ratio',      float,     4./3.,      "Set the upper_ratio in ramdom_crop")
# yapf: enable


def quantize(args):
    #val_reader = reader.train()
    #val_reader = paddle.batch(reader.train(settings=args), batch_size=args.batch_size)
    val_reader = reader.train_yolov3(settings=args)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    assert os.path.exists(args.model_path), "args.model_path doesn't exist"
    assert os.path.isdir(args.model_path), "args.model_path must be a dir"

    exe = fluid.Executor(place)
    #[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=args.model_path,
    #                                                                                      model_filename='model',
    #                                                                                      params_filename='params',
    #                                                                                      executor=exe)

    
    #for img in val_reader():
     #   print('-------', img[0].shape)
     #   print ('-----', img)
     #   print(feed_target_names)
        #[features] = exe.run(inference_program, fetch_list=fetch_targets, feed={feed_target_names[0]:img[0], feed_target_names[1]:np.array([[1,1]))
        #exe.run()
    #return 
    quant_post(
        executor=exe,
        model_dir=args.model_path,
        quantize_model_path=args.save_path,
        sample_generator=val_reader,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        batch_size=args.batch_size,
        batch_nums=args.batch_num,
        activation_quantize_type= args.activation_quantize_type,
        weight_quantize_type=args.weight_quantize_type)


def main():
    args = parser.parse_args()
    print_arguments(args)
    quantize(args)


if __name__ == '__main__':
    main()
