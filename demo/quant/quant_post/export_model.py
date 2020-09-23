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
import models
from utility import add_arguments, print_arguments
import models.inception_v4 as Model

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('pretrained_model', str,  "../pretrained_model/MobileNetV1_pretained",                "Whether to use pretrained model.")
add_arg('data',             str, "mnist",                 "Which data to use. 'mnist' or 'imagenet'")
add_arg('test_period',      int, 10,                 "Test period in epoches.")

## reader_cv2 eval
add_arg('image_shape',      str, "3,224,224",       "Input image size")
add_arg('params_filename',      str, None,                 "params file name")
add_arg('resize_short_size', int, 256,                      "Set resize short size")


## reader_cv2 train
add_arg('batch_size',       int,   128,                  "Minibatch size.")
add_arg('total_images',     int,   16235,              "Training image number.")
add_arg('use_mixup',      bool,      False,        "Whether to use mixup or not")
add_arg('mixup_alpha',      float,     0.2,      "Set the mixup_alpha parameter")
add_arg('l2_decay',         float, 1e-4,                 "L2_decay parameter.")
add_arg('momentum_rate',    float, 0.9,                  "momentum_rate.")
add_arg('use_label_smoothing',      bool,      False,        "Whether to use label_smoothing or not")
add_arg('label_smoothing_epsilon',      float,     0.2,      "Set the label_smoothing_epsilon parameter")
add_arg('lower_scale',      float,     0.08,      "Set the lower_scale in ramdom_crop")
add_arg('lower_ratio',      float,     3./4.,      "Set the lower_ratio in ramdom_crop")
add_arg('upper_ratio',      float,     4./3.,      "Set the upper_ratio in ramdom_crop")
add_arg('num_epochs',       int,   50,                  "number of epochs.")
add_arg('class_dim',        int,   23,                 "Class number.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def export_model(args):
    if args.data == "mnist":
        import paddle.dataset.mnist as reader
        train_reader = reader.train()
        val_reader = reader.test()
        class_dim = 10
        image_shape = "1,28,28"
    elif args.data == "imagenet":
        import imagenet_reader as reader
        train_reader = reader.train()
        val_reader = reader.val()
        class_dim = 1000
        image_shape = "3,224,224"
    elif args.data == "fruit_veg":
        import reader_cv2 as reader
        train_reader = reader.train(settings=args)
        val_reader = reader.val(settings=args)
        class_dim = 23
        image_shape = "3,224,224"
        resize_short_size = 256
    elif args.data == "yolov3-384":
        import reader_cv2 as reader
        train_reader = reader.train(settings=args)
        val_reader = reader.val(settings=args)
        class_dim = 80
        image_shape = "3,384,384"

    else:
        raise ValueError("{} is not supported.".format(args.data))

    image_shape = [int(m) for m in image_shape.split(",")]
    image = fluid.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    assert args.model in model_list, "{} is not in lists: {}".format(
        args.model, model_list)
    # model definition
    # model = models.__dict__[args.model]()
    model = Model.InceptionV4()
    out = model.net(input=image, class_dim=class_dim)
    val_program = fluid.default_main_program().clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(
                os.path.join(args.pretrained_model, var.name))

        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)
    else:
        assert False, "args.pretrained_model must set"

    fluid.io.save_inference_model(
        './inference_model/' + args.model,
        feeded_var_names=[image.name],
        target_vars=[out],
        executor=exe,
        main_program=val_program,
        model_filename='model',
        params_filename='weights')


def main():
    args = parser.parse_args()
    print_arguments(args)
    export_model(args)


if __name__ == '__main__':
    main()
