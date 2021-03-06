# -*- coding: UTF-8 -*-
"""
模型推理
"""
import os
import sys
import config
import cv2
import numpy as np
import time
import paddle.fluid as fluid
import json
from PIL import Image
from PIL import ImageDraw
import shutil

sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)

import argparse
import functools
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_path',       str,  ".pretrain/ShuffleNetV2-yolov3/",  "model dir")
add_arg('model_filename',       str, None,                 "model file name")
add_arg('params_filename',      str, None,                 "params file name")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")

train_parameters = config.init_train_parameters()
label_dict = train_parameters['num_dict']
yolo_config = train_parameters['yolo_tiny_cfg'] if train_parameters["use_tiny"] else train_parameters["yolo_cfg"]
#place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
#path = train_parameters['freeze_dir']  # 'model/freeze_model'

args = parser.parse_args()
print_arguments(args)
path = args.model_path

place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
#[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe, model_filename='__model__', params_filename='params')
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe, model_filename=args.model_filename, params_filename=args.params_filename)

def draw_bbox_image(img, boxes, labels, gt=False):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    color = ['red', 'blue']
    if gt:
        c = color[1]
    else:
        c = color[0]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, c, width=3)
        draw.text((xmin, ymin), label_dict[int(label)], (255, 255, 0))
    return img


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)

    return img


def read_image(img):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = img
    img = resize_img(origin, yolo_config["input_size"])
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img


def yolov3_infer(image):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    origin, tensor_img, resized_img = read_image(image)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,
                                  feed_target_names[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    period = (time.time() - t1)*1000
    print("predict cost time:{0}".format("%2.2f ms" % period))
    bboxes = np.array(batch_outputs[0])
    #print(bboxes)
    if bboxes.shape[1] != 6:
        # print("No object found")
        return False, [], [], [], [], period
    labels = bboxes[:, 0].astype('int32')
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')
    return True, boxes, labels, scores, bboxes, period


if __name__ == '__main__':
    image_path = '/home/aistudio/work/PaddleSlim/demo/data/pascalvoc/VOCdevkit/VOC2012/JPEGImages/2011_002192.jpg'
    img = cv2.imread(image_path)
    flag, box, label, scores, bboxes, period = yolov3_infer(img)
    if flag:
        img = draw_bbox_image(img, box, label)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        print('检测到目标')
        cv2.imwrite('result.jpg', img)
        print('检测结果保存为result.jpg')
    else:
        print(image_path, "没检测出来")
        pass
    print('infer one picture cost {} ms'.format(period))
