from __future__ import absolute_import
from .mobilenet import MobileNet
from .resnet import ResNet34, ResNet50
from .resnet_vd import ResNet50_vd, ResNet101_vd
from .mobilenet_v2 import MobileNetV2_x0_25, MobileNetV2
from .pvanet import PVANet
from .slimfacenet import SlimFaceNet_A_x0_60, SlimFaceNet_B_x0_75, SlimFaceNet_C_x0_75
from .inception_v4 import InceptionV4
from .shufflenetv2_yolov3 import YOLOv3
__all__ = [
    "model_list", "MobileNet", "ResNet34", "ResNet50", "MobileNetV2", "PVANet",
    "ResNet50_vd", "ResNet101_vd", "MobileNetV2_x0_25", "InceptionV4", "YOLOv3"
]
model_list = [
    'MobileNet', 'ResNet34', 'ResNet50', 'MobileNetV2', 'PVANet',
    'ResNet50_vd', "ResNet101_vd", "MobileNetV2_x0_25", "InceptionV4", "YOLOv3"
]
