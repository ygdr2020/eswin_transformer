from .eswin_transformer import ESwinTransformer
from .swin_transformer import SwinTransformer
from .cait import Cait
from .pit import PoolingVisionTransformer
from .pvt import PyramidVisionTransformer
from .cvt import CvT
from .deit import VisionTransformerDistilled
from .swin_transformer import SwinTransformer
from functools import partial
from .vision_transformer import VisionTransformer
import torch.nn as nn
from .resnet import ResNet
from .hconvmixer import HConvMixer
import torchvision


def build_model():
    
    model = HConvMixer(embed_dim=128, patch_size=1, kernel_size=[9, 7, 5], n_classes=10, depth=[3, 3, 2], r=[4, 4, 4])
    return model
