from .eswin_transformer import ESwinTransformer
from .swin_transformer import SwinTransformer
from .cait import Cait
from .pit import PoolingVisionTransformer
from .pvt import PyramidVisionTransformer
from .cvt import CvT

import torch.nn as nn


def build_model():
    
    # model = ESwinTransformer(img_size=32, patch_size=1, in_chans=3, num_classes=10, embed_dim=128, depths=[2, 2, 8, 2], num_heads=[4, 8, 16, 32], window_size=4, mlp_ratio=2., )

    # model = Cait(img_size=64, patch_size=4, in_chans=3, num_classes=200,  global_pool='token', embed_dim=288, depth=24, num_heads=6,)
    #
    # model = PoolingVisionTransformer(img_size=32, patch_size=1, stride=4, base_dims=[64, 64, 64], depth=[3, 6, 4], heads=[4, 8, 16],
    #          mlp_ratio=4., num_classes=200, in_chans=3)
    #
    # model = PyramidVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1])

    model = CvT(image_size=64, in_channels=3, num_classes=200, dim=64, kernels=[7, 3, 3], strides=[4, 2, 2], heads=[1, 3, 6] , depth = [1, 2, 10], pool='cls', dropout=0., emb_dropout=0., scale_dim=4)
    model = CvT(image_size=64, in_channels=3, num_classes=200, dim=96, kernels=[7, 3, 3], strides=[4, 2, 2], heads=[1, 3, 6] , depth = [1, 4, 16], pool='cls', dropout=0., emb_dropout=0., scale_dim=4)
    return model
