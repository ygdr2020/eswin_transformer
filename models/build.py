from .eswin_transformer import ESwinTransformer
from .swin_transformer import SwinTransformer
from .cait import Cait

import torch.nn as nn


def build_model():
    
    # model = ESwinTransformer(img_size=32, patch_size=1, in_chans=3, num_classes=10,
    #              embed_dim=96, depths=[2, 2, 4, 2], num_heads=[3, 6, 12, 24],
    #              window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #              drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #              norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
    #              use_checkpoint=False)

    model = Cait(img_size=64, patch_size=4, in_chans=3, num_classes=200,  global_pool='token', embed_dim=768, depth=12, num_heads=12,)
    

    return model
