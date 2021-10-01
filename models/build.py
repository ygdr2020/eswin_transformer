from .eswin_transformer import ESwinTransformer

import torch.nn as nn


def build_model():
    
    model = ESwinTransformer(img_size=32, patch_size=1, in_chans=3, num_classes=10,
                 embed_dim=192, depths=[2, 2, 16, 2], num_heads=[4, 8, 16, 32],
                 window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False)
    

    return model
