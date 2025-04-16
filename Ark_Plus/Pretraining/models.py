import torch
import torch.nn as nn
from functools import partial
from torch.hub import load_state_dict_from_url

import timm.models.vision_transformer as vit
import timm.models.swin_transformer as swin
from convnext import ConvNeXt
 
from timm.models.helpers import load_state_dict
from safetensors.torch import load_file

from utils import remap_pretrained_keys_swin

reuseVinDrHead = False

class Projector(nn.Module):
    def __init__(self, in_features, out_features, use_mlp):
        super().__init__()
        if use_mlp:
            self.projector = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True),
                nn.Linear(out_features, out_features)
            )
        else:
            self.projector = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.projector(x)

class ArkSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        if reuseVinDrHead:
            self.projector = Projector(in_features=1024, out_features=1376, use_mlp=False)
        
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        num_classes = num_classes_list[0] if len(num_classes_list) > 0 else 0
        self.omni_heads = nn.ModuleList([nn.Linear(1376, num_classes) if num_classes > 0 else nn.Identity()])


    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        # Always use the single head
        if head_n is not None:
            return x, self.omni_heads[0](x)
        else:
            return [self.omni_heads[0](x)]
    
    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x


def build_omni_model_from_checkpoint(args, num_classes_list, key):
    if args.model_name == "swin_base": #swin_base_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    elif args.model_name == "swin_large": #swin_large_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_384": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =384, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_768": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =768, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    # elif args.model_name == "conv_base":
    #     model = ArkConvNeXt(num_classes_list, args.projector_features, args.use_mlp, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])

    if args.pretrained_weights is not None:
        checkpoint = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)
        state_dict = checkpoint[key]
        if any([True if 'module.' in k else False for k in state_dict.keys()]):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}

        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))     
           
    return model

def build_omni_model(args, num_classes_list):
    if args.model_name == "swin_base": #swin_base_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    elif args.model_name == "swin_large": #swin_large_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_384": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =384, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_768": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =768, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_1152": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =1152, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    # elif args.model_name == "conv_base":
    #     model = ArkConvNeXt(num_classes_list, args.projector_features, args.use_mlp, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    #     # url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth'
    if args.pretrained_weights is not None:
        if args.pretrained_weights.endswith('safetensor'):
            print(f"Loading safetensor weights from {args.pretrained_weights}")
            state_dict = load_file(args.pretrained_weights, device='cpu')
        elif args.pretrained_weights.startswith('https'):
            state_dict = load_state_dict_from_url(url=args.pretrained_weights, map_location='cpu')
        else:
            state_dict = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() }  
        k_del = []
        for k in state_dict.keys():
            if "attn_mask" in k:
                k_del.append(k)
        print(f"Removing key {k_del} from pretrained checkpoint")
        for k in k_del:
            del state_dict[k]
            
        remove_keys = ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']
        for k in remove_keys:
            if k in state_dict:
                print(f"Removing key {k} from pretrained checkpoint")
                del state_dict[k]
            
        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))
        
        if reuseVinDrHead:
            from_head, to_head = 'omni_heads.4', 'omni_heads.0'
            print(f"Directly copying weights from {from_head} to {to_head}")
            model.state_dict()['projector.projector.weight'].copy_(state_dict['projector.weight'])
            model.state_dict()['projector.projector.bias'].copy_(state_dict['projector.bias'])
            model.state_dict()[to_head + '.weight'].copy_(state_dict[from_head + '.weight'])
            model.state_dict()[to_head + '.bias'].copy_(state_dict[from_head + '.bias'])

    return model

def save_checkpoint(state,filename='model'):

    torch.save(state, filename + '.pth.tar')
