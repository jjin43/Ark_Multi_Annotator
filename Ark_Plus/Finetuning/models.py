import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.swin_transformer import SwinTransformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed

from torch.hub import load_state_dict_from_url
from timm.models.helpers import load_state_dict

from functools import partial
import simmim
#from upernet_swin_transformer import UperNet_swin
#from convnext import ConvNeXt
#from resnet import ResNet50
from utils import load_swin_pretrained

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

def build_classification_model(args):
    model = None
    useVinDrHead = False
    if args.data_set == 'VinDrCXR_17rad':
        print("Using VinDr Head")
        useVinDrHead = False
    
    
    print("Creating model...")
    if args.pretrained_weights is None or args.pretrained_weights =='':
        print('Loading pretrained {} weights for {} from timm.'.format(args.init, args.model_name))
        if args.model_name.lower() == "vit_base":
            if args.init.lower() =="random":
                if args.input_size == 448:
                    model = VisionTransformer(num_classes=args.num_class, img_size = args.input_size,
                        patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                else:
                    model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                # model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('vit_base_patch16_224_in21k', num_classes=args.num_class, pretrained=True)  
            elif args.init.lower() =="sam":
                model = timm.create_model('vit_base_patch16_224_sam', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="dino":
                model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                #model = timm.create_model('vit_base_patch16_224_dino', num_classes=args.num_class, pretrained=True) #not available in current timm version
                url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
                state_dict = torch.hub.load_state_dict_from_url(url=url)
                model.load_state_dict(state_dict, strict=False)
            elif args.init.lower() =="deit":
                model = timm.create_model('deit_base_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="beit":
                model = timm.create_model('beit_base_patch16_224', num_classes=args.num_class, pretrained=True)

        elif args.model_name.lower() == "vit_small":
            if args.init.lower() =="random":
                model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('vit_small_patch16_224_in21k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="dino":
                #model = timm.create_model('vit_small_patch16_224_dino', num_classes=args.num_class, pretrained=True)
                model = VisionTransformer(num_classes=args.num_class,
                    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
                state_dict = torch.hub.load_state_dict_from_url(url=url)
                model.load_state_dict(state_dict, strict=False)
            elif args.init.lower() =="deit":
                model = timm.create_model('deit_small_patch16_224', num_classes=args.num_class, pretrained=True)           
        
        elif args.model_name.lower() == "swin_large":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
            
        elif args.model_name.lower() == "swin_large_384":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size, 
                patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))

        elif args.model_name.lower() == "swin_base": 
            if args.init.lower() =="random":
                if args.input_size == 448:
                    model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
                else:
                    model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_21kto1k":
                model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=True)
            
        elif args.model_name.lower() == "swin_tiny": 
            if args.init.lower() =="random":
                model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=True)
        
        elif args.model_name.lower() == "convx_base":
            if args.init.lower() =="random":
                model = timm.create_model('convnext_base_in22k', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('convnext_base.fb_in1k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('convnext_base_in22k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21kto1k":
                model = timm.create_model('convnext_base_in22ft1k', num_classes=args.num_class, pretrained=True)
        elif args.model_name.lower() == "resnet50":
            if args.init.lower() =="random":
                model = ResNet50(num_classes=args.num_class)
        
    else:
        print("Creating model from pretrained weights: "+ args.pretrained_weights)
        if args.model_name.lower() == "vit_base":
            if args.init.lower() == "simmim":
                model = simmim.create_model(args)
            else:
                model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)
            
        elif args.model_name.lower() == "vit_small":
            model = VisionTransformer(num_classes=args.num_class,
                    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.default_cfg = _cfg()
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)  
            
        elif args.model_name.lower() == "swin_large":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights, args.key, args.scale_up, useVinDrHead=useVinDrHead)
            
        elif args.model_name.lower() == "swin_large_384":
            model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size, 
                patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights, args.key, args.scale_up, useVinDrHead=useVinDrHead)
        
        elif args.model_name.lower() == "swin_base":
            if args.init.lower() == "simmim":
                model = simmim.create_model(args)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class)
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)  
            else:
                print("Using swin_base pretrained weights from Ark.")
                model = SwinTransformer(num_classes=args.num_class, img_size = args.input_size,
                    patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))

                if(useVinDrHead):
                    curr_head = model.head
                    in_features = model.num_features
                    print("init_model_head_features: ", in_features)

                    projector_dim = 1376
                    projector = Projector(in_features, projector_dim, False)
                    vindr_head = nn.Linear(projector_dim, 6)  # shape (6, 1376)
                    model.head = nn.Sequential(
                        projector,  # [batch, 1024] -> [batch, 1376]
                        vindr_head    # [batch, 1376] -> [batch, 6]
                    )
                
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights, args.key, args.scale_up, useVinDrHead=useVinDrHead)  
                
        elif args.model_name.lower() == "swin_tiny": 
            model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class)
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)
            
        elif args.model_name.lower() == "convx_base":
          if args.init.lower().startswith("ark"):
                model = ConvNeXt(num_classes=args.num_class,
                     depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights, args.key)   
          
    if model is None:
        print("Not provide {} pretrained weights for {}.".format(args.init, args.model_name))
        raise Exception("Please provide correct parameters to load the model!")
    return model  
    

def load_pretrained_weights(model, init, pretrained_weights, checkpoint_key = None, scale_up = False, useVinDrHead = False):

    if pretrained_weights.startswith('https'):
        checkpoint = load_state_dict_from_url(url=pretrained_weights, map_location='cpu')
    else:
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
    print(checkpoint.keys())
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']

    if init =="dino":
        checkpoint_key = "teacher"
        if checkpoint_key is not None and checkpoint_key in checkpoint:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = checkpoint[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    elif init =="moco_v3":
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    elif init == "moby":
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder.' in k}
    # elif init == "mae":
    #     state_dict = checkpoint['model']
    elif init.startswith("ark"): 
        print("Loading {} from checkpoint...".format(checkpoint_key))
        state_dict = checkpoint[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() }  
  
    else:
        print("Trying to load the checkpoint for {} at {}, but we cannot guarantee the success.".format(init, pretrained_weights))

    if scale_up:
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


    # Use Vindr Head from pretrained checkpoint
    if useVinDrHead:
        # VinDr head is the 4th head in omni_heads
        from_head, to_head = 'omni_heads.4', 'head.1'
        from_weight = state_dict[from_head + '.weight']  # shape [6, 1376]
        to_weight = model.state_dict()[to_head + '.weight'] # shape [6, 1024]
        print(f"Copying weights from {from_head} with size {from_weight.size(1)} to {to_head} with size {to_weight.size(1)}")
        
        print(f"head weight Before Copy: {model.state_dict()[to_head + '.weight'][:2]}")
        if from_weight.size(1) != to_weight.size(1):
            # copy weights with projector
            # print(f"Projecting weights from {from_head} to {to_head}")
            # projector = Projector(from_weight.size(1), to_weight.size(1), use_mlp=False)
            # with torch.no_grad():
            #     projected_weight = projector(from_weight)
            #     model.state_dict()[to_head + '.weight'].copy_(projected_weight)
            
            # model.state_dict()[to_head + '.bias'].copy_(state_dict[from_head + '.bias'])
            raise ValueError(f"Cannot copy weights from {from_head} to {to_head} due to size mismatch: {from_weight.size(1)} vs {to_weight.size(1)}")
            
        else:
            print(f"Directly copying weights from {from_head} to {to_head}")
            model.state_dict()['head.0.projector.weight'].copy_(state_dict['projector.weight'])
            model.state_dict()['head.0.projector.bias'].copy_(state_dict['projector.bias'])
            model.state_dict()[to_head + '.weight'].copy_(from_weight)
            model.state_dict()[to_head + '.bias'].copy_(state_dict[from_head + '.bias'])
            
        print(f"head weight After Copy: {model.state_dict()[to_head + '.weight'][:2]}")
        print(f"from_head weight: {state_dict[from_head + '.weight'][:2]}")
        
    return model

def save_checkpoint(state,filename='model'):

    torch.save(state, filename + '.pth.tar')

