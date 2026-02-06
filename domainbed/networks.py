# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

# from domainbed.lib import wide_resnet
import copy

import timm

import clip

from PIL import Image
def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        x = self.activation(x) # for URM; does not affect other algorithms
        return x

class DinoV2(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(DinoV2, self).__init__()

        self.network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.n_outputs =  5 * 768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    def forward(self, x):
        x = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(1)
            ], dim=1)
        return self.dropout(linear_input)


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        if hparams['resnet50_augmix']:
            self.network = timm.create_model('resnet50.ram_in1k', pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.activation(self.dropout(self.network(x)))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["freeze_bn"]:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return self.activation(x)


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return custom_clip()
        if hparams["vit"]:
            if hparams["dinov2"]:
                return DinoV2(input_shape, hparams)
            else:
                raise NotImplementedError
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class BaseTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class Simple_TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
class Simple_ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post

        if clip_model.visual.proj is not None:
            self.proj = clip_model.visual.proj
        self.dtype = clip_model.dtype
    def forward(self, image):
        x = self.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model, prompt_learner=None):
        super().__init__()
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
        trainer = cfg.TRAINER.NAME.split('_')[0].upper()
        self.tp = cfg.TRAINER[trainer].TP if hasattr(cfg.TRAINER[trainer], 'TP') else True
        self.deep = cfg.TRAINER[trainer].T_DEEP if hasattr(cfg.TRAINER[trainer], 'T_DEEP') else False
        self.num_tokens = cfg.TRAINER[trainer].N_CTX if hasattr(cfg.TRAINER[trainer], 'N_CTX') else 2
        self.location = cfg.TRAINER[trainer].LOCATION if hasattr(cfg.TRAINER[trainer], 'LOCATION') else 'middle'
        self.deep_layer = cfg.TRAINER[trainer].DEEP_LAYERS if hasattr(cfg.TRAINER[trainer], 'DEEP_LAYERS') else None
        self.num_layer = cfg.MODEL.NUM_LAYER if hasattr(cfg.MODEL, 'NUM_LAYERS') else 12  
        
        dropout = cfg.TRAINER[trainer].prompt_dropout if hasattr(cfg.TRAINER[trainer], 'prompt_dropout') else 0.0
        self.prompt_dropout = nn.Dropout(dropout)
        
        self.enbale_adpater = cfg.TRAINER[trainer].ENABLE_ADAPTER if hasattr(cfg.TRAINER[trainer], 'ENABLE_ADAPTER') else False
        if self.enbale_adpater:
            self.adapter = prompt_learner.adapter
            self.deep_adapter = prompt_learner.deep_adapter
       
    def forward(self, prompts, tokenized_prompts, deep_prompts=None):
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.enbale_adpater or self.deep:
            x = self.transformer_deep(x, deep_prompts)
        else:
            x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(prompts.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
    def transformer_deep(self, x, deep_x):
        if self.deep:
            if deep_x.dim == 2:
                if self.deep_layer == None:
                    deep_x = deep_x.unsueeze(0).expand(self.num_layer - 1, -1, -1)  # all layers exsit prompt
                else:
                    deep_x = deep_x.unsueeze(0).expand(self.deep_layer[1] - self.deep_layer[0] + 1, -1, -1) # only specified layers exsit prompt
            
        for i in range(self.num_layer):
            if i == 0:
                if self.enbale_adpater:     # adapter is not activate
                    x = self.resblocks_adapter(i, x)
                else:
                    x = self.transformer.resblocks[i](x)
                    
            else:
                if self.deep:
                    if self.deep_layer == None:   # only specified layers exsit prompt
                        if i <= deep_x.shape[0]:   
                            deep_ctx_i =  self.prompt_dropout(deep_x[i-1])
                            deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                            if self.location == "middle":
                                x = torch.cat((x[:1, :, :], deep_ctx, x[(1+self.num_tokens):, :, :]), dim=0)
                            else:   # 'last'
                                prefix = x[0: x.shape[0] - self.num_tokens, :, :]
                                x = torch.cat([prefix, deep_ctx], dim=0)
                    else: # all layers exsit prompt
                        j = 0
                        if i in range(self.deep_layer[0], self.deep_layer[1]+1):
                            deep_ctx_i =  self.prompt_dropout(deep_x[j])
                            j = j + 1
                            deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                            if self.location == "middle":
                                x = torch.cat((x[:1, :, :], deep_ctx, x[(1+self.num_tokens):, :, :]), dim=0)
                            else:   # 'last'
                                prefix = x[0: x.shape[0] - self.num_tokens, :, :]
                                x = torch.cat([prefix, deep_ctx], dim=0)
                    
                if self.enbale_adpater:
                    x = self.resblocks_adapter(i, x)  
                else:
                    x = self.transformer.resblocks[i](x)
                    
        return x
    
    def resblocks_adapter(self, i, x):
        attn = self.transformer.resblocks[i].attn
        ln_1 = self.transformer.resblocks[i].ln_1
        mlp = self.transformer.resblocks[i].mlp
        ln_2 = self.transformer.resblocks[i].ln_2
        attn_mask = self.transformer.resblocks[i].attn_mask
        
        def attention(x, attn, attn_mask):
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
            return attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

        x = x + attention(ln_1(x), attn, attn_mask)
        residual = self.adapter(x) * 0.5
        x = x + residual + mlp(ln_2(x))
        return x
    
          
class ImageEncoder_Conv(nn.Module):
    def __init__(self, cfg, clip_model, prompt_learner=None):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = clip_model.visual.conv1, clip_model.visual.bn1, clip_model.visual.relu
        self.conv2, self.bn2, self.relu2 = clip_model.visual.conv2, clip_model.visual.bn2, clip_model.visual.relu
        self.conv3, self.bn3, self.relu3 = clip_model.visual.conv3, clip_model.visual.bn3, clip_model.visual.relu
        self.avgpool = clip_model.visual.avgpool

        self.layer1 = clip_model.visual.layer1
        self.layer2 = clip_model.visual.layer2
        self.layer3 = clip_model.visual.layer3
        self.layer4 = clip_model.visual.layer4
        self.attnpool = clip_model.visual.attnpool
        
        trainer = cfg.TRAINER.NAME.split('_')[0].upper()
        self.prompt_learner = prompt_learner
        self.dim = clip_model.text_projection.shape[1]

    def forward(self, x, vctx=None, deep_vctx=None, return_feat=False):
        '''
        Maybe futherly there will be some prompt tuning methods for convolutional networks
        '''
        def stem(x):
            for conv, bn, relu in [(self.conv1, self.bn1, self.relu1), (self.conv2, self.bn2, self.relu2), (self.conv3, self.bn3, self.relu3)]:
                x = relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, C, H, W] = [B, 2048, 7, 7]
        if return_feat:     # you can modify the code as you need 
            Fs = x.permute(0, 2, 3, 1).view(x.shape[0], -1, x.shape[1]) # [B, 49, 2048]
            Fs = F.adaptive_avg_pool1d(Fs, self.dim) # [B, 49, 1024]
        x = self.attnpool(x)    # [B, 1024]

        if return_feat:
            return x, Fs
        
        return x
    

class ImageEncoder_Trans(nn.Module):
    def __init__(self, cfg, clip_model, prompt_learner=None):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post

        if clip_model.visual.proj is not None:
            self.proj = clip_model.visual.proj
        
        trainer = cfg.TRAINER.NAME.split('_')[0].upper()
        self.vp = cfg.TRAINER[trainer].VP if hasattr(cfg.TRAINER[trainer], 'VP') else False
        self.deep = cfg.TRAINER[trainer].V_DEEP if hasattr(cfg.TRAINER[trainer], 'V_DEEP') else False
        self.num_tokens = cfg.TRAINER[trainer].NUM_TOKENS if hasattr(cfg.TRAINER[trainer], 'NUM_TOKENS') else 10
        self.location = cfg.TRAINER[trainer].LOCATION if hasattr(cfg.TRAINER[trainer], 'LOCATION') else 'middle'
        self.deep_layer = cfg.TRAINER[trainer].DEEP_LAYERS if hasattr(cfg.TRAINER[trainer], 'DEEP_LAYERS') else None
        self.enable_attn = cfg.TRAINER[trainer].ENABLE_ATTN if hasattr(cfg.TRAINER[trainer], 'ENABLE_ATTN') else None
        self.num_layer = cfg.MODEL.NUM_LAYER if hasattr(cfg.MODEL, 'NUM_LAYERS') else 12
        
        dropout = cfg.TRAINER[trainer].prompt_dropout if hasattr(cfg.TRAINER[trainer], 'prompt_dropout') else 0.0
        self.prompt_dropout = nn.Dropout(dropout)
        
        self.prompt_learner = prompt_learner
        
        self.dim = clip_model.text_projection.shape[1]

    def forward(self, x, vctx=None, deep_vctx=None, return_feat=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid] = [B, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] = [B, 196, 768]
        x = torch.cat(  # shape = [*, grid ** 2 + 1, width]
                [
                    self.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
                ], 
                dim=1,
            )  
        x = x + self.positional_embedding.to(x.dtype)   # image embedding [B, 197, 768]

        if self.vp and vctx != None:
            x = self.incorporate_prompt(x, vctx)    # [B, 197+num_token, 768]

        x = self.ln_pre(x)      # [B, 197+num_token, 768]

        x = x.permute(1, 0, 2)  # NLD -> LND [197+num_token, B, 768]
        if not self.deep or deep_vctx == None:
            x = self.transformer(x) # [197+num_token, B, 768]
        else:
            x = self.transformer_deep(x, deep_vctx)
        x = x.permute(1, 0, 2)  # LND -> NLD    [B, 197+num_tokens, 768]
        
        if return_feat:
            Fs = x
            Fs = F.adaptive_avg_pool1d(Fs, self.dim) # [B, 197+num, 512]
        x = self.ln_post(x[:, 0, :])    # [B, 768]
    
        if self.proj is not None:
            x = x @ self.proj   # [B, 512]

        if return_feat:
            return x, Fs
        
        return x

    def transformer_deep(self, x, deep_x):
        if deep_x.dim == 2:
            if self.deep_layer == None:
                deep_x = deep_x.expand(self.num_layer - 1, -1, -1)
            else:
                deep_x = deep_x.expand(self.deep_layer[1] - self.deep_layer[0] + 1, -1, -1)
            
        for i in range(self.num_layer):
            if i == 0:
                x = self.transformer.resblocks[i](x)
            else:
                if self.deep_layer == None:
                    if i <= deep_x.shape[0]:
                        deep_ctx_i =  self.prompt_dropout(deep_x[i-1])
                        deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                        if self.location == "middle":
                            x = torch.cat((x[:1, :, :], deep_ctx, x[(1+self.num_tokens):, :, :]), dim=0)
                        else:   # 'last'
                            prefix = x[0: x.shape[0] - self.num_tokens, :, :]
                            x = torch.cat([prefix, deep_ctx], dim=0)
                else:
                    j = 0
                    if i in range(self.deep_layer[0], self.deep_layer[1]+1):
                        deep_ctx_i =  self.prompt_dropout(deep_x[j])
                        j = j + 1
                        deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                        if self.location == "middle":
                            x = torch.cat((x[:1, :, :], deep_ctx, x[(1+self.num_tokens):, :, :]), dim=0)
                        else:   # 'last'
                            prefix = x[0: x.shape[0] - self.num_tokens, :, :]
                            x = torch.cat([prefix, deep_ctx], dim=0)
                         
                x = self.transformer.resblocks[i](x)

        return x
    
    def incorporate_prompt(self, x, vctx):
        # combine prompt embeddings with image-patch embeddings
        if self.location == "middle":
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(vctx).expand(x.shape[0], -1, -1).half(),
                    x[:, 1:, :]
                ), dim=1)   # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        else:
            visual_ctx = self.prompt_dropout(vctx).expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        
        return x    # [B, 197 + num_token, 768]

class SE_attn(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2, num_layers=2, init_weights=True):
        super(SE_attn, self).__init__()
        self.num_layers = num_layers
        self.reduction_ratio = reduction_ratio
        self.layers = nn.ModuleList()
        
        # 创建第一个全连接层
        num_channels_reduced = num_channels // reduction_ratio
        self.layers.append(nn.Linear(num_channels, num_channels_reduced, bias=True))
        
        # 创建中间的全连接层
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(num_channels_reduced, num_channels_reduced, bias=True))
        
        # 创建最后一个全连接层
        self.layers.append(nn.Linear(num_channels_reduced, num_channels, bias=True))
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.out_features = num_channels

        if init_weights:
            self._initialize_weights()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels)
        :return: output tensor
        """
        batch_size, num_channels = input_tensor.size()
        
        # 通过所有全连接层
        x = input_tensor
        for i in range(self.num_layers - 1):
            x = self.relu(self.layers[i](x))
        x = self.sigmoid(self.layers[-1](x))
        
        # 权重乘原向量
        weighted_tensor = torch.mul(input_tensor, x.view(batch_size, num_channels))
        output_tensor = input_tensor + weighted_tensor

        return output_tensor

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
class custom_clip(nn.Module):
    def __init__(self, prompt_learner=None):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/16")
        self.text_encoder = Simple_TextEncoder(self.clip_model)
        self.image_encoder = Simple_ImageEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.se = SE_attn(512)
        self.n_outputs = 512
        # 不冻结se
        self._freeze_parameters()

    def _freeze_parameters(self):
        # 冻结 CLIP 模型的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 冻结文本编码器和图像编码器的参数
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # 保留 SE 模块的参数为可训练
        for param in self.se.parameters():
            param.requires_grad = True
    def forward(self, image, text=None):
        image = image.type(self.image_encoder.conv1.weight.dtype)
        image_features = self.image_encoder(image)
        if text is not None:
            text_features = self.text_encoder(text)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if text is not None:
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        image_features = image_features.type(torch.float32)
        se_features = self.se(image_features)
        return se_features
        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text
if __name__ == "__main__":
    model = custom_clip()
    for name, param in model.named_parameters():
        print(name, param.requires_grad)