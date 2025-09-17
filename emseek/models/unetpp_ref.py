import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from typing import List
from torchvision.models.resnet import resnet18, resnet34, resnet50
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

# ----------------------------
# Basic convolution block with optional BatchNorm and ReLU activation
# ----------------------------
class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=False))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

# ----------------------------
# Simple attention module (if attention_type is 'scse', applies a squeeze-and-excitation operation;
# otherwise, returns the input unchanged)
# ----------------------------
class Attention(nn.Module):
    def __init__(self, attention_type: str, in_channels: int):
        super().__init__()
        if attention_type == "scse":
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.attention = nn.Identity()
    
    def forward(self, x):
        return x * self.attention(x)

# ----------------------------
# DecoderBlock for Unet++: upsamples, concatenates skip features, and applies two convolution blocks.
# ----------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type: str = None):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention1 = Attention(attention_type, in_channels + skip_channels) if attention_type else nn.Identity()
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(attention_type, out_channels) if attention_type else nn.Identity()
    
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

# ----------------------------
# CenterBlock: two sequential convolution blocks.
# ----------------------------
class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        super().__init__(conv1, conv2)

# ----------------------------
# Unet++ Decoder with dense skip connections.
# ----------------------------
class UnetPlusPlusDecoder(nn.Module):
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int],
                 n_blocks: int = 5, use_batchnorm: bool = True, attention_type: str = None, center: bool = False):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(f"Model depth is {n_blocks}, but you provided decoder_channels for {len(decoder_channels)} blocks.")
        
        # Remove the first encoder channel (input image) and reverse the order
        encoder_channels = encoder_channels[1:][::-1]
        
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        self.depth = len(self.in_channels) - 1
        
        self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm) if center else nn.Identity()
        
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_0_{len(self.in_channels) - 1}"] = DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], **kwargs)
        self.blocks = nn.ModuleDict(blocks)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[1:][::-1]
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features)
        dense_x[f"x_0_{self.depth}"] = self.blocks[f"x_0_{self.depth}"](dense_x[f"x_0_{self.depth - 1}"])
        return dense_x[f"x_0_{self.depth}"]

# ----------------------------
# Segmentation Head: a convolution layer followed by an optional activation and upsampling.
# ----------------------------
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        padding = kernel_size // 2
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        if upsampling > 1:
            upsample = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=True)
        else:
            upsample = nn.Identity()
        if activation is None:
            act = nn.Identity()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "softmax":
            act = nn.Softmax(dim=1)
        else:
            act = activation
        super().__init__(conv, upsample, act)

class PromptEncoder(nn.Module):
    def __init__(self, prompt_in_channels: int, target_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(prompt_in_channels, target_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(target_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_channels // 2, target_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.encoder(x)

# ----------------------------
# ResNet Encoder without using EncoderMixin
# ----------------------------
class ResNetEncoderNoMixin(ResNet):
    """
    ResNet encoder implementation without using EncoderMixin.
    """
    def __init__(self, out_channels: List[int], depth: int = 5, output_stride: int = 32, **kwargs):
        if depth > 5 or depth < 1:
            raise ValueError(f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}")
        super().__init__(**kwargs)
        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride
        
        del self.fc
        del self.avgpool
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features[: self._depth + 1]
    
    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)

def get_encoder(encoder_name: str, **kwargs) -> nn.Module:
    encoder_name = encoder_name.lower()
    if encoder_name in ['resnet18', 'resnet34', 'resnet50']:
        pretrained = kwargs.get('pretrained', True)
        encoder_depth = kwargs.get('encoder_depth', 5)
        encoder_params = {
            'resnet18': {
                'out_channels': [3, 64, 64, 128, 256, 512],
                'block': BasicBlock,
                'layers': [2, 2, 2, 2],
            },
            'resnet34': {
                'out_channels': [3, 64, 64, 128, 256, 512],
                'block': BasicBlock,
                'layers': [3, 4, 6, 3],
            },
            'resnet50': {
                'out_channels': [3, 64, 256, 512, 1024, 2048],
                'block': Bottleneck,
                'layers': [3, 4, 6, 3],
            },
        }
        params = encoder_params[encoder_name]
        encoder = ResNetEncoderNoMixin(
            out_channels=params['out_channels'],
            depth=encoder_depth,
            block=params['block'],
            layers=params['layers']
        )
        if pretrained:
            if encoder_name == 'resnet18':
                pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            elif encoder_name == 'resnet34':
                pretrained_model = resnet34(weights=ResNet34_Weights.DEFAULT)
            elif encoder_name == 'resnet50':
                pretrained_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            state_dict = pretrained_model.state_dict()
            encoder.load_state_dict(state_dict)
        return encoder
    else:
        raise ValueError(f"Encoder {encoder_name} is not supported.")

# class LightweightCrossAttentionFusion(nn.Module):
#     def __init__(self, in_channels, reduction=4):
#         super().__init__()
#         hidden_channels = in_channels // reduction
#         self.conv_q = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
#         self.conv_k = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
#         self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, img_feat, prompt_feat):
#         # img_feat, prompt_feat: (N, C, H, W)
#         Q = self.conv_q(img_feat)       # (N, C_reduced, H, W)
#         K = self.conv_k(prompt_feat)      # (N, C_reduced, H, W)
#         V = self.conv_v(prompt_feat)      # (N, C, H, W)
#         attn = self.sigmoid(Q * K)        # (N, C_reduced, H, W)
#         attn = torch.mean(attn, dim=1, keepdim=True)  # (N, 1, H, W)
#         fused = img_feat + V * attn
#         return fused

class LightweightCrossAttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden_channels = in_channels // reduction
        self.conv_q = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, img_feat, prompt_feat):
        Q = self.conv_q(prompt_feat)
        K = self.conv_k(img_feat)
        V = self.conv_v(img_feat)
        
        Q_resized = F.interpolate(Q, size=K.shape[2:], mode='bilinear', align_corners=False)
        attn = self.sigmoid(Q_resized * K)
        attn = torch.mean(attn, dim=1, keepdim=True)
        fused = img_feat + V * attn
        return fused


class MultiScalePromptFusion(nn.Module):
    def __init__(self, encoder_channels: List[int], reduction: int = 4):
        super().__init__()
        self.fusion_layers = nn.ModuleList([
            LightweightCrossAttentionFusion(in_channels=ch, reduction=reduction)
            for ch in encoder_channels[1:]
        ])
    
    def forward(self, features: List[torch.Tensor], features_prompt: List[torch.Tensor]) -> List[torch.Tensor]:
        fused_features = [features[0]]
        for feat, p_feat, fusion in zip(features[1:], features_prompt[1:], self.fusion_layers):
            fused = fusion(feat, p_feat)
            fused_features.append(fused)
        return fused_features

# ----------------------------
# Modified UnetPlusPlus with a single decoder branch.
# ----------------------------
class UnetPlusPlus(nn.Module):
    def __init__(self,
                 encoder_name: str = 'resnet34',
                 encoder_depth: int = 5,
                 pretrained: bool = True,
                 decoder_channels: List[int] = [256, 128, 64, 32, 16],
                 decoder_use_batchnorm: bool = True,
                 prompt: bool = False,
                 decoder_attention_type: str = None,
                 in_channels: int = 3,
                 classes: int = 1,
                 upsampling: int = 1,
                 checkpoint_path: str = None,
                 encoder_channels: List[int] = None,
                 activation=None):
        super().__init__()

        # Main encoder
        self.encoder = get_encoder(encoder_name,
                                   pretrained=pretrained,
                                   encoder_depth=encoder_depth,
                                   checkpoint_path=checkpoint_path)
        
        if encoder_channels is None:
            encoder_channels = self.encoder._out_channels
        
        self.encoder_channels = encoder_channels
        self.encoder_depth = encoder_depth

        # Create decoder and segmentation head
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(encoder_channels) - 1,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
            center=False
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling,
        )

        if prompt:
            self.visual_prompt = get_encoder(encoder_name,
                                   pretrained=pretrained,
                                   encoder_depth=encoder_depth,
                                   checkpoint_path=checkpoint_path)
            self.multi_scale_prompt_fusion = MultiScalePromptFusion(
                encoder_channels=encoder_channels,
                reduction=4
            )

    def forward(self, x: torch.Tensor, prompt: torch.Tensor = None) -> torch.Tensor:
        features = self.encoder(x)
        if prompt is not None:
            features_prompt = self.visual_prompt(prompt)
            features = self.multi_scale_prompt_fusion(features, features_prompt)
        out = self.decoder(features)
        out = self.segmentation_head(out)
        return out

# ----------------------------
# Example usage:
# ----------------------------
if __name__ == "__main__":
    model = UnetPlusPlus(
        encoder_name='resnet18',
        in_channels=3,
        classes=2,
        prompt=True,                        # enable prompt branch
        decoder_attention_type=None,        # enable if attention is available
    ).eval()

    # Dummy input: batch_size=1, RGB image 1024×1024
    input_tensor = torch.randn(1, 3, 1024, 1024)
    # Dummy reference patch (prompt): same shape
    ref_patch = torch.randn(1, 3, 64, 64)

    from thop import profile, clever_format

    # Forward with two inputs: image + ref patch
    macs, params = profile(model, inputs=(input_tensor, ref_patch), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")

    print(f"FLOPs (including prompt branch): {macs}")
    print(f"Parameters (including prompt branch): {params}")
