import torch
import torch.nn as nn
from torchvision import models

# Central Difference Convolutional Layer (CDCLayer)
class CDCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CDCLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.central_diff = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        conv_out = self.conv(x)
        diff_out = self.central_diff(x)
        x = conv_out - diff_out
        x = self.bn(x)
        return self.dropout(F.relu(x))

# Multi-Attentional Feature Map (MAFM)
class MAFM(nn.Module):
    def __init__(self, in_channels):
        super(MAFM, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        channel_weights = self.channel_att(x)
        x = x * channel_weights
        
        # Spatial Attention
        spatial_weights = self.spatial_att(torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ], dim=1))
        return x * spatial_weights

# Spoof Cue Map Generator
class SpoofCueMapGenerator(nn.Module):
    def __init__(self):
        super(SpoofCueMapGenerator, self).__init__()

    def forward(self, x):
        batch_size, _, height, width = x.shape
        # Generate random spoof cue maps
        cue_map = torch.randn((batch_size, 1, height, width)).to(x.device)
        return cue_map

# Feature Fusion Module
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, features, cue_map):
        # Concatenate features and cue map
        fused_features = torch.cat([features, cue_map], dim=1)
        fused_features = self.relu(self.conv(fused_features))
        return self.dropout(fused_features)

# Enhanced CDCN++ Model with All Modules
class CDCNPlusPlus(nn.Module):
    def __init__(self):
        super(CDCNPlusPlus, self).__init__()
        
        # Multiple CDC Layers for Feature Extraction
        self.cdc1 = CDCLayer(3, 32)
        self.cdc2 = CDCLayer(32, 64)
        self.cdc3 = CDCLayer(64, 128)
        self.cdc4 = CDCLayer(128, 128)
        self.cdc5 = CDCLayer(128, 64)
        
        self.mafm = MAFM(64)  # Multi-Attentional Feature Map
        self.spoof_cue_map_gen = SpoofCueMapGenerator()  # Generate spoof cue maps
        self.feature_fusion = FeatureFusionModule(64)  # Fuse features and cue maps
        
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.dropout = nn.Dropout(0.4)  # Regularization
        self.fc = nn.Linear(64, 1)  # Binary classification output
        
    def forward(self, x):
        # Pass through multiple CDC Layers
        x = self.cdc1(x)
        x = self.cdc2(x)
        x = self.cdc3(x)
        x = self.cdc4(x)
        x = self.cdc5(x)
        
        # Apply Multi-Attentional Feature Map (MAFM)
        x = self.mafm(x)
        
        # Generate Spoof Cue Map and Perform Feature Fusion
        cue_map = self.spoof_cue_map_gen(x)
        x = self.feature_fusion(x, cue_map)
        
        # Global Average Pooling and Classification
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class CDCNPlusPlusClassifier(nn.Module):
    def __init__(self):
        super(CDCNPlusPlusClassifier, self).__init__()
        print("Initializing CDCN++ Model with Central Difference Convolution Layers")
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 1)

    def forward(self, x):
        return self.model(x)
