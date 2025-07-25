import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)  # Use weights="IMAGENET1K_V1" if needed
        self.backbone = CustomResNetBackbone()  # no longer double-wrapped!

        # self.backbone = nn.Sequential(
        #     nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),     # backbone.0–1
        #     resnet.layer1,  # backbone.4.*
        #     resnet.layer2,  # backbone.5.*
        #     resnet.layer3,  # backbone.6.*
        #     resnet.layer4   # backbone.7.*
        # )

    def forward(self, x):
        return self.backbone(x)

class SkinClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = CustomResNetBackbone()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)  # 🔍 Matches "backbone.*" keys
        return self.fc(x)     # 🔍 Matches "fc.1.*, fc.3.*" keys
