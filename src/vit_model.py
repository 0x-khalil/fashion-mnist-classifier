import torch
import torch.nn as nn
import timm


class FashionMNISTViT(nn.Module):
    def __init__(self):
        super(FashionMNISTViT, self).__init__()

        # Load pretrained Vision Transformer
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True
        )

        # Replace classification head for Fashion-MNIST (10 classes)
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.vit(x)
