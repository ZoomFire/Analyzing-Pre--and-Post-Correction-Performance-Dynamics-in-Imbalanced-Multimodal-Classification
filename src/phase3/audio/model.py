import torch
import torch.nn as nn
import timm


class CNNTransformer(nn.Module):

    def __init__(
        self,
        num_classes=50
    ):

        super().__init__()

        # =====================================
        # STRONGER CNN BACKBONE
        # =====================================

        self.cnn = timm.create_model(
            "efficientnet_b2",
            pretrained=True,
            in_chans=1,
            num_classes=0
        )

        cnn_features = self.cnn.num_features

        # =====================================
        # DEEPER TRANSFORMER
        # =====================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_features,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        # =====================================
        # CLASSIFIER
        # =====================================

        self.classifier = nn.Sequential(

            nn.Linear(cnn_features, 1024),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(1024, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        # CNN Features
        features = self.cnn.forward_features(x)

        B, C, H, W = features.shape

        # Flatten
        features = features.reshape(
            B,
            C,
            H * W
        )

        features = features.permute(
            0,
            2,
            1
        )

        # Transformer
        transformer_out = self.transformer(
            features
        )

        # Global Pooling
        transformer_out = transformer_out.mean(
            dim=1
        )

        # Classification
        output = self.classifier(
            transformer_out
        )

        return output