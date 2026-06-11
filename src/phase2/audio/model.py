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
        # CNN BACKBONE
        # =====================================

        self.cnn = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=1,
            num_classes=0
        )

        cnn_features = self.cnn.num_features

        # =====================================
        # TRANSFORMER
        # =====================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_features,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # =====================================
        # CLASSIFIER
        # =====================================

        self.classifier = nn.Sequential(

            nn.Linear(cnn_features, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        """
        Input:
        [B, 128, Time]
        """

        # Add channel dimension
        x = x.unsqueeze(1)

        # =====================================
        # CNN FEATURES
        # =====================================

        features = self.cnn.forward_features(x)

        """
        Shape:
        [B, C, H, W]
        """

        B, C, H, W = features.shape

        # =====================================
        # FLATTEN FOR TRANSFORMER
        # =====================================

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

        """
        Shape:
        [B, Sequence, Features]
        """

        # =====================================
        # TRANSFORMER
        # =====================================

        transformer_out = self.transformer(
            features
        )

        # Global Average Pooling
        transformer_out = transformer_out.mean(
            dim=1
        )

        # =====================================
        # CLASSIFICATION
        # =====================================

        output = self.classifier(
            transformer_out
        )

        return output