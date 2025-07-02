import torch.nn as nn


class CrossAttentionModule(nn.Module):
    def __init__(self, image_channels, text_channels):
        super(CrossAttentionModule, self).__init__()

        # Define linear transformations for image and text
        self.image_linear = nn.Linear(image_channels, image_channels)
        self.text_linear = nn.Linear(text_channels, image_channels)
        self.embed_dim = image_channels
        # Self-attention layers for image and text
        self.image_attention = nn.MultiheadAttention(embed_dim=image_channels, num_heads=8)
        self.text_attention = nn.MultiheadAttention(embed_dim=image_channels, num_heads=8)

    def forward(self, image_tensor, text_tensor):
        # Apply linear transformations to image and text tensors
        image_proj = self.image_linear(image_tensor)
        text_proj = self.text_linear(text_tensor)

        # Calculate cross-attention between image and text
        image_attention_output, _ = self.image_attention(query=image_proj, key=text_proj, value=text_proj)

        return image_attention_output
