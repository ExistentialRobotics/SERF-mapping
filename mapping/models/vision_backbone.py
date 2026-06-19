import torch
import torch.nn.functional as F
from torchvision import transforms


# ==============================================================================================
#  Model Wrappers
# ==============================================================================================

# EVA_CLIP wrapper class
class EvaClipWrapper(torch.nn.Module):
    def __init__(self, clip_model, output_dim=768):
        super().__init__()
        self.clip_model = clip_model
        self.output_dim = output_dim

        # EVA_CLIP normalization
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

    @torch.no_grad()
    def _forward_eva_clip_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-patch token embeddings without the [CLS] token.
        Shape: (B, N, C), where N = (H/14)*(W/14), C = embed_dim.
        """
        vision_model = self.clip_model.visual.trunk
        x = vision_model.forward_features(x)
        x = vision_model.norm(x)
        x = vision_model.fc_norm(x) # fc_norm is not in this version of open_clip
        x = vision_model.head_drop(x)
        x = vision_model.head(x)
        x = x[:, 1:, :] # drop CLS token
        dense_features = F.normalize(x, dim=-1)
        return dense_features

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_bchw: Float tensor in [0, 1], shape (B, 3, H, W)

        Returns:
            fmap: (B, C, Hf, Wf) where C = output_dim and Hf = H//14, Wf = W//14
        """
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()
        # Normalize per EVA_CLIP recipe
        images_bchw = self.normalize(images_bchw)

        B, _, H, W = images_bchw.shape
        with torch.no_grad():
            tokens = self._forward_eva_clip_tokens(images_bchw)  # (B, N, C)

        C = self.output_dim
        Hf, Wf = H // 14, W // 14
        fmap = tokens.permute(0, 2, 1).reshape(B, C, Hf, Wf).contiguous()

        return fmap


# DINOv3 wrapper class (text-aligned DINOTxt)
class DINOv3Wrapper(torch.nn.Module):
    """
    Wrapper for DINOTxt model that extracts text-aligned patch tokens.
    DINOTxt uses dinov3_vitl16 backbone with additional vision head blocks for text alignment.
    """
    def __init__(self, model, tokenizer=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        # Get backbone properties from the vision model
        self.backbone = model.visual_model.backbone
        self.feature_dim = self.backbone.embed_dim  # 1024 for vitl16
        self.patch_size = self.backbone.patch_size  # 16

        # DINOv3 normalization
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """
        Extract text-aligned patch tokens from images.

        Args:
            images_bchw: Float tensor in [0, 1], shape (B, 3, H, W)

        Returns:
            fmap: (B, C, Hf, Wf) where C = feature_dim (1024) and Hf = H//patch_size
        """
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()
        images_bchw = self.normalize(images_bchw)

        B, _, H, W = images_bchw.shape
        with torch.no_grad():
            # Get text-aligned patch tokens (processed through vision head blocks)
            _, patch_tokens, _ = self.model.encode_image_with_patch_tokens(images_bchw)
            # patch_tokens shape: (B, N, C) where N = (H/16) * (W/16), C = 1024

        C = self.feature_dim
        Hf, Wf = H // self.patch_size, W // self.patch_size
        fmap = patch_tokens.permute(0, 2, 1).reshape(B, C, Hf, Wf).contiguous()

        return fmap

    def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode text tokens into text embeddings."""
        with torch.no_grad():
            return self.model.encode_text(text, normalize=normalize)

    def encode_image(self, images_bchw: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode images into image embeddings (global features)."""
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()
        images_bchw = self.normalize(images_bchw)
        with torch.no_grad():
            return self.model.encode_image(images_bchw, normalize=normalize)
