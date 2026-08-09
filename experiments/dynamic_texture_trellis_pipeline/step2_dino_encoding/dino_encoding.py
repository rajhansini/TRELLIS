"""
Step 2: Encode Frames to Tokens using DINOv2 ViT-L/14 with registers.

Matches TRELLIS's exact conditioning format:
  features = model(image, is_training=True)['x_prenorm']   # (1, 1374, 1024)
  tokens   = F.layer_norm(features, features.shape[-1:])   # (1, 1374, 1024)

INPUT:
  window_indices : list[int]        frame indices (from Step 1)
  window_frames  : list[PIL.Image]  corresponding GT frames (from Step 1)
  device         : torch.device

OUTPUT:
  tokens_dict : dict {frame_idx (int): {
      "tokens" : tensor (1374, 1024),  — full conditioning (for MCFM + TRELLIS)
      "cls"    : tensor (1, 1024),     — first token, CLS (for Step 3 projector)
  }}

Model: dinov2_vitl14_reg (frozen, local cache)
Input resolution: 518x518 (TRELLIS native — 518/14=37 → 37x37=1369 patches)
Total tokens: 1 CLS + 4 REG + 1369 patches = 1374
Normalization: ImageNet mean/std (matches TRELLIS's image_cond_model_transform)
Token dim: 1024
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T

DINO_DIM   = 1024
N_TOKENS   = 1374    # 1 CLS + 4 REG + 1369 patches (37x37)
IMAGE_SIZE = 518     # TRELLIS native DINOv2 resolution

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_preprocess = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_dino(device: torch.device) -> torch.nn.Module:
    """Load frozen DINOv2 ViT-L/14 with registers from local hub cache."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_frames(
    window_indices: list,
    window_frames: list,
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """
    Encode each frame through DINOv2 using TRELLIS's exact conditioning format.

    Returns dict {frame_idx: {"tokens": (1374,1024), "cls": (1,1024)}}
    """
    assert len(window_indices) == len(window_frames), \
        "window_indices and window_frames must have same length"

    tokens_dict = {}
    with torch.no_grad():
        for idx, img in zip(window_indices, window_frames):
            if idx in tokens_dict:
                continue  # boundary duplicate — already encoded

            x        = _preprocess(img).unsqueeze(0).to(device)              # (1, 3, 518, 518)
            features = model(x, is_training=True)['x_prenorm']               # (1, 1374, 1024)
            tokens   = F.layer_norm(features, features.shape[-1:]).squeeze(0) # (1374, 1024)

            tokens_dict[idx] = {
                'tokens': tokens,           # (1374, 1024) — full conditioning
                'cls'   : tokens[0:1],      # (1, 1024)   — CLS token for projector
            }

    return tokens_dict
