"""
Step 1: Input Preparation

t       : float 0.0 -> 1.0
k       : int, window half-size (e.g. 1, 3, 5)

Outputs:
  frame_idx       : int in [1, 150], the target frame
  window_indices  : list of ints, [t-k ... t+k], clamped to [1, 150]
  window_frames   : list of PIL.Image, GT frames for the window
  mesh_path       : Path to fixed mesh OBJ
"""

from pathlib import Path
from PIL import Image

GT_FRAMES_DIR = Path(
    '/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
    '/outputs/teapot_lava_kling_premium'
    '/teapot_lava_kling_premium_front/all_frames_150'
)

MESH_PATH = Path(
    '/net/projects/ranalab/rajhansini/multi_iSeg'
    '/meshes/meshestotrain/teapot_homogenized_unwarp.obj'
)

N_FRAMES = 150


def t_to_frame_idx(t: float) -> int:
    """Map t in [0.0, 1.0] to frame index in [1, 150]."""
    assert 0.0 <= t <= 1.0, f"t must be in [0.0, 1.0], got {t}"
    return round(t * (N_FRAMES - 1)) + 1


def get_window_indices(frame_idx: int, k: int) -> list[int]:
    """Return window [frame_idx-k ... frame_idx+k], clamped to [1, 150]."""
    return [max(1, min(N_FRAMES, frame_idx + offset)) for offset in range(-k, k + 1)]


def load_frame(frame_idx: int) -> Image.Image:
    path = GT_FRAMES_DIR / f'frame_{frame_idx:04d}.png'
    return Image.open(path).convert('RGB')


def prepare_input(t: float, k: int):
    """
    Returns:
      frame_idx      : int
      window_indices : list[int]   length = 2k+1
      window_frames  : list[PIL]   length = 2k+1
      mesh_path      : Path
    """
    frame_idx = t_to_frame_idx(t)
    window_indices = get_window_indices(frame_idx, k)
    window_frames = [load_frame(i) for i in window_indices]
    return frame_idx, window_indices, window_frames, MESH_PATH
