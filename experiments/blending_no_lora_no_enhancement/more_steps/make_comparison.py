"""Create side-by-side GT vs more_steps comparison for a given frame."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

_HERE       = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / 'results'
GT_FRAMES   = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs'
                   '/teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150')

frame_i = int(sys.argv[1]) if len(sys.argv) > 1 else 77
fname   = f'frame_{frame_i:04d}.png'

gt   = Image.open(GT_FRAMES   / fname).convert('RGB').resize((518, 518))
pred = Image.open(RESULTS_DIR / fname).convert('RGB')

W, H    = gt.width, gt.height
LABEL_H = 30
out     = Image.new('RGB', (W * 2, H + LABEL_H), (30, 30, 30))

out.paste(gt,   (0,     LABEL_H))
out.paste(pred, (W,     LABEL_H))

draw = ImageDraw.Draw(out)
draw.rectangle([0, 0, W - 1, LABEL_H - 1],     fill=(30, 30, 30))
draw.rectangle([W, 0, W * 2 - 1, LABEL_H - 1], fill=(30, 30, 30))
draw.text((W // 2,     LABEL_H // 2), f'GT frame {frame_i}',         fill='white', anchor='mm')
draw.text((W + W // 2, LABEL_H // 2), f'more_steps=50 frame {frame_i}', fill='white', anchor='mm')

out_path = RESULTS_DIR / f'comparison_f{frame_i:02d}.png'
out.save(out_path)
print(f'Saved: {out_path}')
