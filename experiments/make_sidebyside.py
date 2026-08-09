"""
Create 4-way side-by-side comparison videos for all 4 MCFM modes.

Each video: GT | TRELLIS baseline | MCFM blending ({mode}) | MCFM + LoRA ({mode})

Runs on CPU (PIL only). Safe to run on login node fe01.

Output:
  sidebyside_v2_C.mp4
  sidebyside_v2_D.mp4
  sidebyside_v3_C.mp4
  sidebyside_v3_D.mp4

Usage:
  python make_sidebyside.py
  python make_sidebyside.py --fps 10 --res 320
  python make_sidebyside.py --mode v3_C
"""

import argparse, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

GT_DIR      = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                   '/outputs/teapot_lava_kling_premium'
                   '/teapot_lava_kling_premium_front/all_frames_150')
TRELLIS_DIR = Path('/net/projects/ranalab/rajhansini/mvadaptornew'
                   '/mvadaptorresults/trellis_150_frames')
ENH_DIR     = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement')
OUT_DIR     = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments')
N_FRAMES    = 150
LABELS      = ['GT video', 'TRELLIS baseline', 'MCFM blending', 'MCFM + LoRA']


def get_frame(source: str, i: int, res: int, mode: str) -> Image.Image:
    if source == 'gt':
        p = GT_DIR / f'frame_{i:04d}.png'
    elif source == 'trellis':
        p = TRELLIS_DIR / f'frame_{i:04d}' / 'renders' / 'front.png'
    elif source == 'mcfm':
        p = ENH_DIR / f'results_mcfm_{mode}_seed6_fixednoise' / 'beta0p0' / f'frame_{i:04d}.png'
    elif source == 'lora':
        p = ENH_DIR / f'results_mcfm_{mode}_selfcons_lora_seed6' / 'renders_e050' / 'rendered' / f'frame_{i:04d}.png'

    if not p.exists():
        img = Image.new('RGB', (res, res), (40, 40, 40))
        draw = ImageDraw.Draw(img)
        draw.text((10, res//2), 'missing', fill=(120, 120, 120))
        return img
    return Image.open(p).convert('RGB').resize((res, res), Image.LANCZOS)


def make_video(mode: str, res: int, fps: int, font):
    total_w  = res * 4
    total_h  = res + 30
    sources  = ['gt', 'trellis', 'mcfm', 'lora']

    frames_dir = OUT_DIR / f'_sbs_frames_{mode}'
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_mp4    = OUT_DIR / f'sidebyside_{mode}.mp4'

    print(f'\n[{mode}] Building {N_FRAMES} frames ({total_w}×{total_h})...')
    for i in range(1, N_FRAMES + 1):
        out_path = frames_dir / f'frame_{i:04d}.png'
        if out_path.exists():
            continue

        canvas = Image.new('RGB', (total_w, total_h), (20, 20, 20))
        draw   = ImageDraw.Draw(canvas)

        for col, (src, lbl) in enumerate(zip(sources, LABELS)):
            img = get_frame(src, i, res, mode)
            canvas.paste(img, (col * res, 30))

            draw.rectangle([col * res, 0, (col + 1) * res - 1, 29], fill=(35, 35, 50))
            try:
                bbox = draw.textbbox((0, 0), lbl, font=font)
                tw   = bbox[2] - bbox[0]
            except AttributeError:
                tw = len(lbl) * 8
            draw.text((col * res + (res - tw) // 2, 8), lbl,
                      fill=(220, 220, 220), font=font)

        draw.text((total_w - 55, total_h - 18), f'f{i:03d}',
                  fill=(140, 140, 140), font=font)
        canvas.save(out_path)

        if i % 50 == 0 or i == N_FRAMES:
            print(f'  {i}/{N_FRAMES}')

    print(f'[{mode}] Encoding mp4...')
    fl = frames_dir / '_fflist.txt'
    with open(fl, 'w') as f:
        for i in range(1, N_FRAMES + 1):
            f.write(f"file '{frames_dir / f'frame_{i:04d}.png'}'\n")
            f.write(f"duration {1.0 / fps:.6f}\n")

    r = subprocess.run(
        ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
         '-i', str(fl), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
         str(out_mp4)],
        capture_output=True, text=True)
    fl.unlink(missing_ok=True)

    if r.returncode != 0:
        print(f'[{mode}] ffmpeg error: {r.stderr[-300:]}')
    else:
        print(f'[{mode}] Done → {out_mp4}  ({out_mp4.stat().st_size / 1e6:.1f} MB)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps',  type=int, default=10)
    parser.add_argument('--res',  type=int, default=320)
    parser.add_argument('--mode', type=str, default=None,
                        choices=['v2_C', 'v2_D', 'v3_C', 'v3_D'])
    args  = parser.parse_args()
    modes = [args.mode] if args.mode else ['v2_C', 'v2_D', 'v3_C', 'v3_D']

    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    for mode in modes:
        make_video(mode, args.res, args.fps, font)

    print('\nAll done. Videos:')
    for mode in modes:
        p = OUT_DIR / f'sidebyside_{mode}.mp4'
        if p.exists():
            print(f'  {p}')


if __name__ == '__main__':
    main()
