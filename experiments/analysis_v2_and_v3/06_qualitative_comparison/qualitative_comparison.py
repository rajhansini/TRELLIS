"""
06 — Qualitative comparison: GT vs MCFM renders for all 4 modes.

Reads GT video frames and MCFM rendered frames (beta=0) for v2_C, v2_D, v3_C, v3_D.
Produces:
  1. Keyframe grid figure — rows: [GT, v2_C, v2_D, v3_C, v3_D], cols: keyframes
     Paper-ready single figure showing all modes at representative frames.
  2. 5-way side-by-side comparison video (GT | v2_C | v2_D | v3_C | v3_D)
     for each of the 150 frames, encoded as mp4.

NO GPU required.

Output: 06_qualitative_comparison/results/

Usage:
  python qualitative_comparison.py
  python qualitative_comparison.py --keyframes 1,25,50,75,100,125,150
  python qualitative_comparison.py --skip_video
"""

import sys, argparse, subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE     = Path(__file__).resolve().parent
_ROOT     = _HERE.parent.parent.parent
_ENH      = _ROOT / 'experiments' / 'enhancement'
VIDEO_DIR = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                 '/outputs/teapot_lava_kling_premium'
                 '/teapot_lava_kling_premium_front/all_frames_150')
N_FRAMES   = 150
RENDER_RES = 518   # original render resolution
THUMB      = 224   # per-cell thumbnail for the grid figure
VIDFRAME   = 320   # per-column width in comparison video
MODES      = ['v2_C', 'v2_D', 'v3_C', 'v3_D']
COL_LABELS = ['GT video', 'v2-C', 'v2-D', 'v3-C', 'v3-D']
DEFAULT_KF = [1, 25, 50, 75, 100, 125, 150]


class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1)
    def write(self, msg):
        sys.__stdout__.write(msg); self._f.write(msg)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


def render_dir(mode: str) -> Path:
    return _ENH / f'results_mcfm_{mode}_seed6_fixednoise' / 'beta0p0'


def load_img(path: Path, res: int = RENDER_RES) -> np.ndarray:
    """Load image → (H, W, 3) uint8."""
    return np.array(Image.open(path).convert('RGB').resize((res, res), Image.LANCZOS))


def make_video(frames_dir: Path, out_path: Path, fps: int = 10):
    """Create mp4 from PNGs in frames_dir using ffmpeg concat demuxer."""
    frames = sorted(frames_dir.glob('frame_*.png'))
    if not frames:
        print('  [VIDEO] no frames found, skipping')
        return
    filelist = out_path.parent / '_fflist.txt'
    with open(filelist, 'w') as f:
        for p in frames:
            f.write(f"file '{p}'\n")
            f.write(f"duration {1.0 / fps:.6f}\n")
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
           '-i', str(filelist), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    filelist.unlink(missing_ok=True)
    if r.returncode != 0:
        print(f'  [VIDEO] ffmpeg error: {r.stderr[-300:]}')
    else:
        print(f'  [VIDEO] saved → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyframes', type=str, default=None,
                        help='Comma-separated keyframe indices (default: 1,25,50,75,100,125,150)')
    parser.add_argument('--skip_video', action='store_true',
                        help='Skip 5-way video generation')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    keyframes = (list(map(int, args.keyframes.split(',')))
                 if args.keyframes else DEFAULT_KF)

    out_dir = _HERE / 'results' / 'qualitative_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / 'qualitative_comparison.log')
    sys.stderr = sys.stdout

    print(f'[QUALITATIVE] keyframes={keyframes}')

    # ── Check which modes have renders ────────────────────────────────────────
    available = []
    for mode in MODES:
        d = render_dir(mode)
        sample = d / f'frame_{keyframes[0]:04d}.png'
        if d.exists() and sample.exists():
            available.append(mode)
            print(f'  [{mode}] renders found at {d.name}/beta0p0')
        else:
            print(f'  [{mode}] MISSING renders at {d} — will skip')

    if not available:
        print('[ERROR] No MCFM render dirs found. Run step08/step09 pipeline first.')
        return

    # ── 1. Keyframe grid figure ───────────────────────────────────────────────
    #    Rows: GT + each available mode; Cols: keyframes
    row_labels = ['GT video'] + [m.replace('_', '-') for m in available]
    n_rows = len(row_labels)
    n_cols = len(keyframes)

    print(f'\n[GRID] Building {n_rows}×{n_cols} keyframe grid...')
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.2 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle('MCFM temporal blending — qualitative comparison', fontsize=12, y=1.01)

    for col, kf in enumerate(keyframes):
        # GT
        gt_path = VIDEO_DIR / f'frame_{kf:04d}.png'
        gt_img  = load_img(gt_path, THUMB)
        axes[0, col].imshow(gt_img)
        axes[0, col].set_title(f'Frame {kf}', fontsize=8)
        axes[0, col].axis('off')

        # Each mode
        for row, mode in enumerate(available, start=1):
            rp  = render_dir(mode) / f'frame_{kf:04d}.png'
            img = load_img(rp, THUMB)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')

    # Row labels on left
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                 labelpad=55, va='center', ha='right')

    plt.tight_layout()
    grid_path = out_dir / 'keyframe_grid.png'
    fig.savefig(grid_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'[PLOT] saved keyframe_grid.png  ({n_rows}×{n_cols})')

    # ── 2. 5-way comparison video ──────────────────────────────────────────────
    if args.skip_video:
        print('[VIDEO] skipped (--skip_video)')
    else:
        vid_frames_dir = out_dir / 'video_frames'
        vid_frames_dir.mkdir(exist_ok=True)

        all_cols = ['GT'] + available
        label_h  = 28
        cell_w   = VIDFRAME
        cell_h   = VIDFRAME
        total_w  = cell_w * len(all_cols)
        total_h  = cell_h + label_h

        print(f'\n[VIDEO] Building {len(all_cols)}-way comparison '
              f'({total_w}×{total_h}) for {N_FRAMES} frames...')

        # Try to get a font; fallback to default if not available
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        except Exception:
            font = ImageFont.load_default()

        for fi in range(1, N_FRAMES + 1):
            frame_path = vid_frames_dir / f'frame_{fi:04d}.png'
            if frame_path.exists():
                continue  # resume if interrupted

            canvas = Image.new('RGB', (total_w, total_h), (30, 30, 30))
            draw   = ImageDraw.Draw(canvas)

            for ci, col_id in enumerate(all_cols):
                if col_id == 'GT':
                    src = VIDEO_DIR / f'frame_{fi:04d}.png'
                else:
                    src = render_dir(col_id) / f'frame_{fi:04d}.png'

                if src.exists():
                    img = Image.open(src).convert('RGB').resize((cell_w, cell_h), Image.LANCZOS)
                else:
                    img = Image.new('RGB', (cell_w, cell_h), (60, 60, 60))

                canvas.paste(img, (ci * cell_w, label_h))

                # Label bar
                label = 'GT video' if col_id == 'GT' else col_id.replace('_', '-')
                draw.rectangle([ci * cell_w, 0, (ci + 1) * cell_w - 1, label_h - 1],
                               fill=(45, 45, 60))
                draw.text((ci * cell_w + 6, 7), label, fill=(200, 200, 200), font=font)

            # Frame index overlay bottom-right
            draw.text((total_w - 80, total_h - 20), f'f{fi:03d}',
                      fill=(180, 180, 180), font=font)
            canvas.save(frame_path)

            if fi % 30 == 0 or fi == N_FRAMES:
                print(f'  rendered composite frame {fi}/{N_FRAMES}')

        make_video(vid_frames_dir, out_dir / 'comparison_5way.mp4', fps=args.fps)

    print(f'\n[DONE] results → {out_dir}')


if __name__ == '__main__':
    main()
