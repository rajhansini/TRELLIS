"""
Build one image per config: GT + all betas side by side (1 row per file).
Output: per_config/ folder with one PNG per config.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE    = Path(__file__).resolve().parent
GT_PATH = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs'
               '/teapot_lava_kling_premium/teapot_lava_kling_premium_front'
               '/all_frames_150/frame_0075.png')

BETA_TAGS   = ['beta0p0', 'beta1p0', 'beta2p0', 'beta3p0',
               'beta4p0', 'beta6p0', 'beta8p0', 'beta16p0']
BETA_LABELS = ['β=0', 'β=1', 'β=2', 'β=3', 'β=4', 'β=6', 'β=8', 'β=16']

THUMB   = 300
LABEL_H = 32

CONFIGS = [
    ('raw',   'raw',            'raw  lc=1.0 ln=0.0'),
    ('C0',    'phase_c_v1_C0',  'C0   lc=1.0 ln=0.0'),
    ('C1',    'phase_c_v1_C1',  'C1   lc=0.9 ln=0.1'),
    ('C2',    'phase_c_v1_C2',  'C2   lc=0.8 ln=0.2'),
    ('C3',    'phase_c_v1_C3',  'C3   lc=0.7 ln=0.3'),
    ('C4',    'phase_c_v1_C4',  'C4   lc=0.6 ln=0.4'),
    ('C5',    'phase_c_v1_C5',  'C5   lc=0.5 ln=0.5'),
    ('C6',    'phase_c_v1_C6',  'C6   lc=0.4 ln=0.6'),
    ('C7',    'phase_c_v1_C7',  'C7   lc=0.3 ln=0.7'),
    ('C8',    'phase_c_v1_C8',  'C8   lc=0.2 ln=0.8'),
    ('C9',    'phase_c_v1_C9',  'C9   lc=0.1 ln=0.9'),
    ('C10',   'phase_c_v1_C10', 'C10  lc=0.0 ln=1.0'),
    ('C_v2',  'phase_c_v2',     'C_v2 attn-blend'),
    ('D0',    'phase_d_v1_D0',  'D0   lp=0.0  lc=1.0  ln=0.0'),
    ('D1',    'phase_d_v1_D1',  'D1   lp=0.1  lc=0.8  ln=0.1'),
    ('D2',    'phase_d_v1_D2',  'D2   lp=0.2  lc=0.6  ln=0.2'),
    ('D3',    'phase_d_v1_D3',  'D3   lp=0.25 lc=0.5  ln=0.25'),
    ('D4',    'phase_d_v1_D4',  'D4   lp=0.33 lc=0.33 ln=0.33'),
    ('D5',    'phase_d_v1_D5',  'D5   lp=0.4  lc=0.2  ln=0.4'),
    ('D6',    'phase_d_v1_D6',  'D6   lp=0.5  lc=0.0  ln=0.5'),
    ('D_v2',  'phase_d_v2',     'D_v2 attn-blend'),
]

OUT_DIR = BASE / 'per_config'
OUT_DIR.mkdir(exist_ok=True)


def try_font(size=14):
    for name in ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
                 '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
                 '/usr/share/fonts/truetype/freefont/FreeMono.ttf']:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_thumb(path):
    if not Path(path).exists():
        img = Image.new('RGB', (THUMB, THUMB), (60, 60, 60))
        ImageDraw.Draw(img).text((8, THUMB//2), 'MISSING', fill=(255, 80, 80))
        return img
    return Image.open(path).convert('RGB').resize((THUMB, THUMB), Image.LANCZOS)


font = try_font(14)

n_cols = 1 + len(BETA_TAGS)   # GT + 8 betas
W = n_cols * THUMB
H = LABEL_H + THUMB

for name, res_dir, label in CONFIGS:
    canvas = Image.new('RGB', (W, H), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)

    # header: config label + beta labels
    draw.rectangle([(0, 0), (W, LABEL_H)], fill=(40, 40, 40))
    draw.text((6, 8), label, fill=(255, 220, 80), font=font)
    for bi, bl in enumerate(BETA_LABELS):
        x = (1 + bi) * THUMB + THUMB // 2 - 20
        draw.text((x, 8), bl, fill=(200, 200, 200), font=font)

    # GT
    gt = load_thumb(GT_PATH)
    canvas.paste(gt, (0, LABEL_H))

    # betas
    for bi, btag in enumerate(BETA_TAGS):
        img_path = BASE / f'results_{res_dir}' / btag / 'frame_0075.png'
        canvas.paste(load_thumb(img_path), ((1 + bi) * THUMB, LABEL_H))

    # vertical divider after GT
    draw.line([(THUMB, 0), (THUMB, H)], fill=(100, 100, 100), width=2)

    out = OUT_DIR / f'{name}.png'
    canvas.save(out)
    print(f'  {out.name}')

print(f'\nDone. {len(CONFIGS)} images in {OUT_DIR}')
