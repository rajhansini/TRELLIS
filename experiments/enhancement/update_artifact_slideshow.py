"""
Regenerate the 5-panel slideshow (GT | v2_C | v2_D | v3_C | v3_D) for all 150
frames at JPEG quality=75 and inject into the artifact HTML.
Run anywhere — no GPU needed.
"""
import sys, io, base64, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

N_FRAMES = 150
CELL     = 192   # px per panel (5 × 192 = 960 wide)
LABEL_H  = 24
JPEG_Q   = 75

ENH = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement')
GT  = Path('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
           '/outputs/teapot_lava_kling_premium'
           '/teapot_lava_kling_premium_front/all_frames_150')

SOURCES = [
    (GT,                                                   'GT video'),
    (ENH/'results_mcfm_v2_C_seed6_fixednoise'/'beta0p0', 'MCFM v2_C'),
    (ENH/'results_mcfm_v2_D_seed6_fixednoise'/'beta0p0', 'MCFM v2_D'),
    (ENH/'results_mcfm_v3_C_seed6_fixednoise'/'beta0p0', 'MCFM v3_C'),
    (ENH/'results_mcfm_v3_D_seed6_fixednoise'/'beta0p0', 'MCFM v3_D'),
]

HTML_IN  = Path('/tmp/claude-27281/-net-projects-ranalab-rajhansini-TRELLIS'
                '/7ada7845-acf6-41ec-add5-5c673a2a3048/scratchpad/mcfm_experiments.html')

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 11)
except:
    font = ImageFont.load_default()

def make_frame(fi):
    canvas = Image.new('RGB', (CELL * 5, CELL + LABEL_H), (15, 15, 15))
    draw   = ImageDraw.Draw(canvas)
    for col, (d, lbl) in enumerate(SOURCES):
        p = d / f'frame_{fi:04d}.png'
        img = (Image.open(p).convert('RGB').resize((CELL, CELL), Image.LANCZOS)
               if p.exists() else Image.new('RGB', (CELL, CELL), (40, 40, 40)))
        canvas.paste(img, (col * CELL, LABEL_H))
        draw.rectangle([col*CELL, 0, (col+1)*CELL-1, LABEL_H-1], fill=(30, 30, 45))
        try:   tw = draw.textbbox((0, 0), lbl, font=font)[2]
        except: tw = len(lbl) * 6
        draw.text((col*CELL + (CELL-tw)//2, 5), lbl, fill=(210, 210, 210), font=font)
    draw.text((CELL*5 - 40, CELL+LABEL_H-13), f'f{fi:03d}', fill=(110,110,110), font=font)
    buf = io.BytesIO()
    canvas.save(buf, format='JPEG', quality=JPEG_Q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')

print(f'Generating {N_FRAMES} frames at {CELL*5}×{CELL+LABEL_H} JPEG q={JPEG_Q}...')
frames_b64 = []
total_bytes = 0
for fi in range(1, N_FRAMES + 1):
    b64 = make_frame(fi)
    frames_b64.append(f'"data:image/jpeg;base64,{b64}"')
    total_bytes += len(b64)
    if fi % 30 == 0:
        print(f'  {fi}/{N_FRAMES}  total so far: {total_bytes/1e6:.2f} MB (base64)')

print(f'Total slideshow data: {total_bytes/1e6:.2f} MB (base64)')

new_frames_js = 'const FRAMES = [' + ',\n'.join(frames_b64) + '];\n'

print(f'\nReading HTML: {HTML_IN}  ({HTML_IN.stat().st_size/1e6:.1f} MB)')
html = HTML_IN.read_text(encoding='utf-8')

# Replace FRAMES array
pattern = r'const FRAMES = \[.*?\];'
m = re.search(pattern, html, re.DOTALL)
if not m:
    print('ERROR: could not find FRAMES array')
    sys.exit(1)
print(f'  Found FRAMES array at char {m.start()}–{m.end()} ({(m.end()-m.start())/1e6:.2f} MB)')
html = html[:m.start()] + new_frames_js + html[m.end():]

# Update label from "30-frame preview" to "150 frames"
html = html.replace(
    '5-Way Video — GT | v2_C | v2_D | v3_C | v3_D (30-frame preview)',
    '5-Way Video — GT | v2_C | v2_D | v3_C | v3_D (150 frames)'
)

HTML_IN.write_text(html, encoding='utf-8')
print(f'Written: {HTML_IN}  ({HTML_IN.stat().st_size/1e6:.1f} MB)')
print('DONE.')
