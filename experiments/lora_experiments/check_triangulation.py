import os, sys
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS')

import numpy as np
import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp

print("Loading pipeline...", flush=True)
pipeline = TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')
pipeline.cuda()
dec = pipeline.models['slat_decoder_mesh']
for p in dec.parameters(): p.requires_grad_(False)

print("Loading slat cache...", flush=True)
cache = np.load('/net/projects/ranalab/rajhansini/TRELLIS/experiments/results/phase7/slat_cache.npz')
slats  = cache['slats']
coords = cache['coords']
N = coords.shape[0]
coords_full = np.concatenate([np.zeros((N, 1), dtype=np.int32), coords], axis=1)
coords_t = torch.from_numpy(coords_full).cuda()

check_frames = [0, 24, 49, 74, 99, 124, 149]
results = []
for fi in check_frames:
    feats = torch.from_numpy(slats[fi]).cuda()
    st = sp.SparseTensor(feats=feats, coords=coords_t)
    with torch.no_grad():
        mesh = dec(st)[0]
    v = mesh.vertices.shape[0]
    f = mesh.faces.shape[0]
    results.append((fi+1, v, f))
    print(f"frame {fi+1:3d}: verts={v} faces={f}", flush=True)

verts = [r[1] for r in results]
faces = [r[2] for r in results]
print(f"Same triangulation: {len(set(verts))==1 and len(set(faces))==1}", flush=True)
print(f"verts range: {min(verts)}-{max(verts)}  faces range: {min(faces)}-{max(faces)}", flush=True)
