"""
Pinpoint exactly where the gradient dies.

Tests run from innermost to outermost:
  T1: index_add_  on zeros tensor — does grad flow from source?
  T2: get_dense_attrs scatter — does grad flow from feats?
  T3: nvdiffrast  — does grad flow from vertex_attrs to color?
  T4: full cube2mesh path with detached slat (no dec_mesh, no OOM)

Run:
  cd .../step8_train
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SPCONV_ALGO=native ATTN_BACKEND=xformers \
  python diagnose_grad.py 2>&1 | tee ../results/diagnose_grad.log
"""

import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SPCONV_ALGO']          = 'native'
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']      = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DEVICE = torch.device('cuda')


def bar(s): print(f'\n{"═"*65}\n  {s}\n{"═"*65}')
def ok(v): return '✓ nonzero' if (v is not None and v.abs().max() > 0) else '✗ EXACTLY ZERO'


# ══════════════════════════════════════════════════════════════════════════════
bar('T1: index_add_ on zeros — does grad flow from source?')

src = torch.randn(5, 3, device=DEVICE, requires_grad=True)
target = torch.zeros(10, 3, device=DEVICE)
idx = torch.tensor([0, 2, 5, 7, 9], device=DEVICE)

result = target.index_add_(0, idx, src)
loss = result.sum()
loss.backward()
print(f'  src.grad: {ok(src.grad)}')
print(f'  src.grad values: {src.grad}')
print(f'  result.requires_grad: {result.requires_grad}')
print(f'  result.grad_fn: {result.grad_fn}')


# ══════════════════════════════════════════════════════════════════════════════
bar('T2: get_dense_attrs scatter — does grad flow from feats?')

from trellis.representations.mesh.utils_cube import get_dense_attrs

feats2 = torch.randn(20, 4, device=DEVICE, requires_grad=True)
# use small res=8 to avoid OOM
coords2 = torch.randint(0, 8, (20, 3), device=DEVICE)
dense2 = get_dense_attrs(coords2, feats2, res=8, sdf_init=False)
print(f'  dense2.requires_grad: {dense2.requires_grad}  grad_fn: {dense2.grad_fn}')
loss2 = dense2.sum()
loss2.backward()
print(f'  feats2.grad: {ok(feats2.grad)}')


# ══════════════════════════════════════════════════════════════════════════════
bar('T3: nvdiffrast — does grad flow from vertex_attrs?')

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers.mesh_renderer import MeshRenderer
from trellis.representations.mesh.cube2mesh import MeshExtractResult

GT_FRAME_75 = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
               '/outputs/teapot_lava_kling_premium'
               '/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')
RENDER_RES = 518
_fx_n = 1.0 / (2.0 * math.tan(math.radians(20.0)))
INTRINSICS = torch.tensor([[_fx_n,0.,0.5],[0.,_fx_n,0.5],[0.,0.,1.]], dtype=torch.float32, device=DEVICE)
EXTRINSICS = torch.eye(4, dtype=torch.float32, device=DEVICE); EXTRINSICS[2, 3] = 2.0

renderer = MeshRenderer(
    rendering_options={'resolution': RENDER_RES, 'near': 0.1, 'far': 100., 'ssaa': 1},
    device=str(DEVICE),
)
img_pil = Image.open(GT_FRAME_75).convert('RGB')
gt = (torch.from_numpy(np.array(img_pil.resize((RENDER_RES, RENDER_RES))))
      .float().div(255.).permute(2, 0, 1).to(DEVICE))

# Get a real mesh from the pipeline (no grad needed here)
from trellis.modules import sparse as sp
from step6_5_lora.lora import insert_lora
from step6_5_lora.ray_attention import dual_path_ctx
from step1_input_prep.input_prep import load_frame, t_to_frame_idx, get_window_indices
from step2_dino_encoding.dino_encoding import load_dino, encode_frames
from step3_frame_weights.frame_weights import FrameWeightProjector, get_frame_weights
from step4_mcfm.mcfm import run_mcfm
import torch.nn as nn

SLAT_MEAN = torch.tensor([-2.1687545776367188,-0.004347046371549368,-0.13352349400520325,
    -0.08418072760105133,-0.5271206498146057,0.7238689064979553,-1.1414450407028198,1.2039363384246826])
SLAT_STD  = torch.tensor([2.377650737762451,2.386378288269043,2.124418020248413,
    2.1748552322387695,2.663944721221924,2.371192216873169,2.6217446327209473,2.684523105621338])

_t = np.linspace(1,0,26); _t = 3.0*_t/(1+2.0*_t); T_PAIRS=[(_t[i],_t[i+1]) for i in range(25)]

pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
pipeline.to(DEVICE)
flow_model = pipeline.models['slat_flow_model']
dec_mesh   = pipeline.models['slat_decoder_mesh']

dino = load_dino(DEVICE)
img75 = load_frame(75)
tokens = encode_frames([75], [img75], dino, DEVICE)
projector = FrameWeightProjector(k=3).to(DEVICE)
for p in projector.parameters(): p.requires_grad_(False)
t_val=74/149; fi=t_to_frame_idx(t_val); wi=get_window_indices(fi,3)
tw={i: tokens[75] for i in wi}
lv,_=get_frame_weights(t_val,fi,wi,tw,projector)
Kh,_=run_mcfm('v2',tw,wi,fi,lv); cond_gl=Kh.unsqueeze(0)
K_pooled=tokens[75]['tokens']
del dino, tokens, tw, projector; torch.cuda.empty_cache()

cond_gs = pipeline.get_cond([img_pil])
torch.manual_seed(42)
coords = pipeline.sample_sparse_structure(cond_gs, num_samples=1)
for k in ['sparse_structure_flow_model','sparse_structure_decoder']: del pipeline.models[k]
del cond_gs; torch.cuda.empty_cache()

lora_blocks, alpha = insert_lora(flow_model, rank=4)
lora_blocks = lora_blocks.to(DEVICE)
alpha = nn.Parameter(alpha.data.to(DEVICE))

torch.manual_seed(42)
x = sp.SparseTensor(feats=torch.randn(coords.shape[0], flow_model.in_channels, device=DEVICE), coords=coords)
flow_model.eval()
with torch.no_grad():
    with dual_path_ctx(flow_model, K_pooled, lora_blocks, alpha, E=None):
        for t,tp in T_PAIRS:
            tt=torch.tensor([1000.*t],device=DEVICE,dtype=torch.float32)
            v=flow_model(x,tt,cond_gl); x=x.replace(x.feats-(t-tp)*v.feats)
slat_feats_np = (x.feats * SLAT_STD.to(DEVICE) + SLAT_MEAN.to(DEVICE)).detach()

# Free everything except dec_mesh
for k in list(pipeline.models.keys()):
    if k != 'slat_decoder_mesh': pipeline.models[k].cpu(); del pipeline.models[k]
del flow_model, lora_blocks, alpha, K_pooled, cond_gl, x, v
torch.cuda.empty_cache()
print(f'  GPU after freeing flow_model: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB')

# Decode in NO_GRAD to get a real mesh with real vertex_attrs
dec_mesh.eval()
slat_for_decode = sp.SparseTensor(feats=slat_feats_np, coords=coords)
with torch.no_grad():
    decoded = dec_mesh(slat_for_decode)
mesh_real = decoded[0]
print(f'  mesh.success={mesh_real.success}  vertex_attrs shape={tuple(mesh_real.vertex_attrs.shape)}')

# Now detach vertex_attrs and reattach as leaf — test ONLY nvdiffrast grad
va = mesh_real.vertex_attrs.detach().requires_grad_(True)
# Create a fake MeshExtractResult with this leaf vertex_attrs
class FakeMesh:
    def __init__(self, m, va):
        self.vertices   = m.vertices
        self.faces      = m.faces
        self.vertex_attrs = va
        self.face_normal  = m.face_normal
        self.res        = m.res
        self.success    = m.success
fake_mesh = FakeMesh(mesh_real, va)

result = renderer.render(fake_mesh, EXTRINSICS, INTRINSICS, return_types=['color','mask'])
color = result['color']
mask  = result['mask']
print(f'  color range: [{color.min():.4f}, {color.max():.4f}]  coverage: {mask.float().mean()*100:.1f}%')
print(f'  color.requires_grad: {color.requires_grad}')

loss3 = F.mse_loss(color, gt)
loss3.backward()
print(f'  T3 result → va.grad: {ok(va.grad)}')
if va.grad is not None:
    print(f'  va.grad.max={va.grad.abs().max().item():.3e}  mean={va.grad.abs().mean().item():.3e}')


# ══════════════════════════════════════════════════════════════════════════════
bar('T4: cube2mesh alone — does grad flow from synthetic feats → vertex_attrs?')
# Feed synthetic feats (no transformer, no upsample) directly into SparseFeatures2Mesh.
# This isolates FlexiCubes from the sparse network entirely.

from trellis.representations.mesh.cube2mesh import SparseFeatures2Mesh

mesh_extractor = dec_mesh.mesh_extractor   # already on DEVICE

# Use coords from the real slat (real voxel structure)
# feats: shape (N_vox, 101) — out_channels of dec_mesh
N_vox4 = coords.shape[0]
feats4 = torch.randn(N_vox4, dec_mesh.out_channels, device=DEVICE, requires_grad=True)
sp_in  = sp.SparseTensor(feats=feats4, coords=coords)
# SparseTensor [i] returns a single-batch item; cube2mesh expects that
sp_item = sp_in[0]   # SparseTensor with coords[:,1:]

print(f'  feats4: shape={tuple(feats4.shape)}  requires_grad={feats4.requires_grad}')
torch.cuda.empty_cache()
print(f'  GPU before T4 cube2mesh: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB')

mesh4 = mesh_extractor(sp_item, training=False)
print(f'  mesh4.success={mesh4.success}')
if mesh4.vertex_attrs is None:
    print('  vertex_attrs=None — no color gradient path')
else:
    print(f'  vertex_attrs shape={tuple(mesh4.vertex_attrs.shape)}  requires_grad={mesh4.vertex_attrs.requires_grad}')
    loss4 = mesh4.vertex_attrs.sum()
    loss4.backward()
    t4_ok = ok(feats4.grad)
    print(f'  T4 result → feats4.grad: {t4_ok}')
    if feats4.grad is not None:
        print(f'  feats4.grad.max={feats4.grad.abs().max().item():.3e}  mean={feats4.grad.abs().mean().item():.3e}')

# ══════════════════════════════════════════════════════════════════════════════
bar('T5: sparse upsample backward — does grad flow through SparseSubdivide?')
# Run dec_mesh.upsample[0] + upsample[1] + out_layer with synthetic input.
# No FlexiCubes, no OOM risk.

import trellis.modules.sparse as sp2

# Read channel counts directly from the loaded dec_mesh
model_ch = dec_mesh.upsample[0].channels   # should be 768
print(f'  dec_mesh.upsample[0].channels = {model_ch}')
print(f'  dec_mesh.upsample[1].channels = {dec_mesh.upsample[1].channels}')
print(f'  dec_mesh.out_channels         = {dec_mesh.out_channels}')

# Create synthetic input in fp16 — upsample blocks have fp16 weights
# (in the real pipeline, SparseTransformerBase.forward does this cast internally)
N5 = coords.shape[0]
feats5_fp32 = torch.randn(N5, model_ch, device=DEVICE, dtype=torch.float32)
feats5 = feats5_fp32.half().requires_grad_(True)   # fp16 leaf, grad tracked
sp5 = sp2.SparseTensor(feats=feats5, coords=coords)
print(f'  Input: {tuple(sp5.feats.shape)}  requires_grad={feats5.requires_grad}')

torch.cuda.empty_cache()
print(f'  GPU before T5: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB')

dec_mesh.eval()
# Run upsample blocks + out_layer (no transformer, no FlexiCubes)
h5 = dec_mesh.upsample[0](sp5)
print(f'  after upsample[0]: {tuple(h5.feats.shape)}  rg={h5.feats.requires_grad}  gfn={h5.feats.grad_fn is not None}')
h5 = dec_mesh.upsample[1](h5)
print(f'  after upsample[1]: {tuple(h5.feats.shape)}  rg={h5.feats.requires_grad}  gfn={h5.feats.grad_fn is not None}')
h5 = h5.replace(h5.feats.float())   # fp16 → fp32 (same as dec_mesh.forward: h.type(x.dtype) where x=slat is fp32)
h5 = dec_mesh.out_layer(h5)
print(f'  after out_layer:   {tuple(h5.feats.shape)}  rg={h5.feats.requires_grad}  gfn={h5.feats.grad_fn is not None}')

loss5 = h5.feats.sum()
loss5.backward()
t5_ok = ok(feats5.grad)
print(f'  T5 result → feats5.grad: {t5_ok}')
if feats5.grad is not None:
    print(f'  feats5.grad.max={feats5.grad.abs().max().item():.3e}  mean={feats5.grad.abs().mean().item():.3e}')


# ══════════════════════════════════════════════════════════════════════════════
bar('T6: fp16 underflow — tiny gradient → upsample backward → zero?')
# T3 showed va.grad.max = 9.467e-06.
# fp16 minimum NORMAL value = 2^-14 ≈ 6.1e-05.
# fp16 minimum SUBNORMAL   = 2^-24 ≈ 6.0e-08.
# CUDA spconv may flush subnormals to zero (DAZ mode).
# Hypothesis: the render gradient underflows in fp16 upsample backward.

# We will test with two loss scales through the exact same upsample backward path.

def run_upsample_backward(scale, label):
    feat_t = feats5_fp32.half().requires_grad_(True)
    sp_t   = sp2.SparseTensor(feats=feat_t, coords=coords)
    h_t    = dec_mesh.upsample[0](sp_t)
    h_t    = dec_mesh.upsample[1](h_t)
    h_t    = h_t.replace(h_t.feats.float())
    h_t    = dec_mesh.out_layer(h_t)
    # sum() × scale mimics: render gradient arriving at out_layer output at magnitude 'scale'
    (h_t.feats.sum() * scale).backward()
    g = feat_t.grad
    print(f'  scale={scale:.0e}  feats.grad: {ok(g)}'
          + (f'  max={g.abs().max().item():.3e}' if g is not None else ''))

run_upsample_backward(1e+0, 'normal (sum)')
run_upsample_backward(1e-5, 'render-magnitude (9e-6)')
run_upsample_backward(1e-4, '10× render')
run_upsample_backward(1e-3, '100× render')
run_upsample_backward(1e-2, '1000× render')


# ══════════════════════════════════════════════════════════════════════════════
bar('T7: full dec_mesh transformer + upsample backward (no cube2mesh)')
# This tests the COMPLETE dec_mesh forward except cube2mesh.
# If feats7.grad is ZERO → transformer backward is broken.
# If feats7.grad is NONZERO → bug is in the composition with slat/LoRA in the training loop.

torch.cuda.empty_cache()
print(f'  GPU before T7: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB')

# Synthetic slat: same shape as real slat (7301 × latent_channels)
latent_ch = dec_mesh.input_layer.weight.shape[1] if hasattr(dec_mesh, 'input_layer') else 8
print(f'  latent_channels = {latent_ch}')

feats7 = torch.randn(coords.shape[0], latent_ch, device=DEVICE, dtype=torch.float32, requires_grad=True)
sp7 = sp2.SparseTensor(feats=feats7, coords=coords)
print(f'  Input: {tuple(sp7.feats.shape)}  requires_grad={feats7.requires_grad}')

dec_mesh.eval()
# Full forward: transformer → upsample → cast → out_layer (identical to dec_mesh.forward but stops before to_representation)
import trellis.models.structured_latent_vae.base as _base
h7 = _base.SparseTransformerBase.forward(dec_mesh, sp7)   # run only the transformer
print(f'  after transformer: {tuple(h7.feats.shape)}  rg={h7.feats.requires_grad}  gfn={h7.feats.grad_fn is not None}')
h7 = dec_mesh.upsample[0](h7)
print(f'  after upsample[0]: {tuple(h7.feats.shape)}  rg={h7.feats.requires_grad}')
h7 = dec_mesh.upsample[1](h7)
print(f'  after upsample[1]: {tuple(h7.feats.shape)}  rg={h7.feats.requires_grad}')
h7 = h7.replace(h7.feats.float())
h7 = dec_mesh.out_layer(h7)
print(f'  after out_layer:   {tuple(h7.feats.shape)}  rg={h7.feats.requires_grad}')

loss7 = h7.feats.sum()
loss7.backward()
t7_ok = ok(feats7.grad)
print(f'  T7 result → feats7.grad: {t7_ok}')
if feats7.grad is not None:
    print(f'  feats7.grad.max={feats7.grad.abs().max().item():.3e}  mean={feats7.grad.abs().mean().item():.3e}')


# ══════════════════════════════════════════════════════════════════════════════
bar('T7b: same as T7 but with use_checkpoint DISABLED')
# Hypothesis: torch.utils.checkpoint with use_reentrant=False + SparseTensor (non-Tensor input)
# silently zeros gradients because autograd only tracks tensor args.

ckpt_flags = [block.use_checkpoint for block in dec_mesh.blocks]
print(f'  dec_mesh.blocks[0].use_checkpoint = {ckpt_flags[0]}  (all: {set(ckpt_flags)})')

# Disable checkpoint on all blocks
for block in dec_mesh.blocks:
    block.use_checkpoint = False

torch.cuda.empty_cache()
feats7b = torch.randn(coords.shape[0], latent_ch, device=DEVICE, dtype=torch.float32, requires_grad=True)
sp7b = sp2.SparseTensor(feats=feats7b, coords=coords)

dec_mesh.eval()
h7b = _base.SparseTransformerBase.forward(dec_mesh, sp7b)
h7b = dec_mesh.upsample[0](h7b)
h7b = dec_mesh.upsample[1](h7b)
h7b = h7b.replace(h7b.feats.float())
h7b = dec_mesh.out_layer(h7b)

loss7b = h7b.feats.sum()
loss7b.backward()
t7b_ok = ok(feats7b.grad)
print(f'  T7b result → feats7b.grad: {t7b_ok}')
if feats7b.grad is not None:
    print(f'  feats7b.grad.max={feats7b.grad.abs().max().item():.3e}  mean={feats7b.grad.abs().mean().item():.3e}')

# Restore checkpoint flags
for block, flag in zip(dec_mesh.blocks, ckpt_flags):
    block.use_checkpoint = flag


# ══════════════════════════════════════════════════════════════════════════════
bar('T8: binary search — where in the transformer does gradient die?')

torch.cuda.empty_cache()

def test_partial_transformer(num_blocks, label):
    fi8 = torch.randn(coords.shape[0], latent_ch, device=DEVICE, dtype=torch.float32, requires_grad=True)
    sp8 = sp2.SparseTensor(feats=fi8, coords=coords)
    # input_layer
    h8 = dec_mesh.input_layer(sp8)
    h8.feats.retain_grad()
    g_after_input_layer = h8.feats
    # pos embedder + type cast
    h8 = h8 + dec_mesh.pos_embedder(sp8.coords[:, 1:])
    h8 = h8.type(dec_mesh.dtype)
    h8.feats.retain_grad()
    g_after_cast = h8.feats
    # run only first N blocks
    for i, block in enumerate(dec_mesh.blocks[:num_blocks]):
        h8 = block(h8)
    h8.feats.sum().backward()
    g0 = ok(fi8.grad)
    g1 = ok(g_after_input_layer.grad)
    g2 = ok(g_after_cast.grad)
    print(f'  [{label}]  fi8.grad={g0}  after_input_layer={g1}  after_cast={g2}')

# 0 blocks: just input_layer + pos_embed + cast
test_partial_transformer(0, '0 blocks (input_layer+cast only)')
# 1 block
test_partial_transformer(1, '1 block')
# 4 blocks
test_partial_transformer(4, '4 blocks')
# 8 blocks
test_partial_transformer(len(dec_mesh.blocks), f'all {len(dec_mesh.blocks)} blocks')


# ══════════════════════════════════════════════════════════════════════════════
bar('T9: transformer output → upsample, but DETACH to isolate spconv cache issue')
# T8: transformer alone = NONZERO
# T5: fresh fp16 leaf → upsample = NONZERO
# T7: transformer → upsample = ZERO
# T9a: detach transformer output, then feed to upsample
#   If NONZERO → transformer output SparseTensor's grad_fn chain breaks upsample backward
#   If ZERO   → upsample breaks when transformer's spconv internal state is present

torch.cuda.empty_cache()
fi9 = torch.randn(coords.shape[0], latent_ch, device=DEVICE, dtype=torch.float32, requires_grad=True)
sp9 = sp2.SparseTensor(feats=fi9, coords=coords)
dec_mesh.eval()

# Run full transformer (all blocks)
h9 = _base.SparseTransformerBase.forward(dec_mesh, sp9)
print(f'  transformer output: {tuple(h9.feats.shape)}  rg={h9.feats.requires_grad}  gfn={h9.feats.grad_fn is not None}')

# Detach the feats but KEEP the SparseTensor structure (same spconv state, spatial_cache, etc.)
h9_detached_feats = h9.feats.detach().requires_grad_(True)   # fp16 leaf
h9_leaf = h9.replace(h9_detached_feats)   # same SparseTensor structure, new leaf feats
print(f'  h9_leaf: rg={h9_leaf.feats.requires_grad}  gfn={h9_leaf.feats.grad_fn is not None}  (leaf: {h9_detached_feats.is_leaf})')

# Now run upsample on the detached (leaf) version
h9 = dec_mesh.upsample[0](h9_leaf)
h9 = dec_mesh.upsample[1](h9)
h9 = h9.replace(h9.feats.float())
h9 = dec_mesh.out_layer(h9)
h9.feats.sum().backward()
print(f'  T9a (detached feats, real spconv structure) → h9_detached_feats.grad: {ok(h9_detached_feats.grad)}')
if h9_detached_feats.grad is not None:
    print(f'  max={h9_detached_feats.grad.abs().max().item():.3e}')

# T9b: also test the full chain including fi9 (through the transformer grad_fn)
torch.cuda.empty_cache()
fi9b = torch.randn(coords.shape[0], latent_ch, device=DEVICE, dtype=torch.float32, requires_grad=True)
sp9b = sp2.SparseTensor(feats=fi9b, coords=coords)
h9b = _base.SparseTransformerBase.forward(dec_mesh, sp9b)
# Use a FRESH SparseTensor with the transformer's feats but no spconv indice state
h9b_fresh = sp2.SparseTensor(feats=h9b.feats, coords=h9b.coords)
print(f'  h9b_fresh: rg={h9b_fresh.feats.requires_grad}  gfn={h9b_fresh.feats.grad_fn is not None}')
h9b_fresh = dec_mesh.upsample[0](h9b_fresh)
h9b_fresh = dec_mesh.upsample[1](h9b_fresh)
h9b_fresh = h9b_fresh.replace(h9b_fresh.feats.float())
h9b_fresh = dec_mesh.out_layer(h9b_fresh)
h9b_fresh.feats.sum().backward()
print(f'  T9b (non-leaf feats, FRESH SparseTensor) → fi9b.grad: {ok(fi9b.grad)}')
if fi9b.grad is not None:
    print(f'  max={fi9b.grad.abs().max().item():.3e}')


bar('T10: binary search between transformer and upsample — where does grad die?')
# T7=ZERO (transformer→upsample chain breaks), T8=NONZERO (transformer alone works),
# T5=NONZERO (upsample with LEAF input works).
# T9b (fresh SparseTensor with non-leaf feats) should tell us if the issue is the
# SparseTensor's internal state (indice_dict/spatial_cache) vs. the non-leaf feats themselves.
# T10 tests narrow this further by summing BEFORE out_layer, isolating the upsample blocks.

# T10a: transformer alone → sum at transformer OUTPUT → feats10a.grad
# Confirms transformer backward works in this exact setup
torch.cuda.empty_cache()
feats10a = torch.randn(coords.shape[0], latent_ch, device=DEVICE, dtype=torch.float32, requires_grad=True)
sp10a = sp2.SparseTensor(feats=feats10a, coords=coords)
dec_mesh.eval()
h10a = _base.SparseTransformerBase.forward(dec_mesh, sp10a)
h10a.feats.retain_grad()
h10a.feats.sum().backward()
print(f'  T10a transformer→sum→feats10a.grad: {ok(feats10a.grad)}  h10a_feats.grad: {ok(h10a.feats.grad)}')
if feats10a.grad is not None:
    print(f'        feats10a max={feats10a.grad.abs().max().item():.3e}')

# T10b: transformer → upsample[0] → sum (NO out_layer)
# If ZERO: upsample[0] breaks when receiving non-leaf transformer output
# If NONZERO: upsample[0] is fine, issue is downstream
torch.cuda.empty_cache()
feats10b = torch.randn(coords.shape[0], latent_ch, device=DEVICE, dtype=torch.float32, requires_grad=True)
sp10b = sp2.SparseTensor(feats=feats10b, coords=coords)
dec_mesh.eval()
h10b = _base.SparseTransformerBase.forward(dec_mesh, sp10b)
h10b = dec_mesh.upsample[0](h10b)
h10b.feats.sum().backward()
print(f'  T10b transformer→upsample[0]→sum→feats10b.grad: {ok(feats10b.grad)}')
if feats10b.grad is not None:
    print(f'        feats10b max={feats10b.grad.abs().max().item():.3e}')

# T10c: transformer → upsample[0] → upsample[1] → sum (NO out_layer)
torch.cuda.empty_cache()
feats10c = torch.randn(coords.shape[0], latent_ch, device=DEVICE, dtype=torch.float32, requires_grad=True)
sp10c = sp2.SparseTensor(feats=feats10c, coords=coords)
dec_mesh.eval()
h10c = _base.SparseTransformerBase.forward(dec_mesh, sp10c)
h10c = dec_mesh.upsample[0](h10c)
h10c = dec_mesh.upsample[1](h10c)
h10c.feats.sum().backward()
print(f'  T10c transformer→upsample[0,1]→sum→feats10c.grad: {ok(feats10c.grad)}')
if feats10c.grad is not None:
    print(f'        feats10c max={feats10c.grad.abs().max().item():.3e}')

# T10d: transformer → upsample[0] → upsample[1] → out_layer → sum
# Should match T7 exactly
torch.cuda.empty_cache()
feats10d = torch.randn(coords.shape[0], latent_ch, device=DEVICE, dtype=torch.float32, requires_grad=True)
sp10d = sp2.SparseTensor(feats=feats10d, coords=coords)
dec_mesh.eval()
h10d = _base.SparseTransformerBase.forward(dec_mesh, sp10d)
h10d = dec_mesh.upsample[0](h10d)
h10d = dec_mesh.upsample[1](h10d)
h10d = h10d.replace(h10d.feats.float())
h10d = dec_mesh.out_layer(h10d)
h10d.feats.sum().backward()
print(f'  T10d transformer→up[0,1]→out_layer→sum→feats10d.grad: {ok(feats10d.grad)}  (should match T7)')
if feats10d.grad is not None:
    print(f'        feats10d max={feats10d.grad.abs().max().item():.3e}')


bar('Summary')
print('  T1 index_add_    :', ok(src.grad))
print('  T2 get_dense_attrs:', ok(feats2.grad))
print('  T3 nvdiffrast    :', ok(va.grad))
t4_grad = feats4.grad if 'feats4' in dir() else None
print('  T4 cube2mesh     :', ok(t4_grad))
t5_grad = feats5.grad if 'feats5' in dir() else None
print('  T5 upsample+out  :', ok(t5_grad))
t7_grad  = feats7.grad  if 'feats7'  in dir() else None
t7b_grad = feats7b.grad if 'feats7b' in dir() else None
print('  T7  transformer+up (checkpoint ON) :', ok(t7_grad))
print('  T7b transformer+up (checkpoint OFF):', ok(t7b_grad))
print()
print('  T7=ZERO, T7b=NONZERO → use_checkpoint=True + SparseTensor kills gradients → fix: disable checkpoint in dec_mesh before backward')
print('  T7=ZERO, T7b=ZERO    → transformer blocks have a different backward bug')
