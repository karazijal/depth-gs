# %%
import torch
from scene import GaussianModel, Scene
from types import SimpleNamespace

from diff_gaussian_rasterization_depth_acc_mod import depth

args = SimpleNamespace(
    source_path = 'data/nerf_llff_fewshot_resize/fern',
    model_path = 'output/baseline-ferm_baseline_3dgs_fern/',
    resolution = 1,
    images = None,
    white_background = False,
    seed=42,
    kshot=5,
    eval=True,
    data_device='cuda'
)
model = GaussianModel(3)
scene = Scene(args, model, load_iteration=-1)

# %%
cameras = scene.getTrainCameras()
# %%

# %%
import importlib
import gaussian_renderer
importlib.reload(gaussian_renderer)
from gaussian_renderer import render as vanilla_render
# %%
pipe = SimpleNamespace(
    debug=False,
    compute_cov3D_python=False,
    convert_SHs_python=False,
)

render_pkg = vanilla_render(cameras[0], model, pipe, torch.zeros(3).to('cuda'), 1.0)
# %%
from matplotlib import pyplot as plt
img = render_pkg['render'].detach().permute(1, 2, 0).cpu().numpy().clip(0, 1)
img.shape
plt.imshow(img)
# %%

# %%
import math
import diff_gaussian_rasterization_depth_acc_mod
import diff_gaussian_rasterization_depth_acc_mod.depth as depth
importlib.reload(diff_gaussian_rasterization_depth_acc_mod)
importlib.reload(depth)

viewpoint_camera = cameras[0]
pc = model
screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
try:
    screenspace_points.retain_grad()
except:
    pass

# Set up rasterization configuration
tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
scaling_modifier=1.

bg_color = torch.zeros(3, device='cuda')
from diff_gaussian_rasterization_depth_acc_mod import GaussianRasterizationSettings
raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=bg_color,
    scale_modifier=scaling_modifier,
    viewmatrix=viewpoint_camera.world_view_transform.contiguous(),
    projmatrix=viewpoint_camera.full_proj_transform.contiguous(),
    sh_degree=pc.active_sh_degree,
    campos=viewpoint_camera.camera_center.contiguous(),
    prefiltered=False,
    debug=pipe.debug
)

means3D = pc.get_xyz.contiguous()
means2D = screenspace_points
opacity = pc.get_opacity.contiguous()

colors_precomp = torch.Tensor([])
cov3Ds_precomp = torch.Tensor([])

scales = pc.get_scaling.contiguous()
rotations = pc.get_rotation.contiguous()
sh = pc.get_features.contiguous()
opacity = pc.get_opacity.contiguous()

gt_depth = viewpoint_camera.original_depth.clone().cuda().float().contiguous()
print(gt_depth.shape, raster_settings.image_height, raster_settings.image_width)


# %%
# rc, loss, radii, precision, gamma, sigma, *buffers = depth._debug_rasterize_depth_gaussians(
#     means3D,
#     scales,
#     rotations,
#     opacity,
#     gt_depth,
#     raster_settings,
# )

# # plot loss, gamma and sigma as images
# plt.figure(dpi=150, figsize=(15, 4))
# plt.subplot(131)
# plt.imshow(loss.detach().cpu().numpy())
# plt.colorbar()
# plt.axis('off')
# plt.title('$\mathcal{L}$')

# plt.subplot(132)
# plt.imshow(gamma.detach().cpu().numpy())
# plt.colorbar()
# plt.axis('off')
# plt.title('$\gamma$')

# plt.subplot(133)
# plt.imshow(sigma.detach().cpu().numpy())
# plt.colorbar()
# plt.axis('off')
# plt.title('$\sigma$')

# %%
ret_grads, loss, prec, gamma, sigma = depth._debug_rasterize_depth_gaussians(
    means3D,
    scales,
    rotations,
    opacity,
    gt_depth,
    raster_settings,
)
# %%
from matplotlib import pyplot as plt

print('prec', prec.shape, prec.dtype, torch.isfinite(prec).all(), torch.all(prec==0))
for i, r in enumerate(ret_grads):
    if r is not None:
        print(i, r.shape, r.dtype, torch.isfinite(r).all(), torch.all(r==0))
    else:
        print(i, "None")
print()

# plot loss, gamma and sigma as images
plt.figure(dpi=150, figsize=(15, 4))
plt.subplot(131)
plt.imshow(loss.detach().cpu().numpy())
plt.colorbar()
plt.axis('off')
plt.title('$\mathcal{L}$')

plt.subplot(132)
plt.imshow(gamma.detach().cpu().numpy())
plt.colorbar()
plt.axis('off')
plt.title('$G$')

plt.subplot(133)
plt.imshow(sigma.detach().cpu().numpy())
plt.colorbar()
plt.axis('off')
plt.title('$S$')


# %%
plt.imshow(ret_grads[2].detach().cpu().numpy())
plt.colorbar()

# %%
nll, _ = depth.rasterize_depth_gaussians(
    means3D,
    scales,
    rotations,
    opacity,
    gt_depth,
    raster_settings,
)
# %%
nll.shape
# %%
nll.mean().backward()
# %%
print(f"{torch.isfinite(pc._xyz.grad).all()=}")
print(f"{torch.isfinite(pc._rotation.grad).all()=}")
print(f"{torch.isfinite(pc._scaling.grad).all()=}")
print(f"{torch.isfinite(pc._opacity.grad).all()=}")

print(f"{torch.all(pc._xyz.grad==0)=}")
print(f"{torch.all(pc._rotation.grad==0)=}")
print(f"{torch.all(pc._scaling.grad==0)=}")
print(f"{torch.all(pc._opacity.grad==0)=}")
# %%
plt.imshow(1./ cameras[5].original_depth.detach().cpu(), cmap='jet')
plt.axis('off')
# %%
