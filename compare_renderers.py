# %%
import torch
from scene import GaussianModel, Scene
from types import SimpleNamespace

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
import gsplat_renderer
import importlib
importlib.reload(gsplat_renderer)
from gsplat_renderer import render as gsplat_render

gsplat_pkg = gsplat_render(cameras[0], model, pipe, torch.zeros(3).to('cuda'), 1.0)
gsplat_img = gsplat_pkg['render'].detach().permute(1, 2, 0).cpu().numpy().clip(0, 1)
plt.imshow(gsplat_img)
# %%
import numpy as np
plt.imshow(np.power(img - gsplat_img, 2).sum(-1))

# %%
diff_radii = (gsplat_pkg['radii'] * gsplat_pkg['visibility_filter'] - render_pkg['radii'] * render_pkg['visibility_filter']).float().pow(2).detach().cpu().numpy()
plt.plot(diff_radii)
plt.title('Difference in radii of returned Gaussians')
# %%

gsplat_pkg['render'].sum().backward()
render_pkg['render'].sum().backward()

grads = render_pkg['viewspace_points'].grad[:, :2].norm(dim=-1) * render_pkg['visibility_filter']
gsplat_grads = gsplat_pkg['viewspace_points'].grad[:, :2].norm(dim=-1) * gsplat_pkg['visibility_filter']

plt.plot((grads -gsplat_grads).float().pow(2).detach().cpu().numpy())
# %%
plt.title("Squared erorr of returned screenspace gradient norms for Gaussians")
plt.plot((grads -gsplat_grads).float().pow(2).detach().cpu().numpy())
# %%
render_pkg['viewspace_points'].shape
# %%
# render_pkg['viewspace_points']
# %%
# for i in range(100000):
#     gsplat_pkg_new = gsplat_render(cameras[0], model, pipe, torch.zeros(3).to('cuda'), 1.0)
#     diff_radii = (gsplat_pkg['radii'] * gsplat_pkg['visibility_filter'] - gsplat_pkg_new['radii'] * gsplat_pkg_new['visibility_filter']).float().pow(2).detach().cpu().numpy()
#     diff = diff_radii > 0.00000001
#     if np.any(diff):
#         print(diff_radii.mean())
# %%
from diff_gaussian_rasterization_mod import _C as mod_backend
from diff_gaussian_rasterization_mod import GaussianRasterizationSettings
# %%
import math
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

raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=bg_color,
    scale_modifier=scaling_modifier,
    viewmatrix=viewpoint_camera.world_view_transform.contigous(),
    projmatrix=viewpoint_camera.full_proj_transform.contigous(),
    sh_degree=pc.active_sh_degree,
    campos=viewpoint_camera.camera_center.contiguous(),
    prefiltered=False,
    debug=pipe.debug
)

means3D = pc.get_xyz.contiguous()
means2D = screenspace_points
opacity = pc.get_opacity.contiguous()

colors_precomp = None
cov3Ds_precomp = None

scales = pc.get_scaling.contiguous()
rotations = pc.get_rotation.contiguous()
sh = pc.get_features.contiguous()
opacity = pc.get_opacity.contiguous()

ret = mod_backend.rasterize_gaussians_depth_loss(
    raster_settings.bg, 
    means3D,
    colors_precomp,
    opacity,
    scales,
    rotations,
    raster_settings.scale_modifier,
    cov3Ds_precomp,
    raster_settings.viewmatrix,
    raster_settings.projmatrix,
    raster_settings.tanfovx,
    raster_settings.tanfovy,
    raster_settings.image_height,
    raster_settings.image_width,
    sh,
    raster_settings.sh_degree,
    raster_settings.campos,
    raster_settings.prefiltered,
    raster_settings.debug
)