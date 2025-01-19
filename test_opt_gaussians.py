# %%
import torch
import math
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

device = torch.device("cuda:0")
N = 1000
xyz = torch.rand(N,3, dtype=torch.float32, device=device).view(-1,3) * 2
# # xyz = torch.tensor([[0, 1, 0.5]], device=device).repeat(N, 1) * 2
xyz[..., 0] -= 1
xyz[..., 1] -= 1
xyz[..., 2] *= 4
xyz[..., 2] += 2

# # xyz[..., 0] *= 2
# # xyz[..., 1] *= 2
# # Since extrinsics is identity 
# # depth sort 
depth = xyz[..., 2]
ind = torch.argsort(depth, descending=False)
xyz = xyz[ind]

scales = torch.rand(N, 3, device=device).view(-1,3) * 0.2
quats = torch.rand(N, 4, device=device).view(-1,4)
quats /= quats.norm(dim=1, keepdim=True).clamp(min=1e-6)

# xyz = torch.tensor([[ 0.5607, -0.7803,  6.0269]], device='cuda:0', requires_grad=True)
# scales = torch.tensor([[1.5013, 0.7892, 0.5560]], device='cuda:0', requires_grad=True)
# quats = torch.tensor([[0.4965, 0.4484, 0.6553, 0.3507]], device='cuda:0', requires_grad=True)

# xyz = torch.tensor([[ 0.8954, 0.9857, 1.7505]], device='cuda:0', requires_grad=True)
# scales = torch.tensor([[0.2153, 1.3674, 1.9875]], device='cuda:0', requires_grad=True)
# quats = torch.tensor([[0.3264, 0.7270, 0.6041, 0.0047]], device='cuda:0', requires_grad=True)

colors = torch.rand(N, 3, device=device).view(-1,3)
opacity = torch.rand(N, 1, device=device).view(-1,1).clamp(min=0.99)


f = 300
fx = f
fy = f

cx = 128
cy = 128

img_height = 256
img_width = 256

glob_scale = 1.0
R = torch.eye(3).numpy()
T = torch.tensor([0, 0, 0]).numpy()
trans = torch.tensor([0, 0, 0]).numpy()
scale = 1.0
znear = 0.01
zfar = 100.0

FoVx = 2 * math.atan(img_width / (2 * fx))
FoVy = 2 * math.atan(img_height / (2 * fy))

world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

def generate_gaussian_tensor(size, mean, std):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    x, y = torch.meshgrid(x, y)
    gaussian = torch.exp(-((x-mean)**2 + (y-mean)**2) / (2*std**2))
    return gaussian

size = 256
mean = 0.0
std = 0.3
gaussian_tensor = generate_gaussian_tensor(size, mean, std) * 5 + 4

gt_depth = torch.ones(256, 256, device="cuda") * 5
# gt_depth = gaussian_tensor.cuda()

world_view_transform
#%% 
def compute_sigma(rot, s):
    r, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    R = torch.stack([
        torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)], dim=-1),
        torch.stack([2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)], dim=-1),
        torch.stack([2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)
    ], dim=-2)

    S = torch.diag_embed(s)

    M = torch.bmm(S, R)
    Sigma = torch.bmm(M.transpose(1, 2), M)

    return Sigma


def scale_rot_to_cov3d(scales, quats):
    quats = quats / (quats.norm(dim=1, keepdim=True).clamp(min=1e-6))
    return compute_sigma(quats, scales)

def symmetric_to_utril(matrix):
    utril_indices = torch.triu_indices(matrix.shape[-2], matrix.shape[-1])
    utril_vector = matrix[..., utril_indices[0], utril_indices[1]]
    return utril_vector


# %%
import math
from diff_gaussian_rasterization_depth_acc_mod import GaussianRasterizationSettings, GaussianRasterizer

tanfovx = math.tan(FoVx * 0.5)
tanfovy = math.tan(FoVy * 0.5)

raster_settings = GaussianRasterizationSettings(
    image_height=img_height,
    image_width=img_width,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=xyz.new_zeros(3),
    scale_modifier=1,
    viewmatrix=world_view_transform.contiguous(),
    projmatrix=full_proj_transform.contiguous(),
    sh_degree=1,
    campos=xyz.new_zeros(3),
    prefiltered=False,
    debug=False
)

def render(
        means3D,
        opacity,
        scales, 
        rotations,
        colors_precomp,

):
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, depth, acc, radii = rasterizer(
        means3D = means3D.contiguous(),
        means2D = screenspace_points,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity.contiguous(),
        scales = scales.contiguous(),
        rotations = rotations.contiguous(),
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "acc": acc}

ret = render(
    means3D=xyz,
    opacity=opacity,
    scales=scales,
    rotations=quats,
    colors_precomp=colors
)

from matplotlib import pyplot as plt
img = ret["render"].detach().cpu().numpy().transpose(1, 2, 0)
alp = ret["acc"].detach().cpu().numpy()
dis = (ret['depth']).detach().cpu().numpy()
plt.figure(figsize=(20, 5))
plt.subplot(141)
plt.imshow(img)
plt.axis('off')
plt.subplot(142)
plt.imshow(alp, cmap='viridis')
plt.colorbar()
plt.axis('off')
plt.subplot(143)
plt.imshow(dis, cmap='jet', vmin=1, vmax=12)
plt.colorbar()
plt.axis('off')
plt.subplot(144)
plt.imshow(gt_depth.cpu().numpy(), cmap='jet', vmin=1, vmax=12)
plt.colorbar()
plt.axis('off')

# %%
from diff_gaussian_rasterization_depth_acc_mod import depth

nll = depth.rasterize_depth_gaussians(
    xyz,
    scales,
    quats,
    opacity,
    gt_depth,
    raster_settings=raster_settings
)[0]


# %%
plt.imshow(nll.detach().cpu().numpy())
plt.colorbar()
# %%
nll.shape
# %%
from IPython.display import display, clear_output
import io
from PIL import Image
import mediapy as media
import numpy as np

import torch.nn as nn
mu = nn.Parameter(xyz, requires_grad=True)
S = nn.Parameter(scales, requires_grad=True)
q = nn.Parameter(quats, requires_grad=True)
o = nn.Parameter(opacity, requires_grad=True)
opt = torch.optim.Adam([
    {"params": mu, 'lr': 0.00016, 'name': 'xyz'},
    {"params": S, 'lr': 0.005, 'name': 'scales'},
    {"params": q, 'lr': 0.001, 'name': 'quats'},
    {"params": o, 'lr': 0.05 , 'name': 'opacity'}
], eps=1e-15)

spatial_lr_scale = 1.0
min_iters = 1500
position_lr_init = 0.00016
position_lr_final = 0.0000016
position_lr_delay_mult = 0.01
position_lr_max_steps = 30_000
from utils.general_utils import get_expon_lr_func
xyz_scheduler_args = get_expon_lr_func(lr_init=position_lr_init*spatial_lr_scale,
                                        lr_final=position_lr_final*spatial_lr_scale,
                                        lr_delay_mult=position_lr_delay_mult,
                                        max_steps=position_lr_max_steps)


frames = []
for i in range(10000):
    opt.zero_grad()
    for param_group in opt.param_groups:
        if param_group["name"] == "xyz":
            lr = xyz_scheduler_args(i)
            param_group['lr'] = lr
    opa = o.sigmoid()
    nll = depth.rasterize_depth_gaussians(
        mu,
        S,
        q / q.norm(dim=1, keepdim=True).clamp(min=1e-6),
        opa,
        gt_depth,
        raster_settings=raster_settings
    )[0]
    loss = nll.sum()
    loss.backward()
    opt.step()

    # print(loss.item())
    if i % 10 == 0:
        clear_output(wait=True)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ret = render(
            means3D=xyz,
            opacity=opa,
            scales=scales,
            rotations=quats,
            colors_precomp=colors
        )
        img = ret["render"].detach().cpu().numpy().transpose(1, 2, 0)
        alp = ret["acc"].detach().cpu().numpy()
        dis = (ret['depth']).detach().cpu().numpy()

        ax[0].imshow(img)
        ax[0].axis('off')
        ax[0].set_title('RGB')

        ax[1].imshow(dis, cmap='jet', vmin=1, vmax=12)
        ax[1].axis('off')
        ax[1].set_title('Depth')

        ax[2].imshow(nll.detach().cpu().numpy())
        ax[2].axis('off')
        ax[2].set_title('NLL')

        # display(fig)
        with io.BytesIO() as buf:
            fig.savefig(buf, format='png')
            buf.seek(0)
            frame = Image.open(buf).convert('RGB')
        plt.close(fig)
        display(frame)
        frames.append(frame)
# %%
media.show_video([np.array(f) for f in frames], fps=10)

# %%

plt.imshow(gaussian_tensor, cmap='jet', vmin=0, vmax=12)
# %%
gaussian_tensor.min(), gaussian_tensor.max()
# %%
