# %%
import torch
import math
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

device = torch.device("cuda:0")
xyz = torch.tensor([
    [0,0,10], 
    [-1, -1, 5],
    [1, -1, 5],
    [-1, 1, 5],
    [1, 1, 5],
], dtype=torch.float32, device=device).view(-1,3) * 0.5
scales = torch.tensor([
    [10, 10, 1], 
    [1, 1, 1], 
    [1, 1, 1], 
    [1, 1, 1], 
    [1, 1, 1]
], dtype=torch.float32, device=device).view(-1,3) * 0.25
quats = torch.tensor([
    [1,0,0,0], 
    [1,0,0,0], 
    [1,0,0,0], 
    [1,0,0,0], 
    [1,0,0,0]
], dtype=torch.float32, device=device).view(-1,4)

colors = torch.tensor([
    [1, 1, 1],
    [1, 0, 0], 
    [0, 1, 0],
    [0, 0, 1], 
    [1, 1, 0]
    ], dtype=torch.float32, device=device).view(-1,3)
opacity = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32, device=device).view(-1,1)


gt_depth = torch.ones(256, 256, device="cuda") * 6
gt_depth.requires_grad_(True)

xyz.requires_grad_(True)
scales.requires_grad_(True)
quats.requires_grad_(True)
colors.requires_grad_(True)
opacity.requires_grad_(True)


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
world_view_transform
#%% 
import torch

def quaternion_to_rotation_matrix(q):
    q_r, q_i, q_j, q_k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    
    R[..., 0, 0] = 0.5 - (q_j * q_j + q_k * q_k)
    R[..., 0, 1] = q_i * q_j - q_r * q_k
    R[..., 0, 2] = q_i * q_k + q_r * q_j
    
    R[..., 1, 0] = q_i * q_j + q_r * q_k
    R[..., 1, 1] = 0.5 - (q_i * q_i + q_k * q_k)
    R[..., 1, 2] = q_j * q_k - q_r * q_i
    
    R[..., 2, 0] = q_i * q_k - q_r * q_j
    R[..., 2, 1] = q_j * q_k + q_r * q_i
    R[..., 2, 2] = 0.5 - (q_i * q_i + q_j * q_j)
    
    return 2 * R


def scale_rot_to_cov3d(scales, quats):
    S = scales.clamp(min=1e-6).diag_embed()
    quats = quats / (quats.norm(dim=1, keepdim=True).clamp(min=1e-6))
    R = quaternion_to_rotation_matrix(quats)
    print(S.shape, R.shape)

    M = R @ S
    
    return M@M.transpose(-1, -2)

def symmetric_to_utril(matrix):
    utril_indices = torch.triu_indices(matrix.shape[-2], matrix.shape[-1])
    utril_vector = matrix[..., utril_indices[0], utril_indices[1]]
    return utril_vector

cov3d = scale_rot_to_cov3d(scales, quats)
cov3d_utril = symmetric_to_utril(cov3d)
cov3d.inverse()
# def create_ellipsoid(covariance_tensor, mean_tensor, num_points=100):
#     # Ensure tensors are float type for eigen decomposition
#     covariance_tensor = covariance_tensor.float()
    
#     # Compute eigenvalues and eigenvectors using PyTorch
#     eigenvalues, eigenvectors = torch.linalg.eigh(covariance_tensor)

#     # Generate points on a unit sphere
#     u = torch.linspace(0, 2 * torch.pi, num_points)
#     v = torch.linspace(0, torch.pi, num_points)
#     x = torch.outer(torch.cos(u), torch.sin(v))
#     y = torch.outer(torch.sin(u), torch.sin(v))
#     z = torch.outer(torch.ones(num_points), torch.cos(v))
    
#     # Scale points by the square roots of the eigenvalues
#     x = x * torch.sqrt(eigenvalues[0])
#     y = y * torch.sqrt(eigenvalues[1])
#     z = z * torch.sqrt(eigenvalues[2])

#     # Rotate points and add mean
#     xyz = torch.stack([x, y, z]).reshape(3, -1)
#     xyz = torch.mm(eigenvectors, xyz).reshape(3, num_points, num_points)
#     x, y, z = xyz[0] + mean_tensor[0], xyz[1] + mean_tensor[1], xyz[2] + mean_tensor[2]

#     # Convert to NumPy arrays for Plotly
#     return x.numpy(), y.numpy(), z.numpy()

# import plotly.graph_objects as go
# import numpy as np

# fig = go.Figure()

# for i in range(xyz.shape[0]):
#     c = cov3d_utril[i].detach().cpu()
#     cov3d = torch.zeros(3, 3)

#     cov3d[0, 0] = c[0]
#     cov3d[0, 1] = c[1]
#     cov3d[0, 2] = c[2]
#     cov3d[1, 1] = c[3]
#     cov3d[1, 2] = c[4]
#     cov3d[2, 2] = c[5]
#     cov3d[1, 0] = c[1]
#     cov3d[2, 0] = c[2]
#     cov3d[2, 1] = c[4]

#     x, y, z = create_ellipsoid(cov3d, xyz[i].detach().cpu(), num_points=100)
#     clr_scale = f"rgb{tuple(colors[i].detach().cpu().tolist())}"
#     clr_scale = [[0, clr_scale], [1, clr_scale]]
#     fig.add_trace(go.Surface(
#         x=x, y=y, z=z, colorscale=clr_scale, surfacecolor=np.ones_like(z), opacity=opacity[0].item(), showscale=False
#         ))
# fig.update_layout(title='3D Ellipsoid with PyTorch', autosize=False,
#                   width=700, height=700)
# fig.show()
# %%
import math
from diff_gaussian_rasterization_depth_acc_mod import GaussianRasterizationSettings, GaussianRasterizer
def render(
        means3D,
        opacity,
        scales, 
        rotations,
        colors_precomp,

):
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=img_height,
        image_width=img_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=means3D.new_zeros(3),
        scale_modifier=1,
        viewmatrix=world_view_transform.contiguous(),
        projmatrix=full_proj_transform.contiguous(),
        sh_degree=1,
        campos=means3D.new_zeros(3),
        prefiltered=False,
        debug=False
    )

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
# %%
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
dis = (1./ret['depth']).detach().cpu().numpy()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.subplot(132)
plt.imshow(alp, cmap='viridis')
plt.colorbar()
plt.axis('off')
plt.subplot(133)
plt.imshow(dis, cmap='jet')
plt.colorbar()
plt.axis('off')

# %%


cov3d = scale_rot_to_cov3d(scales, quats)
prec = symmetric_to_utril(cov3d.inverse().detach())

from diff_gaussian_rasterization_depth_acc_mod import depth

def test_fn(
        tx_xyz,
        precisions,
        opacities,
        gt_depth):
    
    means3D = xyz.clone().detach().contiguous()
    scaling = scales.clone().detach().contiguous()
    rots = quats.clone().detach().contiguous()
    rots = rots / rots.norm(dim=1, keepdim=True).clamp(min=1e-6)

    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=img_height,
        image_width=img_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=means3D.new_zeros(3),
        scale_modifier=1,
        viewmatrix=world_view_transform.contiguous(),
        projmatrix=full_proj_transform.contiguous(),
        sh_degree=1,
        campos=means3D.new_zeros(3),
        prefiltered=False,
        debug=False
    )
    return depth._RasterizeDepthGaussiansNoTx.apply(
        tx_xyz,
        precisions,
        opacities,
        gt_depth,
        raster_settings,
        means3D,
        scaling,
        rots
    ).sum()

tx_precisions = prec.clone().detach()
tx_means = xyz.clone().detach()
tx_opacities = opacity.clone().detach()
tx_gt_depth = gt_depth.clone().detach()

tx_precisions.requires_grad_(True)
tx_means.requires_grad_(True)
tx_opacities.requires_grad_(True)
tx_gt_depth.requires_grad_(True)

inputs = (tx_means, tx_precisions, tx_opacities, gt_depth)
with torch.no_grad():
    eps = 1e-6
    grads = []
    for idx, inp in enumerate(inputs):
        perturb = torch.zeros_like(inp).view(-1)
        num_grad = torch.zeros_like(perturb).view(-1)
        for i in range(len(perturb)):
            perturb[i] = eps
            plus_input = [x if j!= idx else x + perturb.view(x.shape) for j,x in enumerate(inputs)]
            out_plus = test_fn(*plus_input)
            minu_input = [x if j!= idx else x - perturb.view(x.shape) for j,x in enumerate(inputs)]
            out_minu = foo(*minu_input)
            num_grad[i] = (out_plus - out_minu) / (2 * eps)
            perturb[i] = 0
        num_grad = num_grad.view(inp.shape)
        grads.append(num_grad)

nll = test_fn(
    xyz,
    prec,
    opacity,
    gt_depth
)

nll.backward()

# %%
# from diff_gaussian_rasterization_depth_acc_mod import depth

# def foo(
#     means3D,
#     opacity,
#     scales, 
#     rotations,
#     gt_depth,
#     ):
#     tanfovx = math.tan(FoVx * 0.5)
#     tanfovy = math.tan(FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=img_height,
#         image_width=img_width,
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=means3D.new_zeros(3),
#         scale_modifier=1,
#         viewmatrix=world_view_transform.contiguous(),
#         projmatrix=full_proj_transform.contiguous(),
#         sh_degree=1,
#         campos=means3D.new_zeros(3),
#         prefiltered=False,
#         debug=False
#     )
#     return depth.rasterize_depth_gaussians(
#         means3D,
#         scales,
#         rotations,
#         opacity,
#         gt_depth,
#         raster_settings
#     )[0].sum()

# %%

# %%

# %%
grads[0]
# %%
nll = foo(means3D=xyz, opacity=opacity, scales=scales, rotations=quats, gt_depth=gt_depth)
nll.backward()
xyz.grad
# %%
