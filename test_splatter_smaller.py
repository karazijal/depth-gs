# %%
import torch
import math
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

device = torch.device("cuda:0")
xyz = torch.tensor([
    [-1, -1, 5],
    # [1, -1, 5],
    # [-1, 1, 5],
    [1, 1, 5],
    # [0,0,10], 
], dtype=torch.float32, device=device).view(-1,3) * 0.5
scales = torch.tensor([
    [1, 5, 1], 
    # [1, 1, 1], 
    # [1, 1, 1], 
    [5, 1, 1],
    # [10, 10, 1], 
], dtype=torch.float32, device=device).view(-1,3) * 0.25
quats = torch.tensor([
    [1,0,0,0], 
    # [1,0,0,0], 
    # [1,0,0,0], 
    [1,0,0,0], 
    # [1,0,0,0]
], dtype=torch.float32, device=device).view(-1,4)

colors = torch.tensor([
    [1, 0, 0], 
    # [0, 1, 0],
    # [0, 0, 1], 
    [1, 1, 0],
    # [1, 1, 1],
    ], dtype=torch.float32, device=device).view(-1,3)
opacity = torch.tensor([
    1, 
    # 1, 
    # 1, 
    1, 
    # 1
     ], dtype=torch.float32, device=device).view(-1,1)


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


def get_bounds(p_x, p_y, max_radius):
    BLOCK_X = 16
    BLOCK_Y = 16
    width = img_width
    height = img_height
    grid_x = int((width + BLOCK_X - 1) / BLOCK_X)
    grid_y = int((height + BLOCK_Y - 1) / BLOCK_Y)

    rect_min_x = min(grid_x, max(0, int((p_x - max_radius) / BLOCK_X)))
    rect_min_y = min(grid_y, max(0, int((p_y - max_radius) / BLOCK_Y)))

    rect_max_x = min(grid_x, max(0, int((p_x + max_radius + BLOCK_X - 1) / BLOCK_X)))
    rect_max_y = min(grid_y, max(0, int((p_y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
    return (rect_min_x * grid_x, rect_min_y * grid_y), (rect_max_x * grid_x, rect_max_y * grid_y)


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
xyz_homo = torch.cat([xyz, torch.ones_like(xyz[..., :1])], -1)
xyz_proj = (full_proj_transform.T @ xyz_homo.T).T
xyz_proj = xyz_proj / (xyz_proj[..., -1:] + 1e-7)
uv = torch.stack([
    ((xyz_proj[..., 0] + 1.0) * img_width - 1.0) * 0.5,
    ((xyz_proj[..., 1] + 1.0) * img_height - 1.0) * 0.5
], -1)

uv
# ret["radii"]
# %%

cov3d = scale_rot_to_cov3d(scales, quats)
prec = symmetric_to_utril(cov3d.inverse().detach())
radii = ret["radii"]
# uv 


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
    )

tx_precisions = prec.clone().detach()
tx_means = xyz.clone().detach()
tx_opacities = opacity.clone().detach()
tx_gt_depth = gt_depth.clone().detach()

tx_precisions.requires_grad_(True)
tx_means.requires_grad_(True)
tx_opacities.requires_grad_(True)
tx_gt_depth.requires_grad_(True)

inputs = (tx_means, tx_precisions, tx_opacities, gt_depth)
epss = (1e-3, 1e3, 1e-3, 1e-4)
with torch.no_grad():
    grads = []
    for idx, inp in enumerate(inputs):
        eps = epss[idx]
        perturb = torch.zeros_like(inp).view(-1)
        num_grad = torch.zeros_like(perturb).view(-1)
        for i in range(len(perturb)):
            perturb[i] = eps
            plus_input = [x if j!= idx else x + perturb.view(x.shape) for j,x in enumerate(inputs)]
            out_plus = test_fn(*plus_input).sum()
            minu_input = [x if j!= idx else x - perturb.view(x.shape) for j,x in enumerate(inputs)]
            out_minu = test_fn(*minu_input).sum()
            num_grad[i] = (out_plus - out_minu) / (2 * eps)
            perturb[i] = 0
        num_grad = num_grad.view(inp.shape)
        grads.append(num_grad)
        if idx == 2:
            break

nll = test_fn(*inputs)
nll.sum().backward()

print('Numerical')
print(tx_means.grad.cpu())
print(grads[0].cpu())
print(torch.allclose(tx_means.grad, grads[0], rtol=0.08, atol=1e-5))
abs_err = (tx_means.grad - grads[0]).abs()
rel_err = abs_err / tx_means.grad
print(abs_err.max(), rel_err.max())

print(tx_precisions.grad.cpu())
print(grads[1].cpu())
print(torch.allclose(tx_precisions.grad, grads[1], rtol=0.002, atol=1e-5))
abs_err = (tx_precisions.grad - grads[1]).abs()
rel_err = abs_err / tx_precisions.grad
print(abs_err.max(), rel_err.max())
# %%
o = tx_opacities.detach().clone()
p = tx_precisions.detach().clone()
X = tx_means.detach().clone()

X.requires_grad_(True)
o.requires_grad_(True)
p.requires_grad_(True)
print(X.grad)

P = torch.stack([
    p[..., 0], p[..., 1], p[..., 2],
    p[..., 1], p[..., 3], p[..., 4],
    p[..., 2], p[..., 4], p[..., 5]
], dim=-1).view(-1, 3, 3)

focal_x = img_width / (2 * math.tan(FoVx * 0.5))
focal_y = img_height / (2 * math.tan(FoVy * 0.5))

j,i = torch.meshgrid(torch.arange(img_width), torch.arange(img_height), indexing='ij')
Dx = (i - 0.5 * img_width) / focal_x
Dy = (j - 0.5 * img_height) / focal_y
Dz = torch.ones_like(Dx)
D = torch.stack([Dx, Dy, Dz], dim=-1).cuda()
Dmag = D.norm(dim=-1).clamp(min=1e-6)
D /= Dmag[..., None]
# %%
#%%
s = (gt_depth * Dmag)[..., None]
xs = (s * D).view(-1, 3, 1)
D = D.view(-1, 3, 1)
s = s.view(-1)
i = i.reshape(-1).cuda()
j = j.reshape(-1).cuda()

Ssum = torch.zeros(img_height, img_width, device="cuda")
Gsum = torch.zeros_like(Ssum)

Smax = torch.ones_like(Ssum) * -1e6
for Xi, pi, oi in zip(X, P, o):
    Xi = Xi.view(1, 3, 1)
    Pi = pi.view(1, 3, 3)
    oi = oi.view(1, 1)
    opac = oi.view(1)
    xs_m_mu = xs - Xi
    Si = -0.5 * (xs_m_mu.transpose(-1, -2) @ Pi @ xs_m_mu).squeeze(-1).squeeze(-1)
    mask = Si > 0.0
    Si = Si + opac.log()
    Si = torch.where(mask, -1e6, torch.exp(Si))
    Si = Si.view(img_height, img_width)
    Smax = torch.maximum(Smax, Si)
Smax = Smax.detach()
for Xi, pi, oi, ri, uvi in zip(X, P, o, radii, uv):
    Xi = Xi.view(1, 3, 1)
    Pi = pi.view(1, 3, 3)
    oi = oi.view(1, 1)
    opac = oi.view(1)
    
    ri = ri.view(1).item()
    if not (ri) > 0:
        continue

    u,v = uvi.view(2).unbind()
    u = u.item()
    v = v.item()
    (ix, iy), (sx, sy) = get_bounds(u, v, ri)
    mask = (i >= ix) & (i < sx) & (j >= iy) & (j < sy)
    mask = ~mask

    xs_m_mu = xs - Xi
    Si = -0.5 * (xs_m_mu.transpose(-1, -2) @ Pi @ xs_m_mu).squeeze(-1).squeeze(-1)
    Si = Si + opac.log() - Smax.view(-1)
 
    dprecd = (D.transpose(-1, -2) @ Pi @ D).squeeze(-1).squeeze(-1)
    dprecu = (D.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1)
    uprecu = (Xi.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1).expand(dprecu.shape)
    print(dprecd.shape, dprecu.shape, uprecu.shape)

    fac = 1/ (2 * dprecd).sqrt()
    alpha = math.sqrt(3.14159265359) * fac
    beta = dprecu**2 / 2 / dprecd - uprecu / 2
    gamma = dprecu * fac
    delta = dprecu * fac - s / math.sqrt(2) * (dprecd).sqrt()

    exp_beta = beta.exp()
    erf_diff = gamma.erf() - delta.erf()
    Gi = opac * alpha * exp_beta * erf_diff
    
    mask = mask | (dprecd <= 0.)
    mask = mask | (dprecu <= 0.)
    mask = mask | (uprecu <= 0.)
    mask = mask | (Si > 0.0)
    # mask = mask | (beta > 0.)
    mask = mask.view(img_height, img_width)


    Si = Si.view(img_height, img_width)
    Gi = Gi.view(img_height, img_width)
    Gsum += torch.where(mask, 0.0, Gi)
    Ssum += torch.where(mask, 0.0, Si.exp())

loss = Gsum -(Ssum.clamp(min=1e-42).log() + Smax)

loss.sum().backward()
tval = torch.allclose(nll, loss)
if tval:
    print("Loss ok")
else:
    plt.subplot(131)
    plt.imshow(nll.detach().cpu().numpy())
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(loss.detach().cpu().numpy())
    plt.colorbar()
    plt.subplot(133)
    plt.imshow((nll - loss).detach().cpu().numpy())
    plt.colorbar()
# %%
print('Torch')
print(tx_means.grad.cpu())
print(X.grad.cpu())
print(torch.allclose(tx_means.grad, X.grad, rtol=0.001, atol=1e-5))
# print(tx_means.grad - X.grad)
# %%
print(tx_opacities.grad.cpu())
print(o.grad.cpu())
print(torch.allclose(tx_opacities.grad, o.grad, rtol=0.001, atol=1e-5))
# %%
print(tx_precisions.grad.cpu())
print(p.grad.cpu())
print(torch.allclose(tx_precisions.grad, p.grad, rtol=0.001, atol=1e-5))
# %%
# print('Numerical')
# print(tx_means.grad.cpu())
# print(grads[0].cpu())
# print(torch.allclose(tx_means.grad, grads[0], rtol=0.001, atol=1e-5))
# # %%
# print(tx_precisions.grad.cpu())
# print(grads[1].cpu())
# print(torch.allclose(tx_precisions.grad, grads[1], rtol=0.002, atol=1e-5))
# # %%
# print(tx_opacities.grad.cpu())
# print(grads[2].cpu())
# print(torch.allclose(tx_opacities.grad, grads[2], rtol=0.001, atol=1e-5))
# %%