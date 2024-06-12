# %%
import torch
import math
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
def quat_to_rot(rot):
    r, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    R = torch.stack([
        torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)], dim=-1),
        torch.stack([2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)], dim=-1),
        torch.stack([2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)
    ], dim=-2)
    return R

def compute_sigma(rot, s):
    R = quat_to_rot(rot)    

    S = torch.diag_embed(s)

    M = torch.bmm(S, R)
    Sigma = torch.bmm(M.transpose(1, 2), M)

    return Sigma


def scale_rot_to_cov3d(scales, quats):
    # quats = quats / (quats.norm(dim=1, keepdim=True).clamp(min=1e-6))
    return compute_sigma(quats, scales)

def symmetric_to_utril(matrix):
    utril_indices = torch.triu_indices(matrix.shape[-2], matrix.shape[-1])
    utril_vector = matrix[..., utril_indices[0], utril_indices[1]]
    return utril_vector

device = torch.device("cuda:0")
N = 100
xyz = torch.rand(N,3, dtype=torch.float32, device=device).view(-1,3) * 2
# # xyz = torch.tensor([[0, 1, 0.5]], device=device).repeat(N, 1) * 2
xyz[..., 0] -= 1
xyz[..., 1] -= 1
xyz[..., 2] *= 4
xyz[..., 2] += 2

# # xyz[..., 0] *= 2
# # xyz[..., 1] *= 2
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

gt_depth = torch.ones(256, 256, device="cuda") * 6


world_quat = torch.rand(4, device="cuda")
world_quat = world_quat / world_quat.norm().clamp(min=1e-6)
R = quat_to_rot(world_quat[None])[0]
T = torch.rand(3, device="cuda") * 10

# Move gaussians to random positions in the world
xyz = (R.T @ (xyz - T[None, :]).T).T

# R= torch.tensor([[ 0.5577,  0.5937, -0.5801],
#         [-0.4142,  0.8047,  0.4253],
#         [ 0.7193,  0.0031,  0.6947]], device='cuda:0')
# T = torch.tensor([0.8987, 2.6797, 9.5107], device='cuda:0')
# xyz = torch.tensor([[ 0.0187, -2.8013, -1.1293],
#                     [ 0.2053, -2.8583, -0.3883]], device='cuda:0')
# scales = torch.tensor([[0.0975, 0.1916, 0.0314],
#                       [0.0705, 0.1179, 0.1962]], device='cuda:0')
# quats = torch.tensor([[0.5951, 0.4736, 0.0782, 0.6445],
#         [0.6286, 0.2202, 0.5916, 0.4544]], device='cuda:0')
# quats = quats / (quats.norm(dim=1, keepdim=True).clamp(min=1e-6))

# R= torch.tensor([[ 0.2002,  0.9543,  0.2218],
#         [-0.4614, -0.1079,  0.8806],
#         [ 0.8643, -0.2787,  0.4187]], device='cuda:0')
# T = torch.tensor([4.0134, 4.7615, 1.8147], device='cuda:0')
# xyz = torch.tensor([[ 2.5191, -3.1310, -4.9738]], device='cuda:0')
# scales = torch.tensor([[0.1913, 0.1965, 0.1801]], device='cuda:0')
# quats = torch.tensor([[0.2069, 0.4401, 0.8402, 0.2399]], device='cuda:0')
# quats = quats / (quats.norm(dim=1, keepdim=True).clamp(min=1e-6))


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

znear = 0.01
zfar = 100.0

FoVx = 2 * math.atan(img_width / (2 * fx))
FoVy = 2 * math.atan(img_height / (2 * fy))

world_view_transform = torch.tensor(getWorld2View2(R.T.cpu().numpy(), T.cpu().numpy(), torch.tensor([0, 0, 0]).numpy(), 1.0)).transpose(0, 1).cuda()
projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
world_view_transform
#%% 

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

def check(grad1, grad2, rtol=0.001, atol=1e-5):
    ok = torch.allclose(grad1, grad2, rtol=rtol, atol=atol)
    abs_err = torch.abs(grad1 - grad2)
    rel_err = abs_err / (torch.abs(grad1) + 1e-6)
    print(f'avg abs err {abs_err.mean().item():.1e}')
    print(f'avg rel err {rel_err.mean().item() * 100:.1} %')
    if not ok and grad1.shape[0] < 8:
        print()
        print(grad1.cpu())
        print(grad2.cpu())
        print(abs_err.cpu())
    return ok


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
from diff_gaussian_rasterization_depth_acc_mod import depth

def test_fn(
        tx_xyz,
        tx_scales,
        tx_rots,
        opacities,
        gt_depth,
        prec=None):
    
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=img_height,
        image_width=img_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=tx_xyz.new_zeros(3),
        scale_modifier=1,
        viewmatrix=world_view_transform.contiguous(),
        projmatrix=full_proj_transform.contiguous(),
        sh_degree=1,
        campos=tx_xyz.new_zeros(3),
        prefiltered=False,
        debug=False
    )
    if prec is None:
        prec = torch.zeros(tx_xyz.shape[0], 6, device="cuda").contiguous()
    return depth._RasterizeDepthGaussiansDebug.apply(
        tx_xyz.contiguous(),
        tx_scales.contiguous(),
        tx_rots.contiguous(),
        opacities.contiguous(),
        gt_depth.contiguous(),
        raster_settings,
        prec.contiguous()
    )[0]

tx_means = xyz.clone().detach()
tx_scales = scales.clone().detach()
tx_rots = quats.clone().detach()
tx_opacities = opacity.clone().detach()
tx_gt_depth = gt_depth.clone().detach()

tx_scales.requires_grad_(True)
tx_rots.requires_grad_(True)
tx_means.requires_grad_(True)
tx_opacities.requires_grad_(True)
tx_gt_depth.requires_grad_(True)
# %%
xyz_homo = torch.cat([xyz, torch.ones_like(xyz[..., :1])], -1)
xyz_proj = (full_proj_transform.T @ xyz_homo.T).T
xyz_proj = xyz_proj / (xyz_proj[..., -1:] + 1e-7)
uv = torch.stack([
    ((xyz_proj[..., 0] + 1.0) * img_width - 1.0) * 0.5,
    ((xyz_proj[..., 1] + 1.0) * img_height - 1.0) * 0.5
], -1)
radii = ret["radii"]

o = tx_opacities.detach().clone()
X = tx_means.detach().clone()
Sc = tx_scales.detach().clone()
Ro = tx_rots.detach().clone()
Rad = radii.detach().clone()
uv = uv.detach().clone()

X.requires_grad_(True)
o.requires_grad_(True)
Ro.requires_grad_(True)
Sc.requires_grad_(True)

Xt = (R @ X.T).T + T[None]

cov3d = scale_rot_to_cov3d(Sc, Ro)
cov3d = cov3d + torch.eye(3)[None].cuda() * 0.1
cov3d = R[None] @ cov3d @ R.T[None]
p = symmetric_to_utril(cov3d.inverse())
p.retain_grad()
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

s = (gt_depth * Dmag)[..., None]
xs = (s * D).view(-1, 3, 1)
D = D.view(-1, 3, 1)
s = s.view(-1)
i = i.reshape(-1).cuda()
j = j.reshape(-1).cuda()

Ssum = torch.zeros(img_height, img_width, device="cuda")
Gsum = torch.zeros_like(Ssum)

Smax = torch.ones_like(Ssum) * -1.175494351e-38

reind = torch.argsort(Xt[..., 2])

Smax = torch.ones_like(Ssum) * -1.175494351e-38
with torch.no_grad():
    for Xi, pi, oi, ri, uvi in zip(Xt[reind], P[reind], o[reind], radii[reind], uv[reind]):
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
        S = -0.5 * (xs_m_mu.transpose(-1, -2) @ Pi @ xs_m_mu).squeeze(-1).squeeze(-1)
    
        dprecd = (D.transpose(-1, -2) @ Pi @ D).squeeze(-1).squeeze(-1)
        dprecu = (D.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1)
        uprecu = (Xi.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1).expand(dprecu.shape)
        # print(dprecd.shape, dprecu.shape, uprecu.shape)

        fac = 1/ (2 * dprecd).sqrt()
        alpha = math.sqrt(3.14159265359) * fac
        beta = dprecu**2 / 2 / dprecd - uprecu / 2
        gamma = dprecu * fac
        delta = dprecu * fac - s / math.sqrt(2) * (dprecd).sqrt()
        
        mask = mask | (dprecd <= 0.)
        mask = mask | (dprecu <= 0.)
        mask = mask | (uprecu <= 0.)
        mask = mask | (S >= 0.0)
        mask = mask | (beta > 80.)

        Si = S + opac.log()
        Si = torch.where(mask, -1.175494351e-38, torch.exp(Si))
        Si = Si.view(img_height, img_width)
        Smax = torch.maximum(Smax, Si)


Smax = Smax.detach()
for Xi, pi, oi, ri, uvi in zip(Xt[reind], P[reind], o[reind], radii[reind], uv[reind]):
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
    S = -0.5 * (xs_m_mu.transpose(-1, -2) @ Pi @ xs_m_mu).squeeze(-1).squeeze(-1)
    Si = S + opac.log() - Smax.view(-1)
 
    dprecd = (D.transpose(-1, -2) @ Pi @ D).squeeze(-1).squeeze(-1)
    dprecu = (D.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1)
    uprecu = (Xi.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1).expand(dprecu.shape)
    # print(dprecd.shape, dprecu.shape, uprecu.shape)

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
    mask = mask | (S >= 0.0)
    mask = mask | (beta > 87.)
    mask = mask.view(img_height, img_width)
    Si = Si.view(img_height, img_width)
    Gi = Gi.view(img_height, img_width)

    Gadd = torch.where(mask, 0.0, Gi)
    Sadd = torch.where(mask, 0.0, Si.exp())

    if not torch.isfinite(Gadd).all():
        print("Gi not finite")
        break
    if not torch.isfinite(Sadd).all():
        print("Si not finite")
        break


    Gsum += torch.where(mask, 0.0, Gi)
    Ssum += torch.where(mask, 0.0, Si.exp())

loss = Gsum  -(Ssum.clamp(min=1e-8).log() + Smax)

loss.sum().backward()
tx_prec = torch.zeros(tx_means.shape[0], 6, device="cuda", requires_grad=True)
for t in [tx_means, tx_scales, tx_rots, tx_opacities]:
    t.grad = None
nll = test_fn(tx_means, tx_scales, tx_rots, tx_opacities, gt_depth,tx_prec)
nll.sum().backward()
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
    print((nll - loss).abs().max())
# %%
print('Torch')
print('Torch')
print('means ok=', check(tx_means.grad, X.grad))
print('scales ok=', check(tx_scales.grad, Sc.grad))
print('rots ok=', check(tx_rots.grad, Ro.grad))
# print('opacities ok=', check(tx_opacities.grad, o.grad))

# %%
import numpy as np
epses = np.array([5.e-5, 1.e-04, 5.e-4, 1.e-03, 5.e-03, 1.e-02])
# epses = np.array([0.1])
tx_means = xyz.clone().detach()
tx_scales = scales.clone().detach()
tx_rots = quats.clone().detach()
tx_opacities = opacity.clone().detach()
tx_gt_depth = gt_depth.clone().detach()

tx_scales.requires_grad_(True)
tx_rots.requires_grad_(True)
tx_means.requires_grad_(True)
tx_opacities.requires_grad_(True)
tx_gt_depth.requires_grad_(True)

inputs = (tx_means, tx_scales, tx_rots, tx_opacities, gt_depth)
# epss = (6e-3, 5e-3, 1e-3, 1e-4)
# epss = (1e-3, 5e-4, 1e-3, 1e-4)
pos = 2
nll = test_fn(*inputs)
nll.sum().backward()
abs_errors = []
rel_errors = []
with torch.no_grad():
    for eps in epses:
        inp = inputs[pos]
        grads = []
        perturb = torch.zeros_like(inp).reshape(-1)
        num_grad = torch.zeros_like(perturb).reshape(-1)
        for i in range(len(perturb)):
            perturb[i] = eps
            plus_input = [x if j!= pos else x + perturb.view(x.shape) for j,x in enumerate(inputs)]
            out_plus = test_fn(*plus_input).sum()
            minu_input = [x if j!= pos else x - perturb.view(x.shape) for j,x in enumerate(inputs)]
            out_minu = test_fn(*minu_input).sum()
            num_grad[i] = (out_plus - out_minu) / (2 * eps)
            perturb[i] = 0
        num_grad = num_grad.view(inp.shape)
        abs_error = (inp.grad - num_grad).abs()
        rel_error = abs_error / (inp.grad.abs() + 1e-6)
        abs_errors.append(abs_error.mean().item())
        rel_errors.append(rel_error.mean().item() * 100)
# print(abs_errors, rel_errors)
# check(num_grad, X.grad)
# %%
fig, ax1 = plt.subplots()

x = epses
y1 = abs_errors
y2 = rel_errors

def get_lims(data, factor=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return lower_bound, upper_bound

ax1.set_xscale('log')
ax1.plot(x, y1, 'b-x')
ax1.set_xlabel('$\epsilon$ (log scale)')
ax1.set_ylabel(f'Absolute Error (min={np.min(y1):.1e})', color='b')
y1_lower, y1_upper = get_lims(y1)
ax1.set_ylim(0, y1_upper)

ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-o')
ax2.set_ylabel(f'Relative Error (%) (min={np.min(y2):.1e})', color='r')
plt.title("Gradient errors against numerical estimate for Rotations")
y2_lower, y2_upper = get_lims(y2)
ax2.set_ylim(0, 100)

# %%
# %%
X
# %%
tx_means
# %%
