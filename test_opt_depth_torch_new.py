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
    quats = quats / (quats.norm(dim=1, keepdim=True).clamp(min=1e-6))
    return compute_sigma(quats, scales)

def symmetric_to_utril(matrix):
    utril_indices = torch.triu_indices(matrix.shape[-2], matrix.shape[-1])
    utril_vector = matrix[..., utril_indices[0], utril_indices[1]]
    return utril_vector

device = torch.device("cuda:0")
xyz = torch.tensor([
    # [-1, -1, 5],
    # [1, -1, 5],
    # [-1, 1, 5],
    # [1, 1, 5],
    [0,0,10], 
], dtype=torch.float32, device=device).view(-1,3) * 0.5
scales = torch.tensor([
    # [1, 1, 1], 
    # [1, 1, 1], 
    # [1, 1, 1], 
    # [1, 1, 1],
    [2, 2, 2], 
], dtype=torch.float32, device=device).view(-1,3) * 0.25
quats = torch.tensor([
    # [1,0,0,0], 
    # [1,0,0,0], 
    # [1,0,0,0], 
    # [1,0,0,0], 
    [1,0,0,0]
], dtype=torch.float32, device=device).view(-1,4)

colors = torch.tensor([
    # [1, 0, 0], 
    # [0, 1, 0],
    # [0, 0, 1], 
    # [1, 1, 0],
    [1, 1, 1],
    ], dtype=torch.float32, device=device).view(-1,3)
opacity = torch.tensor([
    # 1, 
    # 1, 
    # 1, 
    # 1, 
    0.99
     ], dtype=torch.float32, device=device).view(-1,1)


N = 100
# xyz = torch.rand(N,3, dtype=torch.float32, device=device).view(-1,3) * 2
# # xyz = torch.tensor([[0, 1, 0.5]], device=device).repeat(N, 1) * 2
# xyz[..., 0] -= 1
# xyz[..., 1] -= 1
# xyz[..., 2] *= 4
# xyz[..., 2] += 2

# # xyz[..., 0] *= 2
# # xyz[..., 1] *= 2
# # Since extrinsics is identity 
# # depth sort 
depth = xyz[..., 2]
ind = torch.argsort(depth, descending=False)
xyz = xyz[ind]

# scales = torch.rand(N, 3, device=device).view(-1,3) * 0.2
# quats = torch.rand(N, 4, device=device).view(-1,4)
# quats /= quats.norm(dim=1, keepdim=True).clamp(min=1e-6)

# xyz = torch.tensor([[ 0.5607, -0.7803,  6.0269]], device='cuda:0', requires_grad=True)
# scales = torch.tensor([[1.5013, 0.7892, 0.5560]], device='cuda:0', requires_grad=True)
# quats = torch.tensor([[0.4965, 0.4484, 0.6553, 0.3507]], device='cuda:0', requires_grad=True)

# xyz = torch.tensor([[ 0.8954, 0.9857, 1.7505]], device='cuda:0', requires_grad=True)
# scales = torch.tensor([[0.2153, 1.3674, 1.9875]], device='cuda:0', requires_grad=True)
# quats = torch.tensor([[0.3264, 0.7270, 0.6041, 0.0047]], device='cuda:0', requires_grad=True)

# colors = torch.rand(N, 3, device=device).view(-1,3)
# opacity = torch.rand(N, 1, device=device).view(-1,1).clamp(min=0.99)


gt_depth = torch.ones(256, 256, device="cuda") * 6
T = torch.zeros(3, device="cuda")
R = torch.eye(3, device='cuda')

# gt_depth.requires_grad_(True)
# xyz.requires_grad_(True)
# scales.requires_grad_(True)
# quats.requires_grad_(True)
# colors.requires_grad_(True)
# opacity.requires_grad_(True)

xyz = torch.nn.Parameter(xyz, requires_grad=True)
scales = torch.nn.Parameter(scales, requires_grad=True)
quats = torch.nn.Parameter(quats, requires_grad=True)
opacity = torch.nn.Parameter(opacity, requires_grad=True)

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

# %%
import math
from diff_gaussian_rasterization_depth_acc_mod import GaussianRasterizationSettings, GaussianRasterizer
def render(
        means3D,
        opacity,
        scales, 
        rotations,
        colors_precomp,
        full_proj_transform,
        world_view_transform
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
    colors_precomp=colors,
    full_proj_transform=full_proj_transform,
    world_view_transform=world_view_transform
)

from matplotlib import pyplot as plt
img = ret["render"].detach().cpu().numpy().transpose(1, 2, 0)
alp = ret["acc"].detach().cpu().numpy()
dis = (ret['depth']).detach().cpu().numpy()
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


def create_ellipsoid(covariance_tensor, mean_tensor, num_points=100):
    # Ensure tensors are float type for eigen decomposition
    covariance_tensor = covariance_tensor.float()
    
    # Compute eigenvalues and eigenvectors using PyTorch
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_tensor)

    # Generate points on a unit sphere
    u = torch.linspace(0, 2 * torch.pi, num_points)
    v = torch.linspace(0, torch.pi, num_points)
    x = torch.outer(torch.cos(u), torch.sin(v))
    y = torch.outer(torch.sin(u), torch.sin(v))
    z = torch.outer(torch.ones(num_points), torch.cos(v))
    
    # Scale points by the square roots of the eigenvalues
    x = x * torch.sqrt(eigenvalues[0])
    y = y * torch.sqrt(eigenvalues[1])
    z = z * torch.sqrt(eigenvalues[2])

    # Rotate points and add mean
    xyz = torch.stack([x, y, z]).reshape(3, -1)
    xyz = torch.mm(eigenvectors, xyz).reshape(3, num_points, num_points)
    x, y, z = xyz[0] + mean_tensor[0], xyz[1] + mean_tensor[1], xyz[2] + mean_tensor[2]

    # Convert to NumPy arrays for Plotly
    return x.numpy(), y.numpy(), z.numpy()

def render_RT(
    means3D,
    opacity,
    scales,
    rotations,
    colors_precomp,
    R,
    T
):
    world_view_transform = torch.tensor(getWorld2View2(R.T.cpu().numpy(), T.cpu().numpy(), torch.tensor([0, 0, 0]).numpy(), 1.0)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    return render(
        means3D=means3D,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        colors_precomp=colors_precomp,
        full_proj_transform=full_proj_transform,
        world_view_transform=world_view_transform
    )

import numpy as np
def look_at_matrix(X, C):
    # Calculate the forward vector
    F = X - C
    F = F / np.linalg.norm(F)

    # Define the up vector
    Up = np.array([0, -1, 0])

    # Calculate the right vector
    R = np.cross(Up, F)
    R = R / np.linalg.norm(R)

    # Recalculate the true up vector
    U = np.cross(F, R)

    # Construct the rotation matrix
    R_matrix = np.array([R, U, F])

    return R_matrix


# theta_deg = 15  # look down 
# theta_rad = torch.tensor(theta_deg, dtype=torch.float32) * torch.pi / 180

# # y axis is pointing down
# R = torch.tensor([
#     [1, 0, 0],
#     [0, torch.cos(theta_rad), -torch.sin(theta_rad)],
#     [0, torch.sin(theta_rad), torch.cos(theta_rad)]
# ], dtype=torch.float32).cuda()

target = np.array([0, 0, gt_depth.mean().item()-1])
def camera_trajecotry(target, num_frames, height_variation=1.0, distance=2.0):
    angles = np.linspace(0, np.pi, num_frames).reshape(-1, 1)
    heights = np.sin(np.linspace(0, 6 * np.pi, num_frames)).reshape(-1, 1) * height_variation
    target = target.reshape(1, 3)

    C = target + np.hstack([np.cos(angles)*distance, heights, -np.sin(angles)*distance])
    return C
cam_pos = camera_trajecotry(target, 60, height_variation=0.0, distance=10.0)
print(cam_pos.shape)

@torch.no_grad()
def render_pt(pos, target, means3D, opacity, scales, rotations, colors):
    R = torch.tensor(look_at_matrix(target, pos)).float().cuda()
    C = torch.tensor(pos).float().cuda()
    T = -R.T @ C
    ret = render_RT(
        means3D=means3D,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        colors_precomp=colors,
        R=R,
        T=T
    )
    return ret["render"].detach().cpu().numpy().transpose(1, 2, 0)


import mediapy as media
frames = []
for pos in cam_pos:
    img = render_pt(pos, target, xyz, opacity, scales, quats, colors)
    frames.append(img)
media.show_video(frames, fps=10)
# %%

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
    if not ok:
        print()
        print(grad1.cpu())
        print(grad2.cpu())
        print(abs_err.cpu())
    return ok


focal_x = img_width / (2 * math.tan(FoVx * 0.5))
focal_y = img_height / (2 * math.tan(FoVy * 0.5))

def depth_loss_old(
        means3D,
        opacity,
        scales, 
        rotations,
        depth,
        R,
        T):
    Xt = (R @ means3D.T).T + T[None]
    Xt = Xt
    reind = torch.argsort(Xt[..., 2])
    # reind = torch.arange(Xt.shape[0])

    cov3d = scale_rot_to_cov3d(scales, rotations)
    cov3d = cov3d + torch.eye(3)[None].cuda() * 0.1
    cov3d = R[None] @ cov3d @ R.T[None]
    p = symmetric_to_utril(cov3d.inverse())

    P = torch.stack([
        p[..., 0], p[..., 1], p[..., 2],
        p[..., 1], p[..., 3], p[..., 4],
        p[..., 2], p[..., 4], p[..., 5]
    ], dim=-1).view(-1, 3, 3)

    j,i = torch.meshgrid(torch.arange(img_width), torch.arange(img_height), indexing='ij')
    Dx = (i - 0.5 * img_width) / focal_x
    Dy = (j - 0.5 * img_height) / focal_y
    Dz = torch.ones_like(Dx)
    D = torch.stack([Dx, Dy, Dz], dim=-1).cuda()
    Dmag = D.norm(dim=-1).clamp(min=1e-6)
    D /= Dmag[..., None]
    s = (depth * Dmag)[..., None]
    xs = (s * D).view(-1, 3, 1)
    D = D.view(-1, 3, 1)
    s = s.view(-1)
    i = i.reshape(-1).cuda()
    j = j.reshape(-1).cuda()

    with torch.no_grad():
        world_view_transform = torch.tensor(getWorld2View2(R.T.cpu().numpy(), T.cpu().numpy(), torch.tensor([0, 0, 0]).numpy(), 1.0)).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        ret = render(
            means3D=means3D,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            colors_precomp=torch.ones_like(means3D),
            full_proj_transform=full_proj_transform,
            world_view_transform=world_view_transform
        )
        radii = ret["radii"]

        xyz_homo = torch.cat([means3D, torch.ones_like(xyz[..., :1])], -1)
        xyz_proj = (full_proj_transform.T @ xyz_homo.T).T
        xyz_proj = xyz_proj / (xyz_proj[..., -1:] + 1e-7)
        uv = torch.stack([
            ((xyz_proj[..., 0] + 1.0) * img_width - 1.0) * 0.5,
            ((xyz_proj[..., 1] + 1.0) * img_height - 1.0) * 0.5
        ], -1)

    Ssum = torch.zeros(img_height, img_width, device="cuda")
    Gsum = torch.zeros_like(Ssum)

    Smax = torch.ones_like(Ssum) * -1.175494351e-38
    Smax = torch.zeros_like(Smax)
    # with torch.no_grad():
    #     for Xi, pi, oi, ri, uvi in zip(Xt[reind], P[reind], o[reind], radii[reind], uv[reind]):
    #         Xi = Xi.view(1, 3, 1)
    #         Pi = pi.view(1, 3, 3)
    #         oi = oi.view(1, 1)
    #         opac = oi.view(1)
            
    #         ri = ri.view(1).item()
    #         if not (ri) > 0:
    #             continue
    #         u,v = uvi.view(2).unbind()
    #         u = u.item()
    #         v = v.item()
    #         (ix, iy), (sx, sy) = get_bounds(u, v, ri)
    #         mask = (i >= ix) & (i < sx) & (j >= iy) & (j < sy)
    #         mask = ~mask
    #         mask = torch.zeros_like(mask)

    #         xs_m_mu = xs - Xi
    #         S = -0.5 * (xs_m_mu.transpose(-1, -2) @ Pi @ xs_m_mu).squeeze(-1).squeeze(-1)
        
    #         dprecd = (D.transpose(-1, -2) @ Pi @ D).squeeze(-1).squeeze(-1)
    #         dprecu = (D.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1)
    #         uprecu = (Xi.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1).expand(dprecu.shape)
    #         # print(dprecd.shape, dprecu.shape, uprecu.shape)

    #         fac = 1/ (2 * dprecd).sqrt()
    #         alpha = math.sqrt(3.14159265359) * fac
    #         beta = dprecu**2 / 2 / dprecd - uprecu / 2
    #         gamma = dprecu * fac
    #         delta = dprecu * fac - s / math.sqrt(2) * (dprecd).sqrt()
            
    #         mask = mask | (dprecd <= 0.)
    #         mask = mask | (dprecu <= 0.)
    #         mask = mask | (uprecu <= 0.)
    #         mask = mask | (S >= 0.0)
    #         mask = mask | (beta > 80.)
    #         # mask = torch.zeros_like(mask)

    #         Si = S + opac.log()
    #         Si = torch.where(mask, -1.175494351e-38, torch.exp(Si))
    #         Si = Si.view(img_height, img_width)
    #         Smax = torch.maximum(Smax, Si)


    Smax = Smax.detach()
    for Xi, pi, oi, ri, uvi in zip(Xt[reind], P[reind], opacity[reind], radii[reind], uv[reind]):
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
        mask = torch.zeros_like(mask)

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
        # mask = mask | (S >= 0.0)
        mask = mask | (beta > 80.)
        mask = mask.view(img_height, img_width)
        mask = torch.zeros_like(mask)
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

    loss_spatial= Gsum  -(Ssum.clamp(min=1e-37).log() + Smax)
    # loss_spatial = Gsum  -(Ssum.log() + Smax)
    # loss = loss_spatial.sum()
    return loss_spatial

def depth_loss(
        means3D,
        opacity,
        scales, 
        rotations,
        depth,
        R,
        T):
    Xt = (R @ means3D.T).T + T[None]
    Xt = Xt
    reind = torch.argsort(Xt[..., 2])
    # reind = torch.arange(Xt.shape[0])

    cov3d = scale_rot_to_cov3d(scales, rotations)
    cov3d = cov3d + torch.eye(3)[None].cuda() * 0.1
    cov3d = R[None] @ cov3d @ R.T[None]
    p = symmetric_to_utril(cov3d.inverse())

    P = torch.stack([
        p[..., 0], p[..., 1], p[..., 2],
        p[..., 1], p[..., 3], p[..., 4],
        p[..., 2], p[..., 4], p[..., 5]
    ], dim=-1).view(-1, 3, 3)

    j,i = torch.meshgrid(torch.arange(img_width), torch.arange(img_height), indexing='ij')
    Dx = (i - 0.5 * img_width) / focal_x
    Dy = (j - 0.5 * img_height) / focal_y
    Dz = torch.ones_like(Dx)
    D = torch.stack([Dx, Dy, Dz], dim=-1).cuda()
    Dmag = D.norm(dim=-1).clamp(min=1e-6)
    D /= Dmag[..., None]
    s = (depth * Dmag)[..., None]
    xs = (s * D).view(-1, 3, 1)
    D = D.view(-1, 3, 1)
    s = s.view(-1)
    i = i.reshape(-1).cuda()
    j = j.reshape(-1).cuda()

    with torch.no_grad():
        world_view_transform = torch.tensor(getWorld2View2(R.T.cpu().numpy(), T.cpu().numpy(), torch.tensor([0, 0, 0]).numpy(), 1.0)).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        ret = render(
            means3D=means3D,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            colors_precomp=torch.ones_like(means3D),
            full_proj_transform=full_proj_transform,
            world_view_transform=world_view_transform
        )
        radii = ret["radii"]

        xyz_homo = torch.cat([means3D, torch.ones_like(xyz[..., :1])], -1)
        xyz_proj = (full_proj_transform.T @ xyz_homo.T).T
        xyz_proj = xyz_proj / (xyz_proj[..., -1:] + 1e-7)
        uv = torch.stack([
            ((xyz_proj[..., 0] + 1.0) * img_width - 1.0) * 0.5,
            ((xyz_proj[..., 1] + 1.0) * img_height - 1.0) * 0.5
        ], -1)

    Et = D.new_zeros(img_height, img_width)
    Etsq = D.new_zeros(img_height, img_width)
    prod_alpha = D.new_ones(img_height, img_width)
    for Xi, pi, oi, ri, uvi in zip(Xt[reind], P[reind], opacity[reind], radii[reind], uv[reind]):
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
        mask = torch.zeros_like(mask)

    
        dprecd = (D.transpose(-1, -2) @ Pi @ D).squeeze(-1).squeeze(-1)
        dprecu = (D.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1)
        uprecu = (Xi.transpose(-1, -2) @ Pi @ Xi).squeeze(-1).squeeze(-1).expand(dprecu.shape)
        # print(dprecd.shape, dprecu.shape, uprecu.shape)

        # print(f"{logpti.shape=}")

        


        fac = 1/ (2 * dprecd).sqrt()
        alpha = math.sqrt(3.14159265359) * fac
        beta = dprecu**2 / 2 / dprecd - uprecu / 2
        gamma = dprecu * fac
        # delta = dprecu * fac - s / math.sqrt(2) * (dprecd).sqrt()

        exp_beta = beta.exp()
        # erf_diff = gamma.erf() - delta.erf()

        Zi = opac * alpha * exp_beta
        bi = gamma
        ai = (dprecd / 2).sqrt()

        xsq = dprecd / 2
        zsq = uprecu / 2
        ysq = dprecu / 2 * dprecu / 2 / xsq
        mult = (-Zi * bi.erf() + ysq - zsq).exp() * math.sqrt(3.14159265359) / xsq / xsq.sqrt() * prod_alpha.view(-1) * opac
        # print(f"{mult.shape=}")
        # return mult
        Et_i = mult * dprecu / 2
        Etsq_i = mult * (2*ysq + 1) / 2 

        
        Pi_D = Pi - (Pi @ D @ D.mT @ Pi / (D.mT @ Pi @ D))
        # print(f"{Pi_D.shape=} {D.shape=} {Xi.shape=}")
        exp_term = (- 0.5 * Xi.mT @ Pi_D @ Xi).exp().squeeze(-1).squeeze(-1)
        # print(f"{exp_term.shape=}")
        log_alpha = -(2*3.14 / dprecd).sqrt() * opac * exp_term
        # print(f"{log_alpha.shape=}")
        prod_alpha = prod_alpha * log_alpha.view(img_height, img_width).exp()

        mask = mask | (dprecd <= 0.)
        mask = mask | (dprecu <= 0.)
        mask = mask | (uprecu <= 0.)
        # mask = mask | (S >= 0.0)
        # mask = mask | (beta > 80.)
        mask = mask.view(img_height, img_width)
        mask = torch.zeros_like(mask)

        Et += Et_i.view(img_height, img_width)
        Etsq += Etsq_i.view(img_height, img_width)

    sgt = s.view(img_height, img_width)
    loss_spatial = Etsq - 2*sgt*Et + sgt**2
    return loss_spatial


# %%
from matplotlib import pyplot as plt
with torch.no_grad():
    T = torch.zeros(3, device="cuda")
    R = torch.eye(3, device='cuda')
    loss = depth_loss(xyz, opacity, scales, quats, gt_depth, R, T)
plt.imshow(loss.view(img_height, img_width).detach().cpu().numpy())
plt.colorbar()
# %%
# Etsq, Et = others
# plt.imshow(Et.detach().cpu().numpy())
# plt.colorbar()


# %%
from IPython.display import display, clear_output
import torch.nn as nn
import io
from PIL import Image
from datetime import datetime
mu = nn.Parameter(xyz, requires_grad=True)
S = nn.Parameter(scales, requires_grad=True)
q = nn.Parameter(quats, requires_grad=True)
o = nn.Parameter(opacity, requires_grad=True)
lr_mult = 0.1
# opt = torch.optim.Adam([
#     {"params": mu, 'lr': 0.00016*lr_mult, 'name': 'xyz'},
#     {"params": S, 'lr': 0.005*lr_mult, 'name': 'scales'},
#     {"params": q, 'lr': 0.001*lr_mult, 'name': 'quats'},
#     {"params": o, 'lr': 0.05*lr_mult, 'name': 'opacity'}
# ], eps=1e-15)
opt = torch.optim.Adam([mu, S, q, o], lr=0.0001)

spatial_lr_scale = 1.0
min_iters = 1500
position_lr_init = 0.00016*lr_mult
position_lr_final = 0.0000016*lr_mult
position_lr_delay_mult = 0.01
position_lr_max_steps = 30_000
from utils.general_utils import get_expon_lr_func
xyz_scheduler_args = get_expon_lr_func(lr_init=position_lr_init*spatial_lr_scale,
                                        lr_final=position_lr_final*spatial_lr_scale,
                                        lr_delay_mult=position_lr_delay_mult,
                                        max_steps=position_lr_max_steps)


Tw = torch.zeros(3, device="cuda")
Rw = torch.eye(3, device='cuda')
frames = []
losses = []
itr_times = None
for i in range(10000):
    itr_start = datetime.now()
    opt.zero_grad()
    # for param_group in opt.param_groups:
    #     if param_group["name"] == "xyz":
    #         lr = xyz_scheduler_args(i)
    #         param_group['lr'] = lr
    opa = o.sigmoid()
    nll = depth_loss(
        mu,
        opa,
        S,
        q / q.norm(dim=1, keepdim=True).clamp(min=1e-6),
        gt_depth,
        Rw,
        Tw
    )
    loss = nll.sum()
    loss.backward()
    losses.append(loss.item())
    opt.step()
    itr_end = datetime.now()
    itr_time = (itr_end - itr_start).total_seconds()
    if itr_times is None:
        itr_times = itr_time
    else:
        itr_times = 0.9 * itr_times + 0.1 * itr_time
    if i % 50 == 0:
        with torch.no_grad():
            clear_output(wait=True)
            fig, ax = plt.subplots(2, 3, figsize=(15, 10))
            ax[0,0].plot(losses)
            ret = render_RT(
                means3D=mu,
                opacity=opa,
                scales=S,
                rotations=q,
                colors_precomp=colors,
                R=Rw,
                T=Tw
            )
            img = ret["render"].detach().cpu().numpy().transpose(1, 2, 0)
            ax[0, 1].imshow(img)
            ax[0, 1].axis('off')
            ax[0, 2].imshow(nll.detach().cpu().numpy())
            ax[0, 2].axis('off')
            img0 = render_pt(
                cam_pos[10],
                target,
                mu,
                opa,
                S,
                q,
                colors
            )
            ax[1,0].imshow(img0)
            ax[1,0].axis('off')
            img1 = render_pt(
                cam_pos[len(cam_pos) // 2],
                target,
                mu,
                opa,
                S,
                q,
                colors
            )
            ax[1,0].set_title(f"Step {i}")
            ax[1,1].imshow(img1)
            ax[1,1].axis('off')
            ax[1,1].set_title(f"depth at {gt_depth.mean().item()}")
            img2 = render_pt(
                cam_pos[-10],
                target,
                mu,
                opa,
                S,
                q,
                colors
            )
            ax[1,2].imshow(img2)
            ax[1,2].axis('off')
            ax[1,2].set_title(f"Loss {loss.item():.2e}")
            fig.tight_layout()
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png')
                buf.seek(0)
                frame = Image.open(buf).convert('RGB')
            plt.close(fig)
            display(frame)
            frames.append(frame)
        print(f"Step {i} Loss {loss.item():.2e} Time {itr_times:.2e}")
# %%
media.show_video([np.array(f) for f in frames], fps=3)
# %%
media.save_video("depth_opt_windows.mp4", [np.array(f) for f in frames], fps=10)
# %%
