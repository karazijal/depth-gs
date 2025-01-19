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
    [0,0,5], 
], dtype=torch.float32, device=device).view(-1,3) * 0.5
scales = torch.tensor([
    # [1, 1, 1], 
    # [1, 1, 1], 
    # [1, 1, 1], 
    [1, 1, 1],
    # [10, 10, 0.2], 
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

model_params, first_param = torch.load('output/lego-bsl-k5-nll-norgb-v2_lego_200/chkpnt10000.pth')
(_, 
xyz, 
__, 
___,
scales, 
quats, 
opacity,
*____) = model_params

op = opacity.sigmoid().detach().cpu().flatten().numpy()
print(f"culling to {(op > 0.1).sum()}")
cull_mask = op > 0.1
xyz = xyz[cull_mask]
scales = scales[cull_mask]
quats = quats[cull_mask]
opacity = opacity[cull_mask]



colors = torch.ones_like(xyz)

print(xyz.shape)
print(scales.shape)
print(quats.shape)
print(opacity.shape)
print(colors.shape)

gt_depth = torch.ones(256, 256, device="cuda") * 6
T = torch.zeros(3, device="cuda")
R = torch.eye(3, device='cuda')

# world_quat = torch.rand(4, device="cuda")
# world_quat = world_quat / world_quat.norm().clamp(min=1e-6)
# R = quat_to_rot(world_quat[None])[0]
# T = torch.rand(3, device="cuda") * 10
# R = torch.eye(3, device='cuda')

# [ 9.5106e-01,  1.2896e-01, -2.8082e-01,  0.0000e+00],
# [ 3.0902e-01, -3.9689e-01,  8.6428e-01,  0.0000e+00],
# [ 7.3514e-09, -9.0876e-01, -4.1731e-01,  0.0000e+00],
# [ 7.0415e-09, -1.4225e-08,  4.0311e+00,  1.0000e+00]


R = torch.tensor([
    [ 9.5106e-01,  1.2896e-01, -2.8082e-01],
    [ 3.0902e-01, -3.9689e-01,  8.6428e-01],
    [ 7.3514e-09, -9.0876e-01, -4.1731e-01],
], device="cuda").T
T = torch.tensor([7.0415e-09, -1.4225e-08,  4.0311e+00], device="cuda")

# theta_deg = 45  # look down 
# theta_rad = torch.tensor(theta_deg, dtype=torch.float32) * torch.pi / 180

# # y axis is pointing down
# R = torch.tensor([
#     [1, 0, 0],
#     [0, torch.cos(theta_rad), -torch.sin(theta_rad)],
#     [0, torch.sin(theta_rad), torch.cos(theta_rad)]
# ], dtype=torch.float32).cuda()

# C = torch.tensor([0, -2, 0]).float().cuda() # go up by one
# T = -R.T @ C
# # Move gaussians to random positions in the world
# xyz = (R.T @ (xyz - T[None, :]).T).T

# gt_depth.requires_grad_(True)
# xyz.requires_grad_(True)
# scales.requires_grad_(True)
# quats.requires_grad_(True)
# colors.requires_grad_(True)
# opacity.requires_grad_(True)


f = 200
fx = f
fy = f

cx = 64
cy = 64

img_height = 128
img_width = 128

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
    if not ok:
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
    opacity=opacity.sigmoid(),
    scales=scales.exp(),
    rotations=quats / quats.norm(dim=1, keepdim=True).clamp(min=1e-6),
    colors_precomp=colors
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


# %%
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from PIL import Image
import mediapy as media
import io

def backproject_depth(depth_param, K):
    """
    Backprojects the depth map to 3D coordinates using the intrinsic matrix K.

    Parameters:
    - depth_param: The depth map (tensor of z coordinates).
    - K: The intrinsic matrix of the camera.

    Returns:
    - points_3d: The 3D coordinates of the points.
    """
    height, width = depth_param.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    i = i.reshape(-1)
    j = j.reshape(-1)
    z = depth_param.reshape(-1)
    ones = np.ones_like(z)
    pixels = np.vstack((i, j, ones))

    K_inv = np.linalg.inv(K)
    points_3d = K_inv @ (pixels * z)

    return points_3d

def fig_to_frames(fig, d=2.5):
    frames = []
    for angle in range(-180, 180, 5):
        camera = dict(
            eye=dict(y=d*np.sin(np.radians(angle)), z=0, x=d*np.cos(np.radians(angle)))
        )
        fig.update_layout(scene_camera=camera)
        img_bytes = pio.to_image(fig, format='png')
        img = Image.open(io.BytesIO(img_bytes))
        frames.append(np.array(img)[..., :3])
    return frames

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

# %%

focal_x = img_width / (2 * math.tan(FoVx * 0.5))
focal_y = img_height / (2 * math.tan(FoVy * 0.5))


xyz_homo = torch.cat([xyz, torch.ones_like(xyz[..., :1])], -1)
xyz_proj = (full_proj_transform.T @ xyz_homo.T).T
xyz_proj = xyz_proj / (xyz_proj[..., -1:] + 1e-7)
uv = torch.stack([
    ((xyz_proj[..., 0] + 1.0) * img_width - 1.0) * 0.5,
    ((xyz_proj[..., 1] + 1.0) * img_height - 1.0) * 0.5
], -1)
radii = ret["radii"]

o = opacity.detach().clone().sigmoid()
X = xyz.detach().clone()
Sc = scales.detach().clone().exp()
Ro = quats.detach().clone()
Ro = Ro / Ro.norm(dim=-1, keepdim=True).clamp(min=1e-6)
Rad = radii.detach().clone()
uv = uv.detach().clone()


Xt = (R @ X.T).T + T[None]
Xt = Xt
reind = torch.argsort(Xt[..., 2])

depth_param = torch.nn.Parameter(torch.ones(img_height, img_width, device="cuda") * Xt[...,-1].mean(), requires_grad=True)

cov3d = scale_rot_to_cov3d(Sc, Ro)
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

Xt = Xt[reind]
P = P[reind]
o = o[reind]
radii = Rad[reind]
uv = uv[reind]

# Xt[reind], P[reind], o[reind], radii[reind], uv[reind]
# %%
depth_param.mean()
# %%
import io
from IPython.display import display, clear_output

opt = torch.optim.Adam([depth_param], lr=5e-3)
she = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.9, verbose=True)
from tqdm import tqdm
with tqdm(range(5000)) as pbar:
    for itr in pbar:
        opt.zero_grad(set_to_none=True)
        D = torch.stack([Dx, Dy, Dz], dim=-1).cuda()
        Dmag = D.norm(dim=-1).clamp(min=1e-6)
        D /= Dmag[..., None]

        depth = torch.nn.functional.softplus(depth_param) + 0.1
        s = (depth * Dmag)[..., None]
        xs = (s * D).view(-1, 3, 1)
        D = D.view(-1, 3, 1)
        s = s.view(-1)
        i = i.reshape(-1).cuda()
        j = j.reshape(-1).cuda()

        Ssum = torch.zeros(img_height, img_width, device="cuda")
        Gsum = torch.zeros_like(Ssum)

        Smax = torch.ones_like(Ssum) * -1.175494351e-38
        # Smax = torch.zeros_like(Smax)
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
        # for Xi, pi, oi, ri, uvi in zip(Xt[reind], P[reind], o[reind], radii[reind], uv[reind]):
        index = 0
        for Xi, pi, oi, ri, uvi in zip(Xt, P, o, radii, uv):
            # print(index)
            index += 1
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
        loss_spatial = Gsum  -(Ssum.log() + Smax)
        loss = loss_spatial.sum()
        loss.backward()
        opt.step()
        she.step(loss)
        lr = opt.param_groups[0]['lr']
        pbar.set_postfix_str(f"loss: {loss.item():.5e} lr: {lr:.2e}")
        if lr < 1e-7:
            break
        
        clear_output(wait=True)
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(depth.detach().cpu().numpy(), cmap='jet')
        plt.colorbar()
        with io.BytesIO() as buf:
            fig.savefig(buf, format='png')
            buf.seek(0)
            frame = Image.open(buf).convert('RGB')
        plt.close(fig)
        display(frame)

# %%
depth_param
# %%
plt.imshow(depth_param.detach().cpu().numpy(), cmap='jet')
plt.colorbar()
# %%
data = []
for i in range(len(xyz)):
    x, y, z = create_ellipsoid(cov3d[i].detach().cpu(), xyz[i].detach().cpu(), num_points=10)
    clr_scale = f"rgb{tuple(colors[i].detach().cpu().tolist())}"
    clr_scale = [[0, clr_scale], [1, clr_scale]]
    # data.append(go.Scatter3d(
    #     x=x.flatten(),
    #     y=z.flatten(),  # Swap y and z to make y point up
    #     z=y.flatten(), # Swap y and z to make y point up and invert z to make it correct
    #     mode='markers',
    #     marker=dict(
    #         size=1,
    #         color=np.ones_like(z),
    #         colorscale=clr_scale,
    #         opacity=0.8
    #     )
    # ))
    data.append(go.Surface(x=x, y=z, z=y, colorscale=clr_scale, surfacecolor=np.ones_like(z), opacity=0.3, showscale=False))

points_3d = backproject_depth(depth_param.detach().cpu().numpy(), K)
clrs_3d = ret["render"].detach().cpu().numpy().transpose(1, 2, 0).reshape(-1, 3)
data.append(go.Scatter3d(
    x=points_3d[0],
    y=points_3d[2],
    z=points_3d[1],
    mode='markers',
    marker=dict(
        size=2,
        color=clrs_3d,
        opacity=1.0
    )
))

fig = go.Figure(data=data)
fig.update_layout(autosize=False, width=512, height=512)
fig.show()
# %%
media.show_video(fig_to_frames(fig), fps=10)
# %%

# %%
Xt
# %%
