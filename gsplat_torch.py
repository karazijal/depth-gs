import struct
from typing import Tuple

import torch
from torch import Tensor


def _quat_scale_to_covar_perci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    global_scale: float = 1.0,
    compute_covar: bool = True,
    compute_perci: bool = True,
    triu: bool = False,
):
    """PyTorch implementation."""
    quats = quats / torch.norm(quats, dim=-1, keepdim=True).clamp(min=1e-6)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )

    R = R.reshape(quats.shape[:-1] + (3, 3))  # (..., 3, 3)
    # R.register_hook(lambda grad: print("grad R", grad))

    if compute_covar:
        M = R * scales[..., None, :] * global_scale  # (..., 3, 3)
        covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            covars = covars.reshape(covars.shape[:-2] + (9,))  # (..., 9)
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # (..., 6)
    if compute_perci:
        P = R * (1 / scales[..., None, :])  # (..., 3, 3)
        percis = torch.bmm(P, P.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            percis = percis.reshape(percis.shape[:-2] + (9,))
            percis = (
                percis[..., [0, 1, 2, 4, 5, 8]] + percis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0

    return covars if compute_covar else None, percis if compute_perci else None

def _force_symmetric(covars: Tensor) -> Tensor:
    dest = torch.zeros_like(covars)
    dest[..., 0, 0] = covars[..., 0, 0]
    dest[..., 1, 1] = covars[..., 1, 1]
    dest[..., 2, 2] = covars[..., 2, 2]
    dest[..., 0, 1] = dest[..., 1, 0] = covars[..., 0, 1]
    dest[..., 0, 2] = dest[..., 2, 0] = covars[..., 0, 2]
    dest[..., 1, 2] = dest[..., 2, 1] = covars[..., 1, 2]
    return dest


def _persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """PyTorch implementation."""
    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N]
    rz = 1.0 / tz.clamp(min=1e-3)  # [C, N]
    rz2 = rz**2  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x = 1.3 * tan_fovx
    lim_y = 1.3 * tan_fovy
    tx = tz * torch.clamp(tx * rz, min=-lim_x, max=lim_x)
    ty = tz * torch.clamp(ty * rz, min=-lim_y, max=lim_y)

    O = torch.zeros((C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx * rz,       O, -fx * tx * rz2, 
               O, fy * rz, -fy * ty * rz2,
               O,       O,          O + 1], dim=-1 \
    ).view(C, N, 3, 3)

    # cov3d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    
    *batch, _, __ = covars.shape
    covars = covars.view(-1, 3, 3)
    J = J.view(-1, 3, 3)
    cov3d = (J @ covars @ J.transpose(-1, -2)).view(*batch, 3, 3)
    
    # means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means @ Ks[:, :2, :3].transpose(-1, -2)
    means2d = means2d * rz[..., None]  # [C, N, 2]
    means3d = torch.cat([means2d, tz[..., None]], dim=-1)
    return means3d, cov3d  # [C, N, 2], [C, N, 3, 3]


def _world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation."""
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # (C, N, 3)
    covars_c = torch.einsum("cij,njk,clk->cnil", R, covars, R)  # [C, N, 3, 3]
    return means_c, covars_c


def _get_tile_bbox(pix_center, pix_radius, tile_bounds, block_width):
    tile_size = torch.tensor(
        [block_width, block_width], dtype=torch.float32, device=pix_center.device
    )
    tile_center = pix_center / tile_size
    tile_radius = pix_radius[..., None] / tile_size

    top_left = (tile_center - tile_radius).to(torch.int32)
    bottom_right = (tile_center + tile_radius).to(torch.int32) + 1
    tile_min = torch.stack(
        [
            torch.clamp(top_left[..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    return tile_min, tile_max


def _projection(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    block_width: int = 16,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """PyTorch implementation."""
    means_c, covars_c = _world_to_cam(means, covars, viewmats)
    means3d, covars3d = _persp_proj(means_c, covars_c, Ks, width, height)
    
    covars3d = torch.nan_to_num(covars3d, nan=0.0, posinf=0.0, neginf=0.0)
    covars3d = (covars3d + covars3d.transpose(-1, -2)) * 0.5  # [C, N, 3, 3]

    det = (
        covars3d[..., 0, 0] * covars3d[..., 1, 1]
        - covars3d[..., 0, 1] * covars3d[..., 1, 0]
    )
    det_clamp = det.clamp(min=1e-5)

    covars3d = covars3d + torch.eye(3, device=covars3d.device, dtype=covars3d.dtype)[None, None] * eps2d
    # covars3d[..., 0, 0] = covars3d[..., 0, 0] + eps2d
    # covars3d[..., 1, 1] = covars3d[..., 1, 1] + eps2d
    # covars3d[..., 2, 2] = covars3d[..., 2, 2] + eps2d

    covars2d = covars3d[..., :2, :2]  # [C, N, 2, 2]

    conics = torch.stack([
        covars2d[..., 1, 1],
        -covars2d[..., 0, 1],
        covars2d[..., 0, 0],
    ], dim=-1) / det_clamp[..., None].clamp(min=1e-5)  # [C, N, 3]

    det3d = covars3d[..., 0, 0] * covars3d[..., 1, 1] * covars3d[..., 2, 2] + \
        2 * covars3d[..., 0, 1] * covars3d[..., 1, 2] * covars3d[..., 0, 2] - \
        covars3d[..., 0, 0] * covars3d[..., 1, 2]**2 - \
        covars3d[..., 1, 1] * covars3d[..., 0, 2]**2 - \
        covars3d[..., 2, 2] * covars3d[..., 0, 1]**2
    det3d = det3d.clamp(min=1e-5)

    conv_inv = torch.stack([
        covars3d[..., 1, 1] * covars3d[..., 2, 2] - covars3d[..., 1, 2]**2,
        covars3d[..., 0, 2] * covars3d[..., 1, 2] - covars3d[..., 0, 1] * covars3d[..., 2, 2],
        covars3d[..., 0, 1] * covars3d[..., 1, 2] - covars3d[..., 0, 2] * covars3d[..., 1, 1],
        covars3d[..., 0, 0] * covars3d[..., 2, 2] - covars3d[..., 0, 2]**2,
        covars3d[..., 0, 1] * covars3d[..., 0, 2] - covars3d[..., 0, 0] * covars3d[..., 1, 2],
        covars3d[..., 0, 0] * covars3d[..., 1, 1] - covars3d[..., 0, 1]**2,
    ], dim=-1)  # [C, N, 6]

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b**2 - det_clamp, min=0.1))  # (...,)
    v2 = b - torch.sqrt(torch.clamp(b**2 - det_clamp, min=0.1))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)

    valid = (det > 0) & (means3d[..., 2] > near_plane)
    radius[~valid] = 0.0
    inside = (
        (means3d[..., 0] + radius > 0)
        & (means3d[..., 0] - radius < width)
        & (means3d[..., 1] + radius > 0)
        & (means3d[..., 1] - radius < height)
    )
    radius[~inside] = 0.0
    tile_bounds = (
        (width + block_width - 1) // block_width,
        (height + block_width - 1) // block_width,
        1,
    )
    tile_min, tile_max = _get_tile_bbox(means3d[..., :2], radius, tile_bounds, block_width)
    tile_area = (tile_max[..., 0] - tile_min[..., 0]) * (
        tile_max[..., 1] - tile_min[..., 1]
    )

    # det_blur = (covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1]**2).clamp(min=1e-6)
    # compensation = torch.where((det > 0) & (det_blur > 0), torch.sqrt(det / det_blur), torch.tensor(0.0, device=det.device)).clamp(min=0.0)

    mask = inside & valid

    means3d = torch.nan_to_num(means3d, nan=0.0, posinf=0.0, neginf=0.0)
    means3d[~mask] = 0.0

    radius = torch.nan_to_num(radius, nan=0.0, posinf=0.0, neginf=0.0)
    radius[~mask] = 0.0

    conics = torch.nan_to_num(conics, nan=0.0, posinf=0.0, neginf=0.0)
    conics[~mask] = 0.0

    conv_inv = torch.nan_to_num(conv_inv, nan=0.0, posinf=0.0, neginf=0.0)
    conv_inv[~mask] = 0.0

    tile_area = torch.nan_to_num(tile_area, nan=0.0, posinf=0.0, neginf=0.0)
    tile_area[~mask] = 0.0

    # compensation = torch.nan_to_num(compensation, nan=0.0, posinf=0.0, neginf=0.0)
    # compensation[~mask] = 0.0
    
    compensation = torch.ones_like(det)
    
    num_tiles_hit = tile_area.int()
    radii = radius.int()
 
    return radii, means3d, conics, conv_inv, num_tiles_hit, compensation

def project(
    xyz: Tensor,  # [N, 3]
    scales: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    viewmat: Tensor,  # [4, 4]
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    block_width: int = 16,
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """PyTorch implementation."""
    Ks = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
    Ks[0, 0] = fx
    Ks[1, 1] = fy
    Ks[0, 2] = cx
    Ks[1, 2] = cy

    covars, _ = _quat_scale_to_covar_perci(quats, scales)
    covars = _force_symmetric(covars).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    radii, means3d, conics, cov3d_inv_utril, num_tiles_hit, compensation = _projection(
        xyz, covars, viewmat[None], Ks[None], img_width, img_height, near_plane=clip_thresh, block_width=block_width
    )
    return means3d[0], radii[0], conics[0], num_tiles_hit[0], compensation[0], cov3d_inv_utril[0]


@torch.no_grad()
def _isect_tiles(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pytorch implementation"""
    C, N = means2d.shape[:2]
    device = means2d.device

    # compute tiles_per_gauss
    tile_means2d = means2d / tile_size
    tile_radii = radii / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii[..., None]).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii[..., None]).int()
    tile_mins[..., 0] = torch.clamp(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = torch.clamp(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = torch.clamp(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = torch.clamp(tile_maxs[..., 1], 0, tile_height)
    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)  # [C, N]
    tiles_per_gauss *= radii > 0.0

    n_isects = tiles_per_gauss.sum().item()
    isect_ids = torch.empty(n_isects, dtype=torch.int64, device=device)
    gauss_ids = torch.empty(n_isects, dtype=torch.int32, device=device)

    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    tile_n_bits = (tile_width * tile_height).bit_length()

    def binary(num):
        return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))

    def kernel(cam_id, gauss_id):
        if radii[cam_id, gauss_id] <= 0.0:
            return
        index = cam_id * N + gauss_id
        curr_idx = cum_tiles_per_gauss[index - 1] if index > 0 else 0

        depth_id = struct.unpack("i", struct.pack("f", depths[cam_id, gauss_id]))[0]

        tile_min = tile_mins[cam_id, gauss_id]
        tile_max = tile_maxs[cam_id, gauss_id]
        for y in range(tile_min[1], tile_max[1]):
            for x in range(tile_min[0], tile_max[0]):
                tile_id = y * tile_width + x
                isect_ids[curr_idx] = (
                    (cam_id << 32 << tile_n_bits) | (tile_id << 32) | depth_id
                )
                gauss_ids[curr_idx] = gauss_id
                curr_idx += 1

    for cam_id in range(C):
        for gauss_id in range(N):
            kernel(cam_id, gauss_id)

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        gauss_ids = gauss_ids[sort_indices]

    return tiles_per_gauss.int(), isect_ids, gauss_ids


@torch.no_grad()
def _isect_offset_encode(
    isect_ids: Tensor, C: int, tile_width: int, tile_height: int
) -> Tensor:
    """Pytorch implementation"""
    tile_n_bits = (tile_width * tile_height).bit_length()

    n_isects = len(isect_ids)
    device = isect_ids.device
    tile_counts = torch.zeros(
        (C, tile_height, tile_width), dtype=torch.int64, device=device
    )

    isect_ids_uq, counts = torch.unique_consecutive(isect_ids >> 32, return_counts=True)

    cam_ids_uq = isect_ids_uq >> tile_n_bits
    tile_ids_uq = isect_ids_uq & ((1 << tile_n_bits) - 1)
    tile_ids_x_uq = tile_ids_uq % tile_width
    tile_ids_y_uq = tile_ids_uq // tile_width

    tile_counts[cam_ids_uq, tile_ids_y_uq, tile_ids_x_uq] = counts

    cum_tile_counts = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    offsets = cum_tile_counts - tile_counts
    return offsets.int()


def accumulate(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [C, N, channels]
    gauss_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    camera_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
    prefix_trans: Tensor = None,  # [C, image_height, image_width]
):
    from nerfacc import accumulate_along_rays, render_weight_from_alpha

    C, N = means2d.shape[:2]
    channels = colors.shape[-1]

    pixel_ids_x = pixel_ids % image_width
    pixel_ids_y = pixel_ids // image_width
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
    deltas = pixel_coords - means2d[camera_ids, gauss_ids]  # [M, 2]
    c = conics[camera_ids, gauss_ids]  # [M, 3]
    sigmas = (
        0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
        + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]
    alphas = torch.clamp_max(opacities[gauss_ids] * torch.exp(-sigmas), 0.999)

    if prefix_trans is not None:
        prefix_trans = prefix_trans[camera_ids, pixel_ids_y, pixel_ids_x]
    indices = (camera_ids * image_height * image_width + pixel_ids).long()
    total_pixels = C * image_height * image_width

    weights, trans = render_weight_from_alpha(
        alphas, ray_indices=indices, n_rays=total_pixels, prefix_trans=prefix_trans
    )
    renders = accumulate_along_rays(
        weights, colors[camera_ids, gauss_ids], ray_indices=indices, n_rays=total_pixels
    ).reshape(C, image_height, image_width, channels)
    accs = accumulate_along_rays(
        weights, None, ray_indices=indices, n_rays=total_pixels
    ).reshape(C, image_height, image_width, 1)

    return renders, accs