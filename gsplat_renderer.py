#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import fov2focal

from gsplat import project_gaussians, rasterize_gaussians, spherical_harmonics
from gsplat_torch import project

BLOCK_SIZE = 16

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, anti_aliased = True, ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    img_height = int(viewpoint_camera.image_height)
    img_width = int(viewpoint_camera.image_width)

    # W2C = torch.zeros((4, 4))
    # W2C[:3, :3] = torch.from_numpy(viewpoint_camera.R)
    # W2C[:3, 3] = torch.from_numpy(viewpoint_camera.T)
    # W2C[3, 3] = 1.0

    # W2C = W2C.cuda()

    # print("camera_center", viewpoint_camera.camera_center)
    # print("W2C", W2C)
    # print("T", viewpoint_camera.T, 'T=-RC', -torch.matmul(W2C[:3, :3], viewpoint_camera.camera_center))

    fy = fov2focal(viewpoint_camera.FoVy, img_height)
    fx = fov2focal(viewpoint_camera.FoVx, img_width)
    cx = img_width / 2
    cy = img_height / 2

    xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
        means3d=pc.get_xyz,
        scales=pc.get_scaling,
        glob_scale=scaling_modifier,
        quats=pc.get_rotation / pc.get_rotation.norm(dim=-1, keepdim=True),
        viewmat=viewpoint_camera.world_view_transform.T[:3, :].contiguous(),
        # projmat=viewpoint_camera.full_projection.T,
        fx = fx,
        fy = fy,
        cx = cx,
        cy = cy,
        img_height=img_height,
        img_width=img_width,
        block_width=BLOCK_SIZE,
    )
    # means3d, radii, conics, num_tiles_hit, comp, cov3d_inv_utril = project(
    #     pc.get_xyz,
    #     pc.get_scaling,
    #     pc.get_rotation / pc.get_rotation.norm(dim=-1, keepdim=True).clamp(min=1e-6),
    #     viewpoint_camera.world_view_transform.T[:, :].contiguous(),
    #     fx,
    #     fy,
    #     cx,
    #     cy,
    #     img_height=img_height,
    #     img_width=img_width,
    #     block_width=BLOCK_SIZE,
    # )
    
    # xys = means3d[:, :2]
    # depths = means3d[:, 2]

    try:
        xys.retain_grad()
    except:
        pass

    viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
    rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
    rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

    opacities = pc.get_opacity
    # if anti_aliased is True:
    #     opacities = opacities * comp[:, None]

    rgb, alpha = rasterize_gaussians(  # type: ignore
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,  # type: ignore
        rgbs,
        opacities,
        img_height=img_height,
        img_width=img_width,
        block_width=BLOCK_SIZE,
        background=bg_color,
        return_alpha=True,
    )  # type: ignore

    depth = torch.rand((img_height, img_width), device=rgb.device)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rgb.permute(2, 0, 1),
            "viewspace_points": xys,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "acc": alpha}