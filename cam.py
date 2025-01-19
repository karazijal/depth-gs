import torch
import math
from operator import itemgetter


def project(x):
    return x[..., :-1] / x[..., -1:]


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


class OpenGLCamera:
    def __init__(self, W, H, K, R, t):
        self.W = W
        self.H = H
        self.K = K
        self.R = R
        self.t = t

    @property
    def Q(self):
        return torch.linalg.inv(self.P)

    @property
    def P(self):
        M = torch.cat(
            (
                torch.cat([self.R, self.t.unsqueeze(-1)], dim=1),
                torch.tensor([[0, 0, 0, 1]]),
            ),
            dim=0,
        )
        return self.K @ M

    def pixel_index_to_coord(self, index):
        """Convert pixel indices to pixel coordinates

        An :param:`index` is a pair of integers :math:`(i_x, i_y)` where
        :math:`0 \leq i_x < W` and :math:`0 \leq i_y < H`. A :param:`coord`
        is a pair of floats :math:`(u, v)` where :math:`-1 \leq u \leq 1`
        and :math:`-1 \leq v \leq 1`.

        Args:
            index (Tensor[..., 2]): indices.

        Returns:
            Tensor[..., 2]: coordinates.
        """
        return (2 * index + 1) / index.new_tensor([self.W, self.H]) - 1

    def pixel_coord_to_ray(self, u):
        """Convert pixel coordinates to rays

        An :param:`index` is a pair of integers :math:`(i_x, i_y)` where
        :math:`0 \leq i_x < W` and :math:`0 \leq i_y < H`. A :param:`coord`
        is a pair of floats :math:`(u, v)` where :math:`-1 \leq u \leq 1`
        and :math:`-1 \leq v \leq 1`.

        Args:
            u (Tensor[..., 2]): coordinates.

        Returns:
            - Tensor[..., 2]: first point on the ray.
            - Tensor[..., 2]: last point on the ray.
        """
        u1 = torch.cat(
            (u, -torch.ones_like(u[..., :1]), torch.ones_like(u[..., :1])), dim=-1
        )
        u2 = torch.cat(
            (u, +torch.ones_like(u[..., :1]), torch.ones_like(u[..., :1])), dim=-1
        )
        p1 = project((self.Q @ u1.unsqueeze(-1)).squeeze(-1))
        p2 = project((self.Q @ u2.unsqueeze(-1)).squeeze(-1))
        return p1, p2

    def pixel_index_to_ray(self, index):
        u = self.pixel_index_to_coord(index)
        return self.pixel_coord_to_ray(u)

    def project(self, x):
        x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        x = (self.P @ x.unsqueeze(-1)).squeeze(-1)
        z = x[..., 2]
        return project(x), z

    def look_at(self, eye, center, up):
        eye = eye.type(torch.get_default_dtype())
        center = center.type(torch.get_default_dtype())
        up = up.type(torch.get_default_dtype())
        z = normalize(eye - center)
        x = normalize(torch.cross(up, z))
        y = normalize(torch.cross(z, x))
        self.R = torch.stack((x, y, z), dim=-1).T
        self.t = -self.R @ eye

        # self.t = eye


def make_opengl_camera(W, H, l, r, b, t, n, f):
    """
    Make an OpenGL camera

    Args:

    - W (int): image width
    - H (int): image height
    - r (float): right
    - l (float): left
    - b (float): bottom
    - t (float): top
    - n (float): near
    - f (float): far
    """
    K = torch.tensor(
        [
            [2 * abs(n) / (r - l), 0, (r + l) / (r - l), 0],
            [0, 2 * abs(n) / (t - b), (t + b) / (t - b), 0],
            [0, 0, -(f + n) / (f - n), -2 * f * n / abs(f - n)],
            [0, 0, -1, 0],
        ]
    )
    return OpenGLCamera(W, H, K, torch.eye(3), torch.zeros(3))


def make_opengl_camera_from_fov(W, H, fov, n, f):
    r = math.tan(fov / 2) * abs(n)
    t = r * H / W
    return make_opengl_camera(W, H, -r, r, -t, t, n, f)


class Gaussian:
    def __init__(self, color, opacity, mean, S):
        d = torch.get_default_dtype()
        self.color = color.type(d)
        self.opacity = float(opacity)
        self.mean = mean.type(d)
        self.S = S.type(d)

    def __call__(self, x):
        x_ = x - self.mean
        z_ = (torch.linalg.inv(self.S) @ x_.unsqueeze(-1)).squeeze(-1)
        return torch.exp(-0.5 * (x_ * z_).sum(-1))


def render(cam, gaussians, N=256):
    ix, iy = torch.meshgrid(torch.arange(cam.W), torch.arange(cam.H), indexing="xy")
    indices = torch.stack((ix, iy), dim=-1)
    p1, p2 = cam.pixel_index_to_ray(indices)
    delta = (p2 - p1).norm(dim=-1) / (N - 1)

    color = torch.zeros(cam.H, cam.W, 3)
    log_alpha = color.new_zeros(cam.H, cam.W)
    prev_opacity = color.new_zeros(cam.H, cam.W)
    prev_color_func = torch.zeros(cam.H, cam.W, 3)

    for r in torch.linspace(0, 1, N):
        x = (1 - r) * p1 + r * p2
        opacity = torch.zeros(cam.H, cam.W)
        color_func = torch.zeros(cam.H, cam.W, 3)
        for g in gaussians:
            opacity_ = g.opacity * g(x)
            opacity += opacity_
            color_func += g.color * opacity_.unsqueeze(-1)
        log_alpha -= (prev_opacity + opacity) * 0.5 * delta
        color_func *= torch.exp(log_alpha).unsqueeze(-1)
        color += (prev_color_func + color_func) * 0.5 * delta.unsqueeze(-1)
        prev_opacity = opacity
        prev_color_func = color_func
    return color.permute(2, 0, 1)


def get_dist_volume(cam, gaussians, N=256):
    ix, iy = torch.meshgrid(torch.arange(cam.W), torch.arange(cam.H), indexing="xy")
    indices = torch.stack((ix, iy), dim=-1)
    p1, p2 = cam.pixel_index_to_ray(indices)
    dist = (p2 - p1).norm(dim=-1)
    delta = dist / (N - 1)

    dist_p = torch.zeros(N, cam.H, cam.W)
    dist_v = torch.zeros(N, cam.H, cam.W)
    log_alpha = torch.zeros(cam.H, cam.W)
    prev_opacity = torch.zeros(cam.H, cam.W)

    for j, r in enumerate(torch.linspace(0, 1, N)):
        x = (1 - r) * p1 + r * p2
        opacity = torch.zeros(cam.H, cam.W)
        for g in gaussians:
            opacity_ = g.opacity * g(x)
            opacity += opacity_
        dist_p[j, :, :] = opacity * torch.exp(log_alpha)
        dist_v[j, :, :] = (x - p1).norm(dim=-1)
        log_alpha -= (prev_opacity + opacity) * 0.5 * delta
        prev_opacity = opacity
    return dist_p, dist_v, p1, p2


def render25D(cam, gaussians):
    ix, iy = torch.meshgrid(torch.arange(cam.W), torch.arange(cam.H), indexing="xy")
    indices = torch.stack((ix, iy), dim=-1)
    u = cam.pixel_index_to_coord(indices)

    color = torch.zeros(3, cam.H, cam.W)
    alpha = color.new_ones(cam.H, cam.W)

    for g, z in sorted(gaussians, key=itemgetter(1)):
        opacity = 1 - torch.exp(-g.opacity * g(u))
        color += ((opacity * alpha).unsqueeze(-1) * g.color).permute(2, 0, 1)
        alpha *= 1 - opacity
    return color


def project_gaussian(cam, gaussian):
    mu = gaussian.mean
    u0, z = cam.project(gaussian.mean)

    # u = cam.project(gaussian.mean)[:2]
    # p1, p2 = cam.pixel_coord_to_ray(u)
    # nu0 = (p2 - p1) / (p2 - p1).norm()

    S = gaussian.S

    Q = cam.Q
    den = Q[3, :3] @ u0 + Q[3, 3]
    A = (Q[:3, :2] - mu[:, None] @ Q[3:4, :2]) / den
    a = (Q[:3, 2:3] - mu[:, None] * Q[3, 2]) / den
    nu = a / a.norm()

    C = torch.linalg.inv(S)
    Z = nu.T @ C @ nu
    Cnu = C - (C @ nu @ nu.T @ C) / Z
    Ctilde = A.T @ Cnu @ A
    Stilde = torch.linalg.inv(Ctilde)

    P = cam.P
    den = P[3:4, :3] @ mu + P[3, 3]
    B = (P[:2, :3] - u0[:2, None] @ P[3:4, :3]) / den
    b = (P[2:3, :3] - u0[2] * P[3:4, :3]) / den
    k = b.T * a.norm()

    # (B @ S @ b.T).T @ (B@S@B.T) @ (B @ S @ b.T) = 0 ?

    S = gaussian.S
    Stilde_ = B @ S @ B.T

    Ji = torch.cat([A, a], dim=1)
    Cp = Ji.T @ C @ Ji
    Cp33 = a.T @ C @ a
    J = torch.cat([B, b], dim=0)
    Sp = J @ S @ J.T
    Sp33 = b @ S @ b.T
    # Sp33 = Sp[2,2]
    delta = Sp[2:3, :2] @ torch.linalg.inv(Sp[:2, :2]) @ Sp[:2, 2:3]
    # delta_ = (B @ S @ b.T).T @ Ctilde @ (B @ S @ b.T) #
    delta_ = (B @ S @ b.T).T @ Ctilde @ (B @ S @ b.T)
    delta__ = b @ S @ B.T @ Ctilde @ B @ S @ b.T
    delta___ = b @ S @ B.T @ Ctilde @ B @ S @ b.T
    iSp33 = 1 / (Sp33 - delta)
    # iSp33 = 1 / (Sp33 - (B @ S @ b.T).T @ (B@S@B.T) @ (B @ S @ b.T))

    Z_ = iSp33 / (a.norm() ** 2)

    torch.testing.assert_close(delta.item(), delta_.item())
    torch.testing.assert_close(Cp[2, 2].item(), Cp33.item())
    torch.testing.assert_close(Sp[2, 2].item(), Sp33.item())

    torch.testing.assert_close(Stilde, Stilde_)
    torch.testing.assert_close(Z, Z_)
    # Inu = torch.eye(3) - nu @ nu.T
    # Snu = Inu @ S @ Inu
    # Stilde = B @ Snu @ B.T

    opacity__ = gaussian.opacity * torch.sqrt(2 * math.pi / Z)
    return Gaussian(gaussian.color, opacity__, u0[:2], Stilde), z
