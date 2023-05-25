import os
import numpy as np
import  torch
import torch.nn as nn
import torch.nn.functional as F

from util.nvdiffrast import MeshRenderer


def perspective_projection(focal, center):
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()


class BaseMorphableModel(nn.Module):
    def __init__(self, opt, device='cuda'):
        super().__init__()
        focal = opt.focal
        center = opt.center
        camera_distance = opt.camera_d

        self.shapeBasisSize = 80
        self.albedoBasisSize = 80
        self.expBasisSize = 64
        self.device = device
        self.opt = opt

        # load geometry
        bfm_dir = opt.bfm_folder
        dict = torch.load(os.path.join(bfm_dir, "morphableModel-2009.pkl"))
        self.shapeMean = dict['shapeMean'].to(device)  # [v,3] v=36709
        self.shapeMean = (self.shapeMean - torch.mean(self.shapeMean, dim=0, keepdim=True))
        self.shapePca = dict['shapePca'].to(device)  # [80,v,3]
        self.expressionPca = dict['expressionPca'].to(device)  # [64,v,3]
        self.faces = dict['faces'].to(device)  # [f,3]
        self.uvParametrization = torch.load(os.path.join(bfm_dir, "uvParametrization.pkl"))
        for key in self.uvParametrization:
            if key != 'uvResolution':
                self.uvParametrization[key] = self.uvParametrization[key].to(device)
        self.uvMap = self.uvParametrization['uvVertices'].to(device)
        self.uvMap = 1.0 - self.uvMap
        self.landmarksAssociation = dict['landmarksAssociation'].to(device)  # [68]
        self.point_buf = dict['point_buf'].to(device)  # [v,8]

        # set camera
        self.persc_proj = perspective_projection(focal, center)
        self.persc_proj = torch.tensor(self.persc_proj).to(device)
        self.camera_distance = camera_distance
        self.center = center
        self.focal = focal
        
        # set renderer
        fov = 2 * np.arctan(center / focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=5., zfar=15.,
            rasterize_size=224, renderer="pytorch3d",
        ).to(device)

    def generateTextureFromAlbedo(self, albedo):
        '''
        input: [b,v,c]
        output: [b,c,uh,uw]
        '''
        # assert (albedo.dim() == 3 and albedo.shape[-1] == self.diffuseAlbedoMean.shape[-1] and albedo.shape[-2] == self.diffuseAlbedoMean.shape[-2])
        textureSize = self.uvParametrization['uvResolution']
        halfRes = textureSize // 2
        baryCenterWeights = self.uvParametrization['uvFaces']
        oFaces = self.uvParametrization['uvMapFaces']
        uvxyMap = self.uvParametrization['uvXYMap']

        neighboors = torch.arange(self.faces.shape[-1], dtype = torch.int64, device = self.faces.device)

        texture = (baryCenterWeights[:, neighboors, None] * albedo[:, self.faces[oFaces[:, None], neighboors]]).sum(dim=-2)
        textures = torch.zeros((albedo.size(0), textureSize, textureSize, albedo.shape[-1]), dtype=torch.float32, device = self.faces.device)
        textures[:, uvxyMap[:, 0], uvxyMap[:, 1]] = texture
        textures[:, halfRes, :, :] = (textures[:, halfRes -1, :, :] + textures[:, halfRes + 1, :, :]) * 0.5
        return textures.permute(0, 2, 1, 3).flip([1]).permute(0, 3, 1, 2)

    def computeNormals(self, vertices):
        v1 = vertices[:, self.faces[:, 0]]
        v2 = vertices[:, self.faces[:, 1]]
        v3 = vertices[:, self.faces[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def computeShape(self, shapeCoff, expCoff):
        vertices = self.shapeMean + torch.einsum('ni,ijk->njk', (shapeCoff, self.shapePca)) + torch.einsum('ni,ijk->njk', (expCoff, self.expressionPca))
        return vertices

    def compute_vertex_from_texture(self, texture):
        '''
        input:
            texture: [b,c,uh,uw]
        
        output: 
            attr: [b,v,c]
        '''
        b = texture.shape[0]
        uvMap = self.uvMap[None, None, ...].repeat(b, 1, 1, 1)  # [b,1,v,2]
        uvMap = 2 * uvMap - 1  # [0,1] -> [-1,1]
        attr = F.grid_sample(texture, uvMap, mode="bilinear")  # [b,c,1,v]
        attr = attr.squeeze(2).permute(0, 2, 1)  # [b,v,c]
        return attr
    
    def to_image(self, vertices):
        face_proj = vertices @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]
        return face_proj
    
    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def compute_rotation(self, angles):
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])
        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def transform(self, face_shape, rot, trans):
        return face_shape @ rot + trans.unsqueeze(1)

    def get_view_reflect(self, normal, mode):
        '''
        return:
            view_dir: [b,3,h,w] camera point to surface
            reflect: [b,3,h,w] surface point to light
        '''
        b, _, h, w = normal.size()
        
        # construct view dir, camera point to surface
        if mode == "ortho":
            view_dir = torch.tensor([0, 0, -1.])[None, ..., None, None].to(normal.device)
            view_dir = view_dir.repeat(b, 1, h, w)  # [b,3,h,w]
        elif mode == "persp":
            b = normal.shape[0]
            res = int(self.center * 2)
            x = torch.arange(-self.center, self.center) + 0.5
            x = x[None, None, None, ...].repeat(b, 1, res, 1).to(normal.device)  # [b,1,h,w]
            y = -(torch.arange(-self.center, self.center) + 0.5)
            y = y[None, None, ..., None].repeat(b, 1, 1, res).to(normal.device)  # [b,1,h,w]
            z = -torch.ones_like(x) * self.focal
            view_dir = torch.cat([x, y, z], dim=1)  # [b,3,h,w]
        else:
            raise NotImplementedError
        
        # reflect = view - 2 <view, normal> normal
        # reflect - view = -2 <view, normal> normal
        view_dir = view_dir / (torch.sum(view_dir ** 2, dim=1, keepdim=True) + 1e-6).sqrt()
        view_dot_normal = torch.sum(view_dir * normal, dim=1, keepdim=True)
        reflect = view_dir - 2 * view_dot_normal * normal        
        return view_dir, reflect

    def forward(self):
        return
