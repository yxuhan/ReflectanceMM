"""
This script is the differentiable renderer for Deep3DFaceRecon_pytorch
Attention, antialiasing step is missing in current version.

NOTE we use pytorch3d as our differentiable renderer, 
not the Nvdiffrac one used in the original Deep3DFaceRecon_pytorch repo
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from torch import nn


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)


class MeshRenderer(nn.Module):
    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10, 
                rasterize_size=224,
                renderer="pytorch3d"):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(
                torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.glctx = None
        self.renderer = renderer
    
    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, N, C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 

        vertex_ndc = vertex @ ndc_proj.t()            
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        ########
        # tri: [M,3]  vertex_ndc: [B,N,4]  vertex: [B,N,4]
        b = vertex_ndc.shape[0]
        nf = tri.shape[0]
        
        vertex_ndc = vertex_ndc[..., :-1] / vertex_ndc[..., -1:]
        vertex_ndc[..., :-1] *= -1
        faces = tri.unsqueeze(0).repeat(b, 1, 1)
        mesh = Meshes(vertex_ndc, faces)
        pix_to_face, _, bary_coords, _ = rasterize_meshes(mesh, rsize, faces_per_pixel=1, blur_radius=0)  # [b,h,w,1] [b,h,w,1,3]

        image = None
        attr = vertex[..., 2].unsqueeze(-1)  # [B,N,1] depth
        if feat is not None:
            attr = torch.cat([attr, feat], dim=-1)  # [B,N,4]
        c = attr.shape[-1]
        faces = faces.reshape(b, nf * 3, 1).repeat(1, 1, c)  # [b,3f,c]
        face_attributes = torch.gather(attr, dim=1, index=faces)  # [b,3f,c]
        face_attributes = face_attributes.reshape(b * nf, 3, c)  # in pytorch3d, the index of mesh#2's first vertex is set to nface
        output = interpolate_face_attributes(pix_to_face, bary_coords, face_attributes)
        output = output.squeeze(-2).permute(0, 3, 1, 2)  # [b,4,h,w]

        depth = output[:, :1]
        if feat is not None:
            image = output[:, 1:]
        
        mask = (pix_to_face > -1).permute(0, 3, 1, 2).float()

        return mask, depth, image
