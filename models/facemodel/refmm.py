import  torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import math

from models.utils.sh import SH
from .basemm import BaseMorphableModel


class MorphableModel(BaseMorphableModel):
    def __init__(self, opt, device='cuda'):
        super().__init__(opt, device=device)
        refmm_path = opt.refmm_path
        camera_rig_path = opt.camera_rig_path
        update_model = opt.update_model
        
        # load reflectance model
        # for a face vertex we have reflectance parameters c, w_1, w_8, w_64
        # c is the diffuse albedo
        # w_i is the linear combination weight for the Blinn-Phong lobe with specular exp i
        # NOTE 
        # when building the initial PCA model on Multi-PIE, 
        # we first compute the specular albedo as s = w_1 + w_8 + w_64
        # then, we compute the normalized weight as w_i' = w_i / s (this way, w_1' + w_8' + w_64' = 1)
        # next, we learn a PCA model for the diffuse albedo c and transferred it to s, w_1', w_8', w_64'
        # during model finetuning, we only update the model parameters of c and s, 
        # while keeping the model of w_1', w_8', w_64' unchanged
        # we empirically find this implementation works better than update 
        # all the ReflectanceMM parameters (i.e. w_1, w_8, w_64, and c) during model finetuning
        rdict = torch.load(refmm_path)
        self.diffuseAlbedoMean = rdict["diff_mean"].to(device)  # [v,3]
        self.diffuseAlbedoPca = rdict["diff_basis"][:self.albedoBasisSize].to(device)  # [80,v,3]
        self.diffuseAlbedoPcaVar = rdict["diff_var"][:self.albedoBasisSize].to(device)  # [80]
        specularMean = rdict["spec_mean"].to(device)  # [v,k+1]
        self.specularAlbedoMean = specularMean[:, :1]  # [v,1]
        self.specularWeightMean = specularMean[:, 1:]  # [v,k]
        specularPca = rdict["spec_basis"][:self.albedoBasisSize].to(device)
        self.specularAlbedoPca = specularPca[..., :1]  # [80,v,1]
        self.specularWeightPca = specularPca[..., 1:]  # [80,v,k]
        # NOTE in the transferred model, we have spec_var = diff_var
        self.specularWeightPcaVar = rdict["spec_var"][:self.albedoBasisSize].to(device)  # [80]
        self.expBasis = rdict["exp_basis"]  # len=k
        self.specSH = rdict["spec_sh"].to(device)  # [k,n] n is the order of SH
        self.diffSH = rdict["diff_sh"].to(device)  # [n]

        if opt.scale_basis:
            self.diffuseAlbedoPca = self.diffuseAlbedoPca * self.diffuseAlbedoPcaVar.sqrt()[..., None, None]
            self.diffuseAlbedoPcaVar = torch.ones_like(self.diffuseAlbedoPcaVar)
            self.specularAlbedoPca = self.specularAlbedoPca * self.specularWeightPcaVar.sqrt()[..., None, None]
            self.specularWeightPca = self.specularWeightPca * self.specularWeightPcaVar.sqrt()[..., None, None]
            self.specularWeightPcaVar = torch.ones_like(self.specularWeightPcaVar)
        
        self.camera_rig = torch.load(camera_rig_path)
        for k in self.camera_rig:
            self.camera_rig[k] = self.camera_rig[k].to(device)
        
        # record old reflectance model
        self.oldDiffuseAlbedoMean = torch.clone(self.diffuseAlbedoMean)
        self.oldDiffuseAlbedoPca = torch.clone(self.diffuseAlbedoPca)
        self.oldspecularAlbedoMean = torch.clone(self.specularAlbedoMean)
        self.oldspecularAlbedoPca = torch.clone(self.specularAlbedoPca)

        # set network parameters
        if update_model:
            self.diffuseAlbedoMean = nn.parameter.Parameter(self.diffuseAlbedoMean)
            self.diffuseAlbedoPca = nn.parameter.Parameter(self.diffuseAlbedoPca)
            self.specularAlbedoMean = nn.parameter.Parameter(self.specularAlbedoMean)
            self.specularAlbedoPca = nn.parameter.Parameter(self.specularAlbedoPca)

    def computeReflectance(self, diffAlbedoCoeff, specWeightCoeff):
        diffuseAlbedoPca = self.diffuseAlbedoPca
        specularWeightPca = self.specularWeightPca
        specularAlbedoPca = self.specularAlbedoPca        
        diffAlbedo = self.diffuseAlbedoMean + \
            torch.einsum('ni,ijk->njk', (diffAlbedoCoeff, diffuseAlbedoPca))
        specWeight = self.specularWeightMean + \
            torch.einsum('ni,ijk->njk', (specWeightCoeff, specularWeightPca))
        specAlbedo = self.specularAlbedoMean + \
            torch.einsum('ni,ijk->njk', (specWeightCoeff, specularAlbedoPca))
        specWeight = specAlbedo * specWeight
        return diffAlbedo, specWeight

    def computeShapeAlbedo(self, shapeCoeff, expCoeff, diffAlbedoCoeff, specWeightCoeff):
        vertices = self.computeShape(shapeCoeff, expCoeff)
        diffAlbedo, specWeight = self.computeReflectance(diffAlbedoCoeff, specWeightCoeff)
        return vertices, diffAlbedo, specWeight

    def compute_for_render(self, coeff_dict):
        shapeCoeff = coeff_dict["id"]
        expCoeff = coeff_dict["exp"]
        diffAlbedoCoeff = coeff_dict["diff"]
        specWeightCoeff = coeff_dict["spec"]  # for transferred model, need to manually add the key "spec"
        translation = coeff_dict["trans"]
        rotation = self.compute_rotation(coeff_dict["angle"])
        
        can_vertices, diffAlbedo, specWeight = self.computeShapeAlbedo(
            shapeCoeff,
            expCoeff,
            diffAlbedoCoeff,
            specWeightCoeff,
        )
        
        can_normal = self.computeNormals(can_vertices)  # canonical space
        world_normal = can_normal @ rotation  # world space

        # world space
        world_vertices = self.transform(
            can_vertices,
            rotation,
            translation,
        )
        # camera space
        # camera is locate at world space of [0, 0, d] and looking at -z axis
        # after this transformation, camera is set at [0, 0, 0]
        camera_vertices = self.to_camera(world_vertices)
        
        proj_vertices = self.to_image(camera_vertices)
        proj_lms = proj_vertices[:, self.landmarksAssociation]
         
        return camera_vertices, diffAlbedo, specWeight, world_normal, proj_lms

    def shading_envmap(self, light_sh, diffuseAlbedo, specularWeight, normal, mask):
        '''
        light_sh: [b,c,n2] c = 1 or 3
        diffuseAlbedo: [b,3,h,w]
        specularWeight: [b,k,h,w]
        normal: [b,3,h,w]
        '''
        b, c, n2 = light_sh.size()
        assert c == 1 or c ==3
        sh_order = int(math.sqrt(n2))
        light_sh = light_sh.reshape(b, c, 1, 1, n2)
        _, reflect = self.get_view_reflect(normal, "persp")  # [b,3,h,w]
        dx, dy, dz = normal[:, :1], normal[:, 1:2], normal[:, 2:]
        sx, sy, sz = reflect[:, :1], reflect[:, 1:2], reflect[:, 2:]
        diff_sh = self.diffSH[None, None, None, None, ...]  # [1,1,1,1,n]
        spec_sh = self.specSH[:, None, None, None, None, ...]  # [k,1,1,1,1,n]
        diff_shading = torch.zeros_like(normal)  # [b,3,h,w]
        spec_shading = torch.zeros_like(normal)[None, ...].repeat(len(self.expBasis), 1, 1, 1, 1)  # [k,b,3,h,w]
        cnt = 0
        for l in range(sh_order):
            for m in range(-l, l + 1):
                # [1,1,1,1] x [b,1|3,1,1] x [b,3,h,w] = [b,3,h,w]
                diff_shading += diff_sh[..., l] * light_sh[..., cnt] * SH(l, m, dx, dy, dz)
                # [k,1,1,1,1] x [b,1|3,1,1] x [b,3,h,w] = [k,b,3,h,w]
                spec_shading += spec_sh[..., l] * light_sh[..., cnt] * SH(l, m, sx, sy, sz) 
                cnt += 1
        specularWeight = specularWeight.transpose(0, 1).unsqueeze(2)  # [k,b,1,h,w]
        spec_shading = torch.sum(specularWeight * spec_shading, dim=0)  # [b,3,h,w]
        diff_shading = diff_shading * diffuseAlbedo
        return diff_shading * mask, spec_shading * mask

    def shading_camera_rig_1view(self, diffuse, weight):
        '''
        diffuse: [1,v,3]
        weight: [1,v,n]

        b: num of view
        k: num of point light (k=b)
        '''
        visibility = self.camera_rig["visibility"]  # [v,b]
        cosine = self.camera_rig["clamped_cosine"]  # [v,b]
        light_dir = self.camera_rig["light_dir"]  # [b,v,3]
        normal = self.camera_rig["normal"]  # [v,3]
        angle = self.camera_rig["angle"]  # [p,t,3]
        p, t, _ = angle.size()
        angle = angle.reshape(-1, 3)  # [ b,3]
        b = visibility.shape[-1]
        coeff_dict = {
            "id": torch.zeros(b, 80).to(self.device),
            "exp": torch.zeros(b, 64).to(self.device),
            "diff": torch.zeros(b, 80).to(self.device),
            "spec": torch.zeros(b, 80).to(self.device),
            "trans": torch.zeros(b, 3).to(self.device),
            "angle": angle,
        }
        vertices, albedo, _, _, _ = self.compute_for_render(coeff_dict)

        visibility = visibility.transpose(0, 1)[..., None]  # [k,v,1]
        cosine = cosine.transpose(0, 1)[..., None]  # [k,v,1]
        diffuse_shading = diffuse * cosine  # [k,v,3]

        view_idx = (b + 1) // 2
        half_dir = (light_dir + light_dir[view_idx]) / 2  # [k,v,3]
        half_dir = half_dir / torch.sum(half_dir ** 2, dim=-1, keepdim=True).sqrt()
        lobe = torch.sum(half_dir * normal, dim=-1, keepdim=True)  # [k,v,1]
        lobe_list = []
        for exp in self.expBasis:
            cur_lobe = torch.pow(lobe, exp) * (exp + 2) / (8 * math.pi)
            lobe_list.append(cur_lobe)
        lobe_list = torch.cat(lobe_list, dim=-1)  # [k,v,n]
        specular_shading = torch.sum(lobe_list * weight, dim=-1, keepdim=True)  # [k,v,1]
        cur_vertices = vertices[view_idx:view_idx+1].repeat(b, 1, 1)  # [k,v,3]
        feat = torch.cat([diffuse_shading * visibility, specular_shading * visibility], dim=-1)
        _, _, pred_feat = self.renderer(
            cur_vertices,
            self.faces,
            feat=feat,
        )
        diffuse_image = pred_feat[:, :3] / math.pi
        specular_image = pred_feat[:, 3:]
        shading = diffuse_image + specular_image
        neg_mask = (specular_image < 0).float().expand_as(shading)
        return diffuse_image, shading, neg_mask
    
    def hdr_to_ldr(self, x):
        return torch.pow(x, 1 / 2.2)

    def visualize_model(self):
        '''
        generate mean +- [1,2,3] basis
        '''
        n_vis = 3
        std_mul = 3.
        diff_std = self.diffuseAlbedoPcaVar.sqrt()[..., None, None] * std_mul # [80,1,1]
        spec_std = self.specularWeightPcaVar.sqrt()[..., None, None] * std_mul  # [80,1,1]
        diff_list, spec_list = [], []
        for i in range(-n_vis, n_vis + 1):
            diff = self.diffuseAlbedoMean + i * diff_std * self.diffuseAlbedoPca / n_vis  # [80,v,c]
            spec = self.specularWeightMean + i * spec_std * self.specularWeightPca / n_vis
            spec_abd = self.specularAlbedoMean + i * spec_std * self.specularAlbedoPca / n_vis
            spec = spec * spec_abd
            diff_uv = self.generateTextureFromAlbedo(diff)  # [80,c,uh,uw]
            spec_uv = self.generateTextureFromAlbedo(spec)
            diff_list.append(diff_uv)
            spec_list.append(spec_uv)
        diff_list = torch.stack(diff_list, dim=1)  # [80,nvis,c,uh,uw]
        spec_list = torch.stack(spec_list, dim=1)  # [80,nvis,c,uh,uw]
        return diff_list, spec_list * 5.
