import numpy as np
import os
import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import lpips
import math
from tqdm import tqdm
import kornia

from util import util

from .base_model import BaseModel
from . import networks
from .losses import (
    photo_loss,
    reg_loss,
    landmark_loss,
    weight_box_loss,
    model_reg_loss,
    identity_loss,
)
from .utils.sh import sh_to_envmap
from util.preprocess import estimate_norm_torch
from .facemodel.refmm import MorphableModel


class RefMMModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # Parameter Estimation Network arch
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
        
        # lighting PCA model configs
        parser.add_argument('--sh_model_path', type=str, default="RefMM/lighting_pca_model.pkl")
        parser.add_argument('--pca_num', type=int, default=80)
        parser.add_argument('--pca_scale_channel', type=int, default=3)

        # ReflectanceMM configs
        parser.add_argument('--camera_rig_path', type=str, default="RefMM/camera_rig.pkl")
        parser.add_argument('--scale_basis', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--bfm_folder', type=str, default='BFM')
        parser.add_argument('--refmm_path', type=str, default='RefMM/multipie_initial_refmm_model.pkl')

        # face reconstruction config
        parser.add_argument('--load_prefit_net', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--prefit_net_path', type=str)
        parser.add_argument('--update_model', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--recon_geo', type=util.str2bool, nargs='?', const=True, default=False)
        
        # model finetuning config
        parser.add_argument('--lr_mm_diff', type=float, default=1e-5)
        parser.add_argument('--lr_mm_spec', type=float, default=1e-5)

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)

        if is_train:
            parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'], help='face recog network structure')
            parser.add_argument('--net_recog_path', type=str, default='checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')

            # loss weights
            parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
            parser.add_argument('--w_feat', type=float, default=0.1, help='weight for feat loss')
            
            parser.add_argument('--w_reg', type=float, default=0.001, help='weight for reg loss')
            parser.add_argument('--w_tex', type=float, default=1., help='weight for tex_reg loss')
            parser.add_argument('--w_id', type=float, default=1., help='weight for id_reg loss')
            parser.add_argument('--w_exp', type=float, default=1., help='weight for exp_reg loss')
            parser.add_argument('--w_sh', type=float, default=1., help='weight for sh_reg loss')

            parser.add_argument('--w_gamma', type=float, default=10, help='weight for gamma loss')
            parser.add_argument('--w_weight_box', type=float, default=10, help='weight for spec weight box loss')
            parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
            parser.add_argument('--w_mm_norm', type=float, default=10., help='weight for model reg loss')
            parser.add_argument('--w_mm_diff_ortho', type=float, default=10., help='weight for model reg loss')

        opt, _ = parser.parse_known_args()
        parser.set_defaults(
            focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
        )

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon', 'facemodel']
        self.parallel_names = self.model_names

        # load PCA lighting model
        self.lightmodel = torch.load(opt.sh_model_path, map_location="cuda")

        # set face reconstruction network
        self.net_recon = networks.ReconLightPCANet(opt, self.lightmodel)
        if opt.load_prefit_net:
            self.net_recon.load_state_dict(torch.load(opt.prefit_net_path)["net_recon"])
        
        # set morphable model
        self.facemodel = MorphableModel(opt)
        self.var_dict = {
            "diff": self.facemodel.diffuseAlbedoPcaVar,
            "spec": self.facemodel.specularWeightPcaVar,
            "sh": self.lightmodel["sh_var"][:opt.pca_num],
        }
        if opt.isTrain:
            self.net_recog = networks.define_net_recog(
                net_recog=opt.net_recog, pretrained_path=opt.net_recog_path
            )
            self.parallel_names += ['net_recog']

        # set loss, loss func name: (compute_%s_loss) % loss_name            
        self.loss_names = [
            'all', 'color', 'reg', 'gamma', "weight_box", "lm", "feat", 'mm_norm', 'mm_diff_ortho',
            "diff_pc1", "diff_pc1_abs", "spec_pc1", "spec_pc1_abs", "lpips", "ssim", "psnr", # NOTE record only, do not backpropogate
        ]
        self.comupte_color_loss = photo_loss
        self.compute_feat_loss = identity_loss
        self.compute_reg_loss = reg_loss
        self.compute_weight_box = weight_box_loss

        self.compute_lm_loss = landmark_loss
        
        # NOTE if use lpips loss, add eps to lpips normalize_tensor function
        self.lpips = lpips.LPIPS(net="alex").cuda()
        self.lpips.requires_grad_(False)
        self.ssim = kornia.metrics.ssim
        self.psnr = kornia.metrics.psnr
        
        if opt.isTrain:
            # set optimizer
            self.optimizer_G = torch.optim.Adam(self.net_recon.parameters(), lr=opt.lr)                    
            self.optimizers = [self.optimizer_G]

            if opt.update_model:
                spec_params = [self.facemodel.specularAlbedoMean, self.facemodel.specularAlbedoPca]
                diff_params = [self.facemodel.diffuseAlbedoMean, self.facemodel.diffuseAlbedoPca]
                self.optimizer_mm_diff = torch.optim.Adam(diff_params, lr=opt.lr_mm_diff)
                self.optimizer_mm_spec = torch.optim.Adam(spec_params, lr=opt.lr_mm_spec)
                self.optimizers.append(self.optimizer_mm_diff)
                self.optimizers.append(self.optimizer_mm_spec)

    def set_input(self, input):
        self.input_img = input['imgs'].to(self.device)  # [b,3,h,w]
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None  # [b,1,h,w]
        
        if 'lms' in input:
            self.lm68 = input['lms'].to(self.device)
            self.gt_lm = self.lm68  # [b,68]

        self.img_paths = input['im_paths'] if 'im_paths' in input else None

        if 'geo_dict' in input:
            self.geo_dict = input['geo_dict']
            for k in self.geo_dict: self.geo_dict[k] = self.geo_dict[k].to(self.device)

        if 'vis_dict' in input:
            self.vis_dict = input['vis_dict']
            for k in self.vis_dict: self.vis_dict[k] = self.vis_dict[k].to(self.device)

        self.olat = input['olat'].to(self.device) if 'olat' in input else None  # [1,n,3,h,w]

        self.bs = self.input_img.shape[0]

    def forward(self, isTrain=False):
        '''
        ifTrain is used to determine whether flip the basis of reflectance model
        '''
        # recon net forward
        self.pred_coeffs_dict = self.net_recon(self.input_img)
        self.light_sh = self.pred_coeffs_dict["gamma"]  # [b,1or3,n**2]
        if not self.opt.recon_geo:
            for k in self.geo_dict:
                self.pred_coeffs_dict[k] = self.geo_dict[k]
        self.render(isTrain)
    
    def render(self, isTrain):
        # get per-vertex attributes
        self.pred_vertex, self.pred_diff, self.pred_spec, self.pred_norm, self.pred_lm = \
            self.facemodel.compute_for_render(self.pred_coeffs_dict)

        # get image-space attributes
        feat = torch.cat([self.pred_norm, self.pred_diff, self.pred_spec], dim=-1)
        self.pred_mask, _, self.pred_feat = self.facemodel.renderer(
            self.pred_vertex,
            self.facemodel.faces,
            feat=feat,
        )
        self.pred_norm_image = self.pred_feat[:, :3]  # x to right, y to up, z to front
        self.pred_diffuse_image = self.pred_feat[:, 3:6]  # [b,c,h,w]
        self.pred_specular_image = self.pred_feat[:, 6:]  # specular weight for RefMM

        # shading
        self.pred_diffuse_shading, self.pred_specular_shading = self.facemodel.shading_envmap(
            self.light_sh, self.pred_diffuse_image, self.pred_specular_image, self.pred_norm_image, self.pred_mask
        )
        self.pred_face = self.pred_diffuse_shading + self.pred_specular_shading  # [b,3,h,w]

    def eval_photometric_loss(self):
        pred = self.hdr_to_ldr(torch.clamp(self.pred_face, min=1e-6)) * self.pred_mask
        gt = self.input_img * self.pred_mask
        self.loss_psnr = self.psnr(pred, gt, max_val=1.)
        self.loss_ssim = self.ssim(pred, gt, window_size=3).mean()
        self.loss_lpips = self.lpips(pred, gt, normalize=True).mean()

    def compute_losses(self):
        with torch.no_grad():
            diff_coeff = self.pred_coeffs_dict["diff"][:, 0]
            spec_coeff = self.pred_coeffs_dict["spec"][:, 0]
            self.loss_diff_pc1 = torch.mean(diff_coeff)
            self.loss_diff_pc1_abs = torch.mean(diff_coeff.abs())
            self.loss_spec_pc1 = torch.mean(spec_coeff)
            self.loss_spec_pc1_abs = torch.mean(spec_coeff.abs())
            self.eval_photometric_loss()
            self.diff_coeff = diff_coeff

        pred_face = self.pred_face

        # photometric loss
        hdr_input_img = self.ldr_to_hdr(self.input_img)
        self.loss_color = self.opt.w_color * self.comupte_color_loss(
            pred_face, hdr_input_img, self.atten_mask * self.pred_mask)

        if self.opt.w_feat > 0:
            trans_m = estimate_norm_torch(self.lm68, self.input_img.shape[-2])
            pred_feat = self.net_recog(pred_face, trans_m)
            gt_feat = self.net_recog(hdr_input_img, trans_m)
            self.loss_feat = self.opt.w_feat * self.compute_feat_loss(pred_feat, gt_feat)
        else:
            self.loss_feat = 0.
                
        # reg loss
        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.var_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        # box loss for specular weight
        self.loss_weight_box = self.opt.w_weight_box * self.compute_weight_box(self.pred_specular_image)

        # landmark loss
        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm)

        # model reg loss
        norm_loss, diff_ortho_loss = model_reg_loss(self.facemodel, self.opt)
        self.loss_mm_norm = self.opt.w_mm_norm * norm_loss
        self.loss_mm_diff_ortho = self.opt.w_mm_diff_ortho * diff_ortho_loss

        self.loss_all = self.loss_color + self.loss_reg + self.loss_gamma + \
            self.loss_weight_box + self.loss_lm + self.loss_feat + \
            self.loss_mm_norm + self.loss_mm_diff_ortho
            
    def optimize_parameters(self, isTrain=True):
        self.forward(isTrain)
        self.compute_losses()
        # with autograd.detect_anomaly():
        if isTrain:
            self.optimizer_G.zero_grad()
                        
            if self.opt.update_model:
                self.optimizer_mm_diff.zero_grad()
                self.optimizer_mm_spec.zero_grad()
            
            self.loss_all.backward()

            if self.opt.update_model:
                self.optimizer_mm_diff.step()
                self.optimizer_mm_spec.step()
            
            self.optimizer_G.step()

    def hdr_to_ldr(self, x):
        return torch.pow(x, 1 / 1.2)

    def ldr_to_hdr(self, x):
        return torch.pow(x, 1.2)
    
    def compute_visuals(self):
        normal = self.vis_geometry(self.pred_norm_image)

        pred_face = self.hdr_to_ldr(self.pred_face)
        rerender = pred_face * self.pred_mask
        overlay = rerender + (1 - self.pred_mask) * self.input_img
        
        light = sh_to_envmap(self.light_sh, h=224, w=224)
        light[light < 0] = 0
        # light = light ** (1 / 2.2)
        light = (light / (light + 1)) ** (1 / 2.2)

        ldr_diff = self.hdr_to_ldr(self.pred_diffuse_image)
        ldr_diff_shading = self.hdr_to_ldr(self.pred_diffuse_shading)
        specular = rerender - ldr_diff_shading

        self.output_vis = torch.cat([
            self.input_img,
            normal,
            ldr_diff,
            ldr_diff_shading,
            specular,
            rerender,
            overlay,
            light,
        ], dim=-1)
    
    def visualize_mm(self):
        with torch.no_grad():
            model_diff_uv, model_spec_uv = self.facemodel.visualize_model()  # [80,nvis,c,uh,uw]
            model_spec_abd = torch.sum(model_spec_uv, dim=2, keepdim=True)
            model_spec_uv = torch.cat([model_spec_uv, model_spec_abd], dim=2)
            point_light_render_list = []
            try:
                for i in range(self.bs):
                    diff_coeff = self.pred_diff[i:i+1]
                    spec_coeff = self.pred_spec[i:i+1]
                    diffuse_image, shading, neg_mask = self.facemodel.shading_camera_rig_1view(  # [n,3,h,w]
                        diff_coeff, spec_coeff,
                    )
                    specular_image = (shading - diffuse_image).expand_as(shading)
                    point_light_render_list.append(torch.stack([diffuse_image, shading, specular_image, neg_mask], dim=1))  # [n,4,3,h,w]
                point_light_render_list = torch.stack(point_light_render_list, dim=0)  # [bs,n,3,3,h,w]
                model_diff_uv = self.hdr_to_ldr(model_diff_uv)
                point_light_render_list = self.hdr_to_ldr(point_light_render_list)
            except:
                pass
        return model_diff_uv, model_spec_uv, point_light_render_list
