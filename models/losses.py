import numpy as np
import torch


def identity_loss(id_featureA, id_featureB):
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
    return torch.sum(1 - cosine_d) / cosine_d.shape[0] 


def photo_loss(imageA, imageB, mask, eps=1e-6):
    loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss


def landmark_loss(predict_lm, gt_lm):
    weight = np.ones([68])
    weight[27:30] = 20
    weight[-8:] = 20
    weight = np.expand_dims(weight, 0)
    weight = torch.tensor(weight).to(predict_lm.device)
    loss = torch.sum((predict_lm - gt_lm)**2, dim=-1) * weight
    loss = torch.sum(loss) / (predict_lm.shape[0] * predict_lm.shape[1])
    return loss


def reg_loss(coeffs_dict, var_dict, opt):
    w_tex, w_sh, w_id, w_exp = opt.w_tex, opt.w_sh, opt.w_id, opt.w_exp
    creg_loss = w_tex * torch.sum(coeffs_dict["diff"].pow(2) / var_dict["diff"]) + \
                w_tex * torch.sum(coeffs_dict["spec"].pow(2) / var_dict["spec"]) + \
                w_sh * torch.sum(coeffs_dict["gamma_pca"].pow(2) / var_dict["sh"])

    creg_loss = creg_loss + w_id * torch.sum(coeffs_dict["id"].pow(2)) + \
        w_exp * torch.sum(coeffs_dict["exp"].pow(2))
    creg_loss = creg_loss / coeffs_dict['id'].shape[0]

    # gamma regularization to ensure a nearly-monochromatic light
    gamma = coeffs_dict['gamma']
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    gamma_loss = torch.mean((gamma - gamma_mean) ** 2)

    return creg_loss, gamma_loss


def weight_box_loss(spec_weight):
    return torch.mean(-1 * (spec_weight < 0).float() * spec_weight)


def basis_norm_loss(basis, basis_old, weight):
    '''
    basis: [b,v,c]
    '''
    basis_norm = torch.sum(basis ** 2, dim=(1, 2))
    basis_norm_old = torch.sum(basis_old ** 2, dim=(1, 2))
    basis_norm_loss = ((basis_norm - basis_norm_old) * weight).abs().mean()

    # basis_one = basis / (basis_norm[..., None, None].detach().sqrt())
    # basis_one_old = basis_old / (basis_norm_old[..., None, None].detach().sqrt())
    # odot = torch.sum(basis_one * basis_one_old, dim=(1, 2))
    # basis_para_loss = ((odot - torch.ones_like(odot)) * weight).abs().mean()
    return basis_norm_loss


def basis_ortho_loss(basis):
    corr = torch.sum(basis * basis.unsqueeze(1), dim=(-1, -2))  # [n,n]
    identity = torch.eye(corr.shape[0]).to(corr.device)
    basis_ortho_loss = (corr - identity).abs().mean()
    return basis_ortho_loss


def model_reg_loss(facemodel, opt):
    if not opt.update_model:
        return 0., 0.
    diff_basis_old = facemodel.oldDiffuseAlbedoPca
    diff_basis = facemodel.diffuseAlbedoPca
    diff_weight = facemodel.diffuseAlbedoPcaVar.sqrt()
    diff_weight = diff_weight / torch.mean(diff_weight)
    spec_basis_old = facemodel.oldspecularAlbedoPca
    spec_basis = facemodel.specularAlbedoPca
    spec_weight = facemodel.specularWeightPcaVar.sqrt()
    spec_weight = spec_weight / torch.mean(spec_weight)
    norm_loss = basis_norm_loss(diff_basis, diff_basis_old, diff_weight) + \
        basis_norm_loss(spec_basis, spec_basis_old, spec_weight)
    diff_ortho_loss = basis_ortho_loss(diff_basis)
    return norm_loss, diff_ortho_loss
