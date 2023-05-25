import os
import argparse
import torch


parser = argparse.ArgumentParser(description='convert ckpt to finetuned model')
parser.add_argument('--ckpt_path', type=str, default="checkpoints/0521_reimp_finetune/epoch_latest.pth")
parser.add_argument('--initial_refmm_path', type=str, default="RefMM/multipie_initial_refmm_model.pkl")
parser.add_argument('--refmm_save_path', type=str, default="RefMM/finetuned_refmm_model.pkl")
args = parser.parse_args()


init_refmm = torch.load(args.initial_refmm_path, map_location="cpu")
finetune_refmm = torch.load(args.ckpt_path, map_location="cpu")["facemodel"]

ft_diff_basis = finetune_refmm["diffuseAlbedoPca"]  # [n,v,3]
ft_diff_mean = finetune_refmm["diffuseAlbedoMean"]  # [v,3]
ft_spec_basis = finetune_refmm["specularAlbedoPca"]  # [n,v,1]
ft_spec_mean = finetune_refmm["specularAlbedoMean"]  # [v,1]

num_basis = ft_diff_basis.shape[0]

diff_var = init_refmm["diff_var"][:num_basis, None, None]  # [n,1,1]
spec_var = init_refmm["spec_var"][:num_basis, None, None]  # [n,1,1]
spec_basis = init_refmm["spec_basis"][:num_basis]  # [n,v,1+k]
spec_mean = init_refmm["spec_mean"]  # [v,1+k]

ft_diff_basis = ft_diff_basis / diff_var.sqrt()
ft_spec_basis = ft_spec_basis / spec_var.sqrt()
ft_spec_basis = torch.cat([ft_spec_basis, spec_basis[..., 1:]], dim=-1)
ft_spec_mean = torch.cat([ft_spec_mean, spec_mean[..., 1:]], dim=-1)

model = init_refmm
model.update({
    "diff_basis": ft_diff_basis,
    "spec_basis": ft_spec_basis,
    "diff_mean": ft_diff_mean,
    "spec_mean": ft_spec_mean,
})
torch.save(model, args.refmm_save_path)
