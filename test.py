"""This script is the inference script for Deep3DFaceRecon_pytorch
"""

import os
from options.test_options import TestOptions
from models import create_model
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from torchvision.utils import save_image


def get_data_path(root='examples'):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]
    return im_path, lm_path


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm


def main(rank, opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()

    im_path, lm_path = get_data_path(name)
    lm3d_std = load_lm3d(opt.bfm_folder) 

    os.makedirs(opt.save_folder, exist_ok=True)

    for i in range(len(im_path)):
        print(i, im_path[i])
        if not os.path.isfile(lm_path[i]):
            continue
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {'imgs': im_tensor}
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        save_image(model.output_vis, os.path.join(opt.save_folder, "%05d.png" % i))


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt,opt.img_folder)