import os.path
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import util.util as util
import numpy as np
import torch
from util.preprocess import align_img, estimate_norm
from util.load_mats import load_lm3d
import glob


def parse_label(label):
    return torch.tensor(np.array(label).astype(np.float32))


class RefMMDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.lm3d_std = load_lm3d(opt.bfm_folder)
        self.opt = opt
        self.name = 'train' if opt.isTrain else 'val'
        self.coeff_name = "ffhq-coeffs-%s-refine" % self.name
        self.img_paths = glob.glob(os.path.join(opt.data_root, self.name, "*.jpg"))
        self.img_paths += glob.glob(os.path.join(opt.data_root, self.name, "*.png"))
        self.size = len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(img_path)
        img_name_woext = os.path.splitext(img_name)[0]
        coeff_path = os.path.join(self.opt.data_root, self.coeff_name, "%s.pkl" % img_name_woext)
        msk_path = os.path.join(img_dir, "mask", img_name)
        lm_path = os.path.join(img_dir, "landmarks", "%s.txt" % img_name_woext)

        raw_img = Image.open(img_path).convert('RGB')
        raw_msk = Image.open(msk_path).convert('RGB')
        raw_lm = np.loadtxt(lm_path).astype(np.float32)
        coeff = torch.load(coeff_path, map_location="cpu")
        geo_dict = {
            "id": coeff["id"], "exp": coeff["exp"], "angle": coeff["angle"], "trans": coeff["trans"],
        }

        _, img, lm, msk = align_img(raw_img, raw_lm, self.lm3d_std, raw_msk)
        
        _, H = img.size
        M = estimate_norm(lm, H)
        transform = get_transform()
        img_tensor = transform(img)
        msk_tensor = transform(msk)[:1, ...]
        lm_tensor = parse_label(lm)
        M_tensor = parse_label(M)

        return {'imgs': img_tensor, 
                'lms': lm_tensor, 
                'msks': msk_tensor, 
                'M': M_tensor,
                'im_paths': img_path, 
                'dataset': self.name,
                "geo_dict": geo_dict,
                }
    
    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.size
    