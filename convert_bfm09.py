import os
import argparse
from scipy.io import loadmat
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--bfm_folder', type=str, default="BFM")
parser.add_argument('--save_path', type=str, default="BFM/morphableModel-2009.pkl")
args = parser.parse_args()


bfm_path = os.path.join(args.bfm_folder, "BFM_model_front.mat")
model = loadmat(bfm_path)

# my_bfm = torch.load("BFM/morphableModel-2009.pkl")
# shapeMean: [v,3]
# shapePca: [80,v,3]
# expressionPca: [64,v,3]
# faces: [f,3]
# point_buf: [v,8]
# landmarksAssociation: [68]

mean_shape = model['meanshape'].astype(np.float32)
mean_shape = torch.from_numpy(mean_shape).reshape(-1, 3)
# print(torch.max((mean_shape - my_bfm["shapeMean"]).abs()))

id_base = model['idBase'].astype(np.float32)
id_base = torch.from_numpy(id_base).reshape(-1, 3, 80).permute(2, 0, 1)
# print(torch.max((id_base - my_bfm["shapePca"]).abs()))

exp_base = model['exBase'].astype(np.float32)
exp_base = torch.from_numpy(exp_base).reshape(-1, 3, 64).permute(2, 0, 1)
# print(torch.max((exp_base - my_bfm["expressionPca"]).abs()))

point_buf = model['point_buf'].astype(np.int64) - 1
point_buf = torch.from_numpy(point_buf)
# print(torch.max((point_buf - my_bfm["point_buf"]).abs()))

face_buf = model['tri'].astype(np.int64) - 1
face_buf = torch.from_numpy(face_buf)
# print(torch.max((face_buf - my_bfm["faces"]).abs()))

keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1
keypoints = torch.from_numpy(keypoints)
# print(torch.max((keypoints - my_bfm["landmarksAssociation"]).abs()))

my_bfm = {
    "shapeMean": mean_shape,
    "shapePca": id_base,
    "expressionPca": exp_base,
    "faces": face_buf,
    "point_buf": point_buf,
    "landmarksAssociation": keypoints,
}

torch.save(my_bfm, args.save_path)
