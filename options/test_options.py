"""This script contains the test options for Deep3DFaceRecon_pytorch
"""

from .base_options import BaseOptions
from util import util


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
        parser.add_argument('--img_folder', type=str, default='datasets/examples', help='folder for test images.')
        parser.add_argument('--skip_step', type=int, default=10, help='test step size')
        parser.add_argument('--save_coarse_mesh', type=util.str2bool, nargs='?', const=True, default=False, help='save coarse mesh or not')
        parser.add_argument('--save_mesh', type=util.str2bool, nargs='?', const=True, default=False, help='save mesh or not')
        parser.add_argument('--save_visual', type=util.str2bool, nargs='?', const=True, default=True, help='save mesh or not')
        parser.add_argument('--save_zip', type=util.str2bool, nargs='?', const=True, default=True, help='save zip or not')
        parser.add_argument('--save_coeff', type=util.str2bool, nargs='?', const=True, default=False, help='save zip or not')
        parser.add_argument('--save_coeff_dir', type=str, default='')
        parser.add_argument('--NoW_save_root', type=str, default="NoW", help='save root for NoW benchmark results')

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
