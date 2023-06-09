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
        parser.add_argument('--img_folder', type=str, default='dataset/examples', help='folder for test images.')
        parser.add_argument('--save_folder', type=str, default='workspace/examples', help='folder for save results.')
        parser.add_argument('--dataset_mode', type=str, default='refmm', help='folder for test images.')

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
