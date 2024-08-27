from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument("--image", type=str, default='/home/cimda/zelinli/NucGAN/GAN/Data_folder/test/images/1_4.nii')
        # parser.add_argument("--result", type=str, default='/home/cimda/zelinli/NucGAN/GAN/Data_folder/gen/result1_4.nii', help='path to the .nii result to save')
        # parser.add_argument('--patch_size', default=[128, 128, 64], help='Size of the patches extracted from the image')

        parser.add_argument('--phase', type=str, default='test', help='test')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument("--stride_inplane", type=int, nargs=1, default=32, help="Stride size in 2D plane")
        parser.add_argument("--stride_layer", type=int, nargs=1, default=32, help="Stride size in z direction")
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser