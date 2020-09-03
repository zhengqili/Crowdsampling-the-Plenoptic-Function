import argparse
import os
from util import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
 
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--name', type=str, default='test_local', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--align_data', action='store_true',
                                help='if True, the datasets are loaded from "test" and "train" directories and the data pairs are aligned')

        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')

        self.parser.add_argument('--max_depth', type=float, required=True, help='models are saved here')
        self.parser.add_argument('--min_depth', type=float, default=1, help='models are saved here')
        self.parser.add_argument('--num_mpi_planes', type=int, default=64, help='models are saved here')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        self.parser.add_argument('--mpi_w', type=int, default=768, help='instance normalization or batch normalization')
        self.parser.add_argument('--mpi_h', type=int, default=768, help='instance normalization or batch normalization')
        self.parser.add_argument('--num_mpi_f', type=int, default=8, help='instance normalization or batch normalization')
        self.parser.add_argument('--num_latent_f', type=int, default=16, help='number of latent code')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal | xavier | kaiming | orthogonal]')

        # encoder config
        self.parser.add_argument('--nef', type=int, default=64, help='# of gen filters in encoder')
        self.parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        self.parser.add_argument('--nl', type=str, default='lrelu', help='non-linearity activation: relu | lrelu | elu')

        self.parser.add_argument('--norm_E', type=str, default='none', help='instance|layer|group|pixel| normalization')
        self.parser.add_argument('--norm_G', type=str, default='pixel', help='instance normalization or layer normalization or group normalization or pixel normalization')

        self.parser.add_argument('--ref_fov', type=float, required=True, help='reference field of view in degree')

        self.parser.add_argument('--model_E', type=str, default='munit_encoder', help='munit_encoder|spade_encoder')

        self.parser.add_argument('--where_add', type=str, default='adain', help='input|all|middle|shallow|adain; where to add z in the network G')
        self.parser.add_argument('--upsample', type=str, default='bilinear', help='basic | bilinear')

        self.parser.add_argument('--dataset', type=str, required=True, help='Dataset for training')
        self.parser.add_argument('--warp_src_img', type=int, default=1, help='warp source img')
        self.parser.add_argument('--log_comment', type=str, default='feature_mpi_exp_independent_warp', help='tensorboard log dir comment')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
