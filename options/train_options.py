from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_decay_epoch', type=int, default=100, help='# of epoch to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')

        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--g_lr', type=float, default=0.0004, help='initial learning rate for adam')
        self.parser.add_argument('--d_lr', type=float, default=0.00001, help='initial learning rate for adam')

        self.parser.add_argument('--use_gan', type=int, default=1, help='use_gan')
        self.parser.add_argument('--use_vgg_loss', type=int, default=1, help='use_vgg_loss')
        self.parser.add_argument('--w_grad', type=float, default=0., help='w_grad')
        self.parser.add_argument('--w_l1', type=float, default=1., help='w_l1')
        self.parser.add_argument('--w_gan', type=float, default=0.02, help='w_gan')
        self.parser.add_argument('--recon_x_w', type=float, default=1., help='w_gan')
        self.parser.add_argument('--style_w', type=float, default=5.0, help='w_gan')
        self.parser.add_argument('--recon_s_w', type=float, default=0.1, help='w_gan')

        # self.parser.add_argument('--use_graphite', type=int, required=True, help='use_graphite')

        self.isTrain = True
