import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout


def set_network(opt, ctx, istest):
    depth = opt.depth
    if istest:
        lr = 0
        beta1 = 0
    else:
        lr = opt.lr
        beta1 = opt.beta1
    ndf = opt.ndf
    ngf = opt.ngf
    latent = opt.latent
    print(latent)
    append = opt.append
    netD = None
    netD2 = None
    netDS = None
    trainerD = None
    trainerDS = None
    trainerD2 = None
    # load networks based on opt.ntype (1 - AE 2 - ALOCC 3 - latentD 4 - adnov)
    if append:
        if opt.ntype > 1:
            netD = Discriminator(in_channels=6, n_layers=2, ndf=ndf, istest=istest)
            network_init(netD, ctx=ctx)
            trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
        if opt.ntype > 2:
            netD2 = LatentDiscriminator(in_channels=6, n_layers=2, ndf=ndf, istest=istest)
            network_init(netD2, ctx=ctx)
            trainerD2 = gluon.Trainer(netD2.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
        if opt.ntype > 3:
            netDS = models.Discriminator(in_channels=6, n_layers=2, ndf=ngf)
            network_init(netDS, ctx=ctx)
            trainerDS = gluon.Trainer(netDS.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    else:
        if opt.ntype > 1:
            netD = Discriminator(in_channels=3, n_layers=2, ndf=ndf, istest=istest)
            network_init(netD, ctx=ctx)
            trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
        if opt.ntype > 2:
            netD2 = LatentDiscriminator(in_channels=3, n_layers=2, ndf=ndf, istest=istest)
            network_init(netD2, ctx=ctx)
            trainerD2 = gluon.Trainer(netD2.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
        if opt.ntype > 3:
            netDS = Discriminator(in_channels=3, n_layers=2, ndf=ngf)
            network_init(netDS, ctx=ctx)
            trainerDS = gluon.Trainer(netDS.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    netEn = Encoder(in_channels=3, n_layers=depth, latent=latent, ndf=ngf, istest=istest)
    netDe = Decoder(in_channels=3, n_layers=depth, latent=latent, ndf=ngf, istest=istest)
    network_init(netEn, ctx=ctx)
    trainerEn = gluon.Trainer(netEn.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    network_init(netDe, ctx=ctx)
    trainerDe = gluon.Trainer(netDe.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    return netEn, netDe, netD, netD2, netDS, trainerEn, trainerDe, trainerD, trainerD2, trainerDS


# Define Unet generator skip block
class UnetSkipUnit(HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False,
                 use_dropout=False, use_bias=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            en_conv = Conv2D(channels=inner_channels, kernel_size=5, strides=2, padding=0,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)
            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)

            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=5, strides=2, padding=0,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + decoder
            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=5, strides=2, padding=0,
                                          in_channels=inner_channels)
                encoder = [en_conv]
                decoder = [de_relu, de_conv, Activation(activation='tanh')]
                model = encoder + [inner_block] + decoder
            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=5, strides=2, padding=0,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + [inner_block] + decoder
            if use_dropout:
                model += [Dropout(rate=0.5)]

            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return self.model(x)


# Define Unet generator
class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=True):
        super(UnetGenerator, self).__init__()

        # Build unet generator structure
        unet = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet = UnetSkipUnit(ngf * 8, ngf * 8, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf * 8, ngf * 4, unet)
        unet = UnetSkipUnit(ngf * 4, ngf * 2, unet)
        unet = UnetSkipUnit(ngf * 2, ngf * 1, unet)
        unet = UnetSkipUnit(ngf, in_channels, unet, outermost=True)

        with self.name_scope():
            self.model = unet

    def hybrid_forward(self, F, x):
        print(np.shape(self.model(x)))
        return self.model(x)


# Define the PatchGAN discriminator
class Discriminator(HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False, istest = False, isthreeway = False):
        super(Discriminator, self).__init__()

        with self.name_scope():
            self.model = HybridSequential()
            kernel_size = 5
            padding = 0 #int(np.ceil((kernel_size - 1) / 2))
            self.model.add(Conv2D(channels=ndf, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=in_channels))
            self.model.add(LeakyReLU(alpha=0.2))

            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=2,
                                      padding=padding, in_channels=ndf * nf_mult_prev,
                                      use_bias=use_bias))
                # self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult ))
                self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult ))
                self.model.add(LeakyReLU(alpha=0.2))

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=ndf * nf_mult_prev,
                                  use_bias=use_bias))
            # self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
            self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
            self.model.add(LeakyReLU(alpha=0.2))
            self.model.add(Conv2D(channels=1, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=ndf * nf_mult))
            if isthreeway:
                self.model.add(gluon.nn.Dense(3))
            elif use_sigmoid:
                self.model.add(Activation(activation='sigmoid'))



    def hybrid_forward(self, F, x):
        out = self.model(x)
        #print(np.shape(out))
        return out

class LatentDiscriminator(HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False, istest = False, isthreeway = False):
        super(LatentDiscriminator, self).__init__()

        with self.name_scope():
            self.model = HybridSequential()
            self.model.add(gluon.nn.Dense(128))
            self.model.add(Activation(activation='relu'))
            self.model.add(gluon.nn.Dense(64))
            self.model.add(Activation(activation='relu'))
            self.model.add(gluon.nn.Dense(32))
            self.model.add(Activation(activation='relu'))
            self.model.add(gluon.nn.Dense(16))
            self.model.add(Activation(activation='sigmoid'))
    def hybrid_forward(self, F, x):
        out = self.model(x)
        return out


class Encoder(HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_bias=False, istest=False,latent=256, usetanh = False ):
            super(Encoder, self).__init__()
            self.model = HybridSequential()
            kernel_size = 5
            padding = 0 #int(np.ceil((kernel_size - 1) / 2))
            self.model.add(Conv2D(channels=ndf, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=in_channels))
            self.model.add(LeakyReLU(alpha=0.2))
            nf_mult = 2
            nf_mult_prev = 1

            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = 2 ** n
                self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=2,
                                      padding=padding, in_channels=ndf * nf_mult_prev,
                                      use_bias=use_bias))
                self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
                self.model.add(LeakyReLU(alpha=0.2))

            nf_mult_prev = nf_mult
            nf_mult = 2 ** n_layers
            self.model.add(Conv2D(channels=latent, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=ndf * nf_mult_prev,
                                  use_bias=use_bias))
            #self.model.add(BatchNorm(momentum=0.1, in_channels =latent))
            if usetanh:
                self.model.add(Activation(activation='tanh'))
            else:
                self.model.add(LeakyReLU(alpha=0.2))



    def hybrid_forward(self, F, x):
        out = self.model(x)
        # print(out)
        return out


class Decoder(HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_bias=False, istest=False, latent=256, usetanh = False ):
            super(Decoder, self).__init__()
            self.model = HybridSequential()
            kernel_size = 5
            padding = 0
            nf_mult = 2 ** n_layers
            self.model.add(Conv2DTranspose(channels=ndf * nf_mult/2, kernel_size=kernel_size, strides=2,
                                           padding=padding, in_channels=latent,
                                           use_bias=use_bias))
            self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult / 2))
            self.model.add(Activation(activation='relu'))
            for n in range(1, n_layers):
                nf_mult = nf_mult / 2
                self.model.add(Conv2DTranspose(channels=ndf * nf_mult / 2, kernel_size=kernel_size, strides=2,
                                               padding=padding, in_channels=ndf * nf_mult,
                                               use_bias=use_bias))
                self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult / 2))
                #self.model.add(LeakyReLU(alpha=0.2))
                if n==2:
                      self.model.add(Dropout(rate=0.5))
                self.model.add(Activation(activation='relu'))
            self.model.add(Conv2DTranspose(channels=in_channels, kernel_size=kernel_size, strides=2,
                                           padding=padding, in_channels=ndf))
            self.model.add(Activation(activation='tanh'))

    def hybrid_forward(self, F, x):
        out = self.model(x)
        # print(out)
        return out


def param_init(param, ctx):
    if param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
    elif param.name.find('batchnorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx)
        # Initialize gamma from normal distribution with mean 1 and std 0.02
        if param.name.find('gamma') != -1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape))
    elif param.name.find('dense') != -1:
        param.initialize(init=mx.init.Normal(0.02), ctx=ctx)


def network_init(net, ctx):
    for param in net.collect_params().values():
        param_init(param, ctx)
