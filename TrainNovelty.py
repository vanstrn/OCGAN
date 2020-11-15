from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np
import random
from random import shuffle
import dataloaderiter as dload
import load_image
import visual
import models
from datetime import datetime
import time
import logging
import argparse
import models
import options
import andgan

from sklearn.metrics import roc_curve, auc

def plotloss(loss_vec, fname):
    plt.gcf().clear()
    plt.plot(loss_vec[0], label="Dr", alpha = 0.7)
    plt.plot(loss_vec[4], label="Dl", alpha = 0.7)
    plt.plot(loss_vec[1], label="G", alpha=0.7)
    plt.plot(loss_vec[2], label="R", alpha= 0.7)
    plt.plot(loss_vec[3], label="Acc", alpha = 0.7)
    plt.legend()
    plt.savefig(fname)


def main(opt):
    if opt.seed != -1:
        random.seed(opt.seed)
    ctx = mx.gpu(opt.gpu_num) if opt.use_gpu else mx.cpu()
    print(opt)
    #Modifying the loading to
    inclasspaths , inclasses = dload.loadPaths(opt)
    train_data, val_data,datasize = load_image.load_image(inclasspaths, opt)
    print('Data loading done.')
    networks = models.set_network(opt, ctx, False)
    print('training')
    # train networks based on opt.ntype(1 - AE 2 - ALOCC 3 - latentD  4 - adnov)
    if opt.ntype == 4:
        loss_vec = andgan.trainadnov(opt, train_data, val_data, ctx, networks,datasize)
    elif opt.ntype == 2:
        loss_vec = andgan.traincvpr18(opt, train_data, val_data, ctx, networks,datasize)
    elif opt.ntype == 1:
        loss_vec = andgan.trainAE(opt, train_data, val_data, ctx, networks,datasize)
    plotloss(loss_vec, 'outputs/'+opt.expname+'_loss.png')
    return inclasses



def mainEvaluation(opt):
    ctx = mx.gpu() if opt.use_gpu else mx.cpu()
    testclasspaths = []
    testclasslabels = []
    print('loading test files')
    filename = '_testlist.txt'
    with open(opt.dataset+"_"+opt.expname+filename , 'r') as f:
        for line in f:
            testclasspaths.append(line.split(' ')[0])
            if int(line.split(' ')[1]) == -1:
                testclasslabels.append(0)
            else:
                testclasslabels.append(1)
    neworder = range(len(testclasslabels))
    c = list(zip(testclasslabels, testclasspaths))
    print('shuffling')
    random.shuffle(c)
    testclasslabels, testclasspaths = zip(*c)
    print('loading pictures')
    test_data = load_image.load_test_images(testclasspaths,testclasslabels,opt.batch_size, opt.img_wd, opt.img_ht, ctx, opt.noisevar)
    print('picture loading done')
    opt.istest = True
    networks = models.set_network(opt, ctx, True)
    netEn = networks[0]
    netDe = networks[1]
    netD = networks[2]
    netD2 = networks[3]
    # load_epoch = opt.epochs - 1
    # netEn.load_params('checkpoints/'+opt.expname+'_'+str(load_epoch)+'_En.params', ctx=ctx)
    # netDe.load_params('checkpoints/'+opt.expname+'_'+str(load_epoch)+'_De.params', ctx=ctx)
    # if opt.ntype>1:
    # 	netD.load_params('checkpoints/'+opt.expname+'_'+str(load_epoch)+'_D.params', ctx=ctx)
    # if opt.ntype>2:
	# netD2.load_params('checkpoints/'+opt.expname+'_'+str(load_epoch)+'_D2.params', ctx=ctx)

    print('Model loading done')
    lbllist = [];
    scorelist1 = [];
    scorelist2 = [];
    scorelist3 = [];
    scorelist4 = [];
    test_data.reset()
    count = 0

    for batch in (test_data):
	count = count+1
        output1=np.zeros(opt.batch_size)
        output2=np.zeros(opt.batch_size)
        output3=np.zeros(opt.batch_size)
        output4=np.zeros(opt.batch_size)
        real_in = batch.data[0].as_in_context(ctx)
        real_out = batch.data[1].as_in_context(ctx)
        lbls = batch.label[0].as_in_context(ctx)
        outnn = (netDe(netEn((real_in))))
	out = outnn
        output3 = -1*nd.mean((outnn - real_out)**2, (1, 3, 2)).asnumpy()
        if opt.ntype >1: #AE
        	out_concat = nd.concat(real_in, outnn, dim=1) if opt.append else outnn
        	output1 = nd.mean((netD(out_concat)), (1, 3, 2)).asnumpy()
        	out_concat = nd.concat(real_in, real_in, dim=1) if opt.append else real_in
        	output2 = netD((out_concat))  # Image with no noise
        	output2 = nd.mean(output2, (1,3,2)).asnumpy()
        	out = netDe(netEn(real_out))
        	out_concat =  nd.concat(real_in, out, dim=1) if opt.append else out
        	output = netD(out_concat) #Denoised image
        	output4 = nd.mean(output, (1, 3, 2)).asnumpy()
        lbllist = lbllist+list(lbls.asnumpy())
        scorelist1 = scorelist1+list(output1)
        scorelist2 = scorelist2+list(output2)
        scorelist3 = scorelist3+list(output3)
        scorelist4 = scorelist4+list(output4)
        out = netDe(netEn(real_in))

        # Save some sample results
        fake_img1 = nd.concat(real_in[0],real_out[0], out[0], outnn[0],dim=1)
        fake_img2 = nd.concat(real_in[1],real_out[1], out[1],outnn[1], dim=1)
        fake_img3 = nd.concat(real_in[2],real_out[2], out[2], outnn[2], dim=1)
        fake_img4 = nd.concat(real_in[3],real_out[3],out[3],outnn[3], dim=1)
        fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img4, dim=2)
        visual.visualize(fake_img)
        plt.savefig('outputs/T_'+opt.expname+'_'+str(count)+'.png')

    print("Positives" + str(np.sum(lbllist)))
    print("Negatives" + str(np.shape(lbllist)-np.sum(lbllist) ))
    fpr, tpr, _ = roc_curve(lbllist, scorelist3, 1)
    roc_auc1 = 0
    roc_auc2 = 0
    roc_auc4 = 0
    roc_auc3 = auc(fpr, tpr)
    if int(opt.ntype) >1: #AE
	    fpr, tpr, _ = roc_curve(lbllist, scorelist1, 1)
	    roc_auc1 = auc(fpr, tpr)
	    fpr, tpr, _ = roc_curve(lbllist, scorelist2, 1)
	    roc_auc2 = auc(fpr, tpr)
    	    fpr, tpr, _ = roc_curve(lbllist, scorelist4, 1)
	    roc_auc4 = auc(fpr, tpr)

    return[roc_auc1, roc_auc2, roc_auc3, roc_auc4]



if __name__ == "__main__":
    opt = options.train_options()
    inclasses = main(opt)
    # opt = options.test_options()
    roc_auc =mainEvaluation(opt)
    print(roc_auc)
    with open(opt.expname+".txt", 'w') as f:
        for s in roc_auc:
            f.write(str(s) + '\n')
