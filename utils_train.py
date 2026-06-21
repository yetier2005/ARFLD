import time
import numpy as np
import torch
from torch.functional import norm
import torch.nn as nn
import logging
import copy

from torch.nn.parallel.data_parallel import DataParallel
from torchvision.transforms.transforms import ColorJitter
import optimizer_cust

from dataset.dataset import *
from dataset.autoaug import *
from models import *
from utils import *


def random_perturb(net, new=None):
    if new is None:
        for p in net.parameters():
            gauss = torch.normal(mean=torch.zeros_like(p), std=1)
            if p.grad is None:
                p.grad = gauss
            else:
                p.grad.data.copy_(gauss.data)
        
        norm = torch.norm(
                torch.stack([(p.grad).norm(p=2) for p in net.parameters() if p.grad is not None]),
                p=2)
        # clip_coef = max_norm / (total_norm + 1e-6)

        with torch.no_grad():
            scale = 5.0 / (norm + 1e-12)
            scale = torch.clamp(scale, max=1.0)
            for p in net.parameters():
                if p.grad is None: continue
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)
    
    else:
        norm = torch.norm(
                torch.stack([(q.data).norm(p=2) for q in new.parameters()]), p=2)
        with torch.no_grad():
            # scale = 1 / (norm + 1e-12)
            # scale = 1.0
            for p, q in zip(net.parameters(), new.parameters()):
                # e_w = 1.0 * q.data * scale.to(q)
                e_w = 1.0 * q.data
                p.add_(e_w)

    net.zero_grad()
    return net
    

def epoch_theta_match(mode, dataloader, net, optimizer, criterion, args, aug, d_thetas=None):
    assert args is not None
    """ customized, training lower level on real images for one epoch """
    ## updates != None: match gradient norm as well
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.cuda()
    criterion = criterion.cuda()


    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        ## data
        img = datum[0].float().cuda()
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].long().cuda()
        n_b = lab.shape[0]

        ## closure
        if args.opt_net in ['sam', 'asam', 'sam-rand', 'asam-rand']:
            def closure():
                loss = criterion(net(img), lab); loss.backward(); return loss
        elif args.opt_net == 'sgd':
            closure = None

        ## step
        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            if d_thetas is None:
                optimizer.step(closure=closure)
            else:
                optimizer.step(closure=closure, d_ps_anchor=d_thetas[i_batch])

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def epoch(mode, dataloader, net, optimizer, criterion, args, aug, normalize='none', get_d_thetas=False, grad=None, it_eval=None, net_copy=None):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.cuda()
    criterion = criterion.cuda()

    if mode == 'train':
        net.train()
    else:
        net.eval()

    d_thetas = [] if get_d_thetas else None # return a list of updates
    for i_batch, datum in enumerate(dataloader):
        ## data
        img = datum[0].float().cuda()
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            elif args.aa: # aa transform already added to the dataset
                img = img
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1]
        if not lab.is_cuda: lab = lab.cuda()
        n_b = lab.shape[0]

        # if mode == 'train':
        #     weight = datum[2]
        # else:
        #     weight = torch.ones(n_b, ).cuda()
        
        ## closure
        if args.opt_net in ['sam', 'asam', 'sam-rand', 'asam-rand']:
            def closure():
                loss = criterion(net(img), lab); loss.backward(); return loss
        elif args.opt_net == 'sgd':
            closure = None

        ## step
        output = net(img)
        loss = criterion(output, lab, weight=None)
        # print(loss)
        # loss = criterion(output, lab)

        if len(lab.shape) > 1: # one
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.argmax(dim=-1).cpu().numpy()))
        else:
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            if grad is not None:
                logging.info('WARNING: COPYING GRADIENTS')
                for p, g in zip(net.parameters(), grad):
                    p.grad.data.copy_(g)
            if get_d_thetas:
                _, d_theta = optimizer.step(closure=closure, get_d_thetas=True)
                d_thetas.append(d_theta)
            else:
                optimizer.step(closure=closure)
            if it_eval < 0:
                with torch.no_grad():
                    norm = torch.norm(
                        torch.stack([(p-q).norm(p=2) for p, q in zip(net.parameters(), net_copy.parameters())]), p=2)
                    scale = 5.0 / (norm + 1e-12)
                    scale = torch.clamp(scale, max=1.0)
                    # print(norm)
                    for p, q in zip(net.parameters(), net_copy.parameters()):
                        e_w = (p-q) * scale.to(p) + q - p
                        p.add_(e_w)

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg, d_thetas


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, lr=None, verbose=True, fast=False, weight=None):
    ## get model and optimizer
    net = net.cuda()
    net_copy = copy.deepcopy(net)
    lr = float(args.lr_net) if lr == None else lr
    Epoch = int(args.epoch_eval_train)
    # optimizer = get_optimizer(net.parameters(), args.opt_net, lr=lr, weight_decay=0.0005, rho=args.rho, momentum=0.9)
    optimizer = get_optimizer(net.parameters(), args.opt_net, lr=lr, weight_decay=0.0005, rho=args.rho, momentum=0.9)
    lr_schedule = get_lr_schedule(optimizer, args.opt_net, lr=lr, Epoch=Epoch)
    criterion = cross_entropy_loss_cust(args).cuda() # softlabel

    ## get data
    transform = None
    images_train = images_train.cuda()
    labels_train = labels_train.cuda()
    dst_train = torch.utils.data.TensorDataset(images_train, labels_train)

    # dst_train = torch.utils.data.TensorDataset(images_train, labels_train, weight)

    # dst_train = torch.utils.data.TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    loss_test, acc_test, _ = epoch('test', testloader, net, optimizer, criterion, args, aug=False, normalize='none') # real data
    logging.info('%s Evaluate_%02d: epoch = %04d test acc = %.4f' % (get_time(), it_eval, 0, acc_test))
    ## train on syn set from scratch
    for ep in range(Epoch+1):
        loss_train, acc_train, _ = epoch('train', trainloader, net, optimizer, criterion, args, aug=(not args.no_aug), normalize=args.normalize_input, it_eval=it_eval, net_copy=net_copy) # no_aug mutes aug everywhere but gm
        lr_schedule(ep)

        ## test on full testset
        if (ep != 0 and ep % (Epoch // 2) == 0) or (args.match_mode == '1vN' and ep % 10 == 0):
            loss_test, acc_test, _ = epoch('test', testloader, net, optimizer, criterion, args, aug=False, normalize='none') # real data
            if verbose: logging.info('%s Evaluate_%02d: epoch = %04d train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, ep, loss_train, acc_train, acc_test))
            if fast: break

    if verbose: logging.info(f'Evaluate syns debug: loader_size: {len(testloader)}, batch_size: {testloader.batch_size}')
    return net, acc_train, acc_test


def evaluate_fullset(it_eval, net, loader, args, verbose=True):
    """ evaluate a TRAINED model on fullset """
    if loader is None:
        return -99
    net = net.cuda()
    criterion = cross_entropy_loss_cust(args).cuda()

    loss, acc, _ = epoch('test', loader, net, None, criterion, args, aug=False, normalize=args.normalize_input)
    if verbose:
        logging.info('%s Evaluate_%02d: fullset acc = %.4f' % (get_time(), it_eval, acc))
        logging.info(f'Evaluate full debug: loader_size: {len(loader)}, batch_size: {loader.batch_size}')

    return acc
