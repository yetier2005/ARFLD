import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import shutil
import sys
import glob
import pickle
from pprint import pformat
from torch.nn.functional import binary_cross_entropy
from torchvision.utils import save_image
from tqdm import tqdm
from utils_train import *
from utils import *
from models import *
from dataset import *
from dataset.dataset_perlabel import *
# from cifar10_models.resnet import resnet18


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='FashionMNIST', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=3, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='initialization of synthetic data, noise/real: initialize from random noise or real images.')
    parser.add_argument('--init_path', type=str, default='', help='init from pretrained ckpts')
    parser.add_argument('--data_path', type=str, default='/home/ljq/zff/dataset', help='dataset path')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--gpu', type=str, default='auto', help='gpu ID(s)')
    parser.add_argument('--save', type=str, default='debug', help='saving directory / expid')
    parser.add_argument('--group', type=str, default='extreme', help='extra tag for exp groups, exps in the same group will be saved to the same folder [group]')
    parser.add_argument('--tag', type=str, default='test1', help='extra tag for expid')
    parser.add_argument('--num_users', type=int, default='10', help='# of users')
    parser.add_argument('--frac', type=float, default='1.0', help='the fraction of users')
    parser.add_argument('--alpha', type=float, default='0.5', help='the fraction of users')
    parser.add_argument('--epochs', type=int, default='20', help='# of epochs')
    parser.add_argument('--extreme', type=int, default=0, help='fast mode')

    #### dev
    parser.add_argument('--no_aug', type=int, default=0, help='mute data augmentation at lower level and evaluation time')
    parser.add_argument('--fast', action='store_true', default=False, help='fast mode')
    #### match length
    parser.add_argument('--inner_loop', type=int, default=-1, help='num iterations for upper level (gradient matching)')
    parser.add_argument('--outer_loop', type=int, default=-1, help='num iterationa for lower level (training on syn data)')
    #### match norm
    parser.add_argument('--match_norm', type=int, default=0, help='for theta matching, train real theta with normalized SGD')
    #### mode
    parser.add_argument('--match_mode', type=str, default='regular',
                        help='check if per-label matching is necessary')
    #### sam
    parser.add_argument('--rho', type=float, default=0.5, help="0.5 for sam and 0.05 for asam")
    parser.add_argument('--progress_perturb', type=int, default=0, help="gradually increase rho")
    parser.add_argument("--opt_X", default='sgd', type=str, choices=['sam', 'asam', 'sgd', 'sam-rand', 'asam-rand'],
                        help="optimizer for syn images (X)")
    parser.add_argument("--opt_net", default='sgd', type=str, choices=['sam', 'asam', 'sgd', 'sam-rand', 'asam-rand'],
                        help="optimizer for model weight w, used for both lower level and evaluation")
    parser.add_argument("--opt_perturb", default='none', type=str, choices=['none', 'sam', 'asam', 'sam-rand', 'asam-rand'],
                        help="optimizer for perturbing w during gradient matching")
    parser.add_argument("--weight_decay_net", default=0, type=float)
    #### dsa
    parser.add_argument('--method', type=str, default='DSA', choices=['DC', 'DSA'], help='DC/DSA')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--opt_net_mom', type=float, default=0, help='0 in DSA, 0.5 in DC')
    args = parser.parse_args()
    # For speeding up, we can decrease the Iteration and epoch_eval_train, which will not cause significant performance decrease.

    ## dummy
    args.normalize_input = 'none'

    #### env setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(pick_gpu_lowest_memory()) if args.gpu == 'auto' else args.gpu

    #### args augment
    outer_loop, inner_loop = get_loops(args.ipc)
    if args.outer_loop == -1: args.outer_loop = outer_loop
    if args.inner_loop == -1: args.inner_loop = inner_loop
    args.device = f'cuda' if torch.cuda.is_available() else 'cpu'
    if 'debug' in args.tag: args.group = 'debug'
    if 'dev'   in args.tag: args.group = 'dev'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    args.aa = False # dummy
    max_batch_real = args.batch_real

    ## output dir
    script_name = args.save
    exp_id = '{}'.format(script_name)
    exp_id += f'_[{args.model}]'
    exp_id += f'_[{args.dataset}]'
    exp_id += f'_[ipc-{args.ipc}]'
    # exp_id += f'_[loop={args.outer_loop}x{args.inner_loop}]'
    exp_id += f'_[{args.match_mode}]'
    exp_id += f'_[{args.method}]'
    exp_id += f'_[{args.alpha}]'
    if args.progress_perturb: rho_tag = f'{args.rho}up'
    else: rho_tag = f'{args.rho}'
    if args.init != "noise":  exp_id += f'_[init-{args.init}]'
    if args.opt_X != 'sgd': exp_id += f'_[X-{args.opt_X}]-{rho_tag}'
    if args.opt_net != 'sgd': exp_id += f'_[net-{args.opt_net}]'
    if args.opt_perturb != 'none': exp_id += f'_[gm-{args.opt_perturb}-{rho_tag}]'
    if args.batch_real != 256: exp_id += f'_[bsr={args.batch_real}]'
    if args.no_aug: exp_id += '[no_aug]'
    if args.method == 'DSA' and args.opt_net_mom != 0: exp_id += f'_[mom{args.opt_net_mom}]'
    if args.method == 'DC' and args.opt_net_mom != 0.5: exp_id += f'_[mom{args.opt_net_mom}]'
    if args.tag and args.tag != 'none': exp_id += f'_[tag-{args.tag}]'
    if args.tag == 'none': exp_id += f'_[num_users-{args.num_users}]_[frac-{args.frac}]_[extre-{args.extreme}]'
    if 'debug' in args.tag: exp_id = args.tag
    if args.group == 'none':
        args.save = os.path.join('/home/ljq/zff/feddm/experiments/', exp_id)
    else:
        args.save = os.path.join('/home/ljq/zff/feddm/experiments/', f'{args.group}/', exp_id)
    if not os.path.exists( args.save):
        os.makedirs(args.save)
    ## override path
    # if os.path.exists(args.save):
    #     if 'debug' in args.tag or input('Exp {} exists, override? [y/n]'.format(exp_id)) == 'y': shutil.rmtree(args.save)
    #     else: exit()
    # create_exp_dir(args.save, run_script='./exp_scripts/{}'.format(script_name + '.sh'))
    ## output files
    args.ckpt_path = os.path.join(args.save, 'ckpts')
    args.vis_path  = os.path.join(args.save, 'vis')
    os.makedirs(args.ckpt_path)
    os.makedirs(args.vis_path)
    # logging
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    log_file = 'log.txt'
    log_path = os.path.join(args.save, log_file)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, trainloader_full, testloader = get_dataset(args.dataset, args.data_path)
    args.num_classes = num_classes

    # dst_train, dst_val = torch.utils.data.random_split(dst_train, [40000, 10000], generator=torch.Generator().manual_seed(42))
    data_save = []
    user_states = {}
    # pretrain_model = resnet18(pretrained=True).cuda()

    #### run experiments
    exp_acc_list = []
    for exp in range(args.num_exp):
        # dict_users, dict_classes = partition(dst_train, args.num_users, args.alpha)
        dict_users, dict_classes = partition_extreme(dst_train, args.num_users)
        logging.info(dict_classes)
        perturb = args.opt_perturb in ['sam', 'asam', 'sam-rand', 'asam-rand']
        max_rho = args.rho
        criterion = cross_entropy_loss_cust(args).cuda() # soft-label
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

        for idx in range(args.num_users):
            data_idxs = dict_users[idx]
            classes = dict_classes[idx]
            sub_train = DatasetSplit(dst_train, data_idxs)
            #### organize the real dataset
            if args.dataset == 'ImageNet':
                dst_perlabel = PerLabelLargeDataset(sub_train, num_classes, channel, args)
                imnet_loader = dst_perlabel.loader
                imnet_iterator = iter(imnet_loader)
                imnet_init_images = dst_perlabel.get_init_images(ipc=1)
            else:
                dst_perlabel = PerLabelDatasetNonIID(sub_train, classes, channel, args)
                get_images = dst_perlabel.get_images


            #### initialize the synthetic data
            image_syn = torch.randn(size=(len(classes)*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)


            #### training initialize synthetic data
            user_states[idx] = LocalUser(dst_perlabel, image_syn)
            
        logging.info('%s training begins'%get_time())
        fed_accs = []
        global_model = get_network(args.model, channel, num_classes, im_size).cuda()
        best_acc_test = 0
        all_img_syn = []
        all_lbl_syn = []
        all_weight = []
        for curr_epoch in tqdm(range(args.epochs)):
            args.dsa = True if args.method == 'DSA' else False
            curr_img_syn = []
            curr_lbl_syn = []
            logging.info('\n================== Epoch %d =================='%curr_epoch)
            global_model.train()        
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            logging.info('\nChoosing users {}'.format(' '.join(map(str, idxs_users))))
            for idx in idxs_users: #
                ## Train synthetic data
                user = user_states[idx]
                classes = dict_classes[idx]
                get_images = user.dataset.get_images
                image_syn = torch.randn(size=(len(classes)*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
                # image_syn = user.image_syn
                if args.init == 'real':
                    logging.info('initialize synthetic data from random real images with pseudo labels')
                    if args.dataset == 'ImageNet':
                        image_syn.data.copy_(imnet_init_images.data)
                    else:
                        for i, c in enumerate(classes):
                            image_syn.data[i*args.ipc:(i+1)*args.ipc] = get_images(c, args.ipc, avg=False).detach().data

                elif args.init == 'pretrained':
                    logging.info('initialize synthetic data from pretrained images')
                    ckpt_exp_user_path = os.path.join(args.ckpt_path, 'exp_{}'.format(exp), 'user_{}'.format(idx))
                    if not os.path.exists(ckpt_exp_user_path):
                        os.makedirs(ckpt_exp_user_path)
                    data_path = os.path.join(ckpt_exp_user_path, 'run_%s_%s_%d.pt'%(args.dataset, args.model, curr_epoch-1))
                    syn_state = torch.load(data_path)
                    assert syn_state['data'][0].shape[0] == args.ipc * len(classes)
                    image_syn.data.copy_(syn_state['data'][0].to(args.device))
                    label_syn.data.copy_(syn_state['data'][1])
                else:
                    logging.info('initialize synthetic data from random noise for user %d'%idx)
                    img_real = user.dataset.get_random_images(args.batch_real).detach().data
                    image_syn.requires_grad_(False)
                    image_syn[:,0,:,:] = image_syn[:,0,:,:] / image_syn[:,0,:,:].abs().max() * img_real[:,0,:,:].abs().max()
                    image_syn.requires_grad_(True)


                image_syn.requires_grad_()
                optimizer_img = get_optimizer([image_syn, ], args.opt_X, lr=args.lr_img, weight_decay=0, rho=0, momentum=0.5)
                optimizer_img.zero_grad()
                
                for it in range(args.Iteration+1):
                    loss_avg = 0
                    if curr_epoch != 0:
                        net = random_perturb(copy.deepcopy(global_model))
                    else:
                        net = get_network(args.model, channel, num_classes, im_size).cuda() # get a random model
                    
                    net.train()
                    for param in list(net.parameters()):
                        param.requires_grad = False
                    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed

                    new_net = get_network(args.model, channel, num_classes, im_size).cuda()
                    new_net.train()
                    for param in list(new_net.parameters()):
                        param.requires_grad = False
                    embed_new = new_net.module.embed if torch.cuda.device_count() > 1 else new_net.embed

                    BN_flag = False
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name(): #BatchNorm
                            BN_flag = True

                    ## update synthetic data
                    if not BN_flag:
                        loss = torch.tensor(0.0).cuda()
                        labs_syn = torch.LongTensor([]).cuda()
                        real_feats = []
                        syn_feats = []
                        real_feats_new = []
                        syn_feats_new = []
                        for i, c in enumerate(classes):
                            img_real = get_images(c, args.batch_real)
                            img_syn = image_syn[i*args.ipc:(i+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                            lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                            if args.dsa:
                                seed = int(time.time() * 1000) % 100000
                                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                                img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                            output_real = embed(img_real).detach()
                            output_syn = embed(img_syn)


                            labs_syn  = torch.cat([labs_syn, lab_syn], dim=0)

                            loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                            if curr_epoch > 0:
                                output_real_new = embed_new(img_real).detach()
                                output_syn_new = embed_new(img_syn)

                                loss += torch.sum((torch.mean(output_real_new, dim=0) - torch.mean(output_syn_new, dim=0))**2)


                    else: # for ConvNetBN
                        images_real_all = []
                        images_syn_all = []
                        loss = torch.tensor(0.0).to(args.device)
                        for i, c in enumerate(classes):
                            img_real = get_images(c, args.batch_real)
                            img_syn = image_syn[i*args.ipc:(i+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                            if args.dsa:
                                seed = int(time.time() * 1000) % 100000
                                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                                img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                            images_real_all.append(img_real)
                            images_syn_all.append(img_syn)

                        images_real_all = torch.cat(images_real_all, dim=0)
                        images_syn_all = torch.cat(images_syn_all, dim=0)

                        output_real = embed(images_real_all).detach()
                        output_syn = embed(images_syn_all)

                        loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)
                    
                    optimizer_img.zero_grad()
                    loss.backward()


                    optimizer_img.step()
                    loss_avg += loss.item()


                    loss_avg /= len(classes)
                    if it % 500 == 0 or it == args.Iteration:
                        logging.info('%s user %d loss = %.4f at iteration %d' % (get_time(), idx, loss_avg, it))

                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(labs_syn.detach())
                curr_img_syn.append(image_syn_train)
                curr_lbl_syn.append(label_syn_train)

                # ## visualize and save
                # exp_user_path = os.path.join(args.vis_path, 'exp_{}'.format(exp), 'user_{}'.format(idx))
                # if not os.path.exists(exp_user_path):
                #     os.makedirs(exp_user_path)
                # save_name = os.path.join(exp_user_path, 'vis_%s_%s_%dipc_epoch_%d.png'%(args.dataset, args.model, args.ipc, curr_epoch))
                # image_syn_vis = copy.deepcopy(image_syn_train.detach().cpu())
                # for ch in range(channel):
                #     image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                # image_syn_vis[image_syn_vis<0] = 0.0
                # image_syn_vis[image_syn_vis>1] = 1.0
                # save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
                

                ckpt_exp_user_path = os.path.join(args.ckpt_path, 'exp_{}'.format(exp), 'user_{}'.format(idx))
                if not os.path.exists(ckpt_exp_user_path):
                    os.makedirs(ckpt_exp_user_path)
                data_save = [copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(labs_syn.detach().cpu())]
                torch.save({'data': data_save}, os.path.join(ckpt_exp_user_path, 'run_%s_%s_%d.pt'%(args.dataset, args.model, curr_epoch)))


            all_img_syn.extend(curr_img_syn)
            all_lbl_syn.extend(curr_lbl_syn)
            # update global weights
            # args.dsa = True
            if args.dsa:
                args.epoch_eval_train = 1000
                args.dc_aug_param = None
                logging.info('DSA augmentation strategy: \n%s'%args.dsa_strategy)
                logging.info('DSA augmentation parameters: \n%s'%args.dsa_param.__dict__)
            else:
                args.dc_aug_param = get_daparam(args.dataset, args.model, 'ConvNet') # only for DC. muted when args.dsa is True.
                logging.info('DC augmentation parameters: \n%s'%args.dc_aug_param)
            
            if args.dsa or args.dc_aug_param['strategy'] != 'none':
                args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
            else:
                args.epoch_eval_train = 500

            global_model.train()
            all_img_syn_eval, all_lbl_syn_eval = torch.cat(all_img_syn, dim=0), torch.cat(all_lbl_syn, dim=0)
            if curr_epoch == 0:
                num_img_per_round = all_img_syn_eval.shape[0]
            weights = torch.ones(num_img_per_round, ).cuda() * (curr_epoch+1)
            all_weight.append(weights)
            all_weight_eval = torch.cat(all_weight)
            global_model, acc_syns_train, acc_full_test = evaluate_synset(curr_epoch, global_model, all_img_syn_eval, all_lbl_syn_eval, testloader, args, weight=all_weight_eval)
            logging.info('%s Epoch = %04d test acc = %.4f' % (get_time(), curr_epoch, acc_full_test))
            fed_accs.append(acc_full_test)
            
        
        exp_acc_list.append(fed_accs)

    
    exp_acc_list = np.array(exp_acc_list)
    acc_mean = np.mean(exp_acc_list, axis=0)
    acc_std = np.std(exp_acc_list, axis=0)
    logging.info(acc_mean)
    logging.info(acc_std)




if __name__ == '__main__':
    main()

