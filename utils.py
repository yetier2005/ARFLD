import os
import glob
import pdb
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from models.networks import MLP, MLP_TINY, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP
import logging
import optimizer_cust

from dataset import *
from models import *

class LocalUser(object):
    def __init__(self, dataset, img_syn):
        self.dataset = dataset
        self.image_syn = img_syn

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))



def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 3 or ipc == 5: # investigate batch splitting
        outer_loop, inner_loop = 10, 50
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 25: # wrc added
        outer_loop, inner_loop = 25, 20
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    elif ipc in [100, 500, 1000]: # temp
        outer_loop, inner_loop = 0, 0
    else:
        outer_loop, inner_loop = 0, 0
        exit('DC error: loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop



def pick_gpu_lowest_memory():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU



def create_exp_dir(path, run_script=None):
    import os
    import shutil
    if not os.path.exists(path):
        os.makedirs(path)
    # print('Experiment dir : {}'.format(path))
    
    script_path = os.path.join(path, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path)

    tracked_items = getItemList('.', omit_list_file='.gitignore')
    for item in tracked_items:
        if 'ipynb' in item:
            continue
        if 'exp_scripts' in item: item = run_script

        dst_item = os.path.join(script_path, os.path.basename(item))

        if os.path.isdir(item):
            shutil.copytree(item, dst_item)
        else:
            shutil.copyfile(item, dst_item)


def getItemList(path, omit_list_file=None, omitted_paths=None):
    """ currently assume omit only contains paths with one level """
    def item_match(item1, item2):
        if item1[-1] == '/':  item1 = item1[:-1]
        if item1[:2] == './': item1 = item1[2:]
        if item2[-1] == '/':  item2 = item2[:-1]
        if item2[:2] == './': item2 = item2[2:]
        return item1 == item2

    # return nothing if path is a file
    if os.path.isfile(path):
        return []

    # get gitignored dirs
    if omitted_paths is None:
        omitted_paths = []
        if omit_list_file is not None:
            with open(os.path.join(path, omit_list_file), 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    omitted_paths.append(line)
    
    tracked_items = []
    for item in glob.glob(os.path.join(path, '*')):
        match = sum([item_match(item, it) for it in omitted_paths])
        if match == 0:
            tracked_items.append(item)
    return tracked_items


#### dataset pruning
def prune_synset(image_syn, label_syn, testloader, args,
                 evaluate_synset, get_network_closure):
    prune_ipc = args.prune_ipc
    cur_ipc = args.ipc

    image_syn = image_syn.detach()
    label_syn = label_syn.detach()

    if args.prune_method == 'enum':

        max_test, pruned_synset = 0, None
        for idx in range(cur_ipc // prune_ipc):
            mask = torch.ones((args.num_classes, cur_ipc), dtype=torch.bool)
            mask[:,idx*prune_ipc:(idx+1)*prune_ipc] = False
            mask = mask.flatten()
            assert mask.shape[0] == image_syn.shape[0]
            image_syn_loocv = image_syn[mask]
            label_syn_loocv = label_syn[mask]

            acc_full_test_list = []
            for it_eval in range(args.num_eval // 4):
                net_eval = get_network_closure()
                _, _, acc_full_test = evaluate_synset(it_eval, net_eval, image_syn_loocv, label_syn_loocv, testloader, args, 
                                                    verbose=False, fast=True)
                acc_full_test_list.append(acc_full_test)
            acc_full_test_avg = np.mean(acc_full_test_list)

            if acc_full_test_avg > max_test:
                max_test = acc_full_test_avg
                pruned_synset = [image_syn_loocv.requires_grad_(True), label_syn_loocv]

            if args.fast: break
    elif args.prune_method == 'exhaustive':
        pruned_ids = []
        for c in reversed(range(args.num_classes)): # per class pruning
            
            max_test, prune_mask = 0, None
            for idx in reversed(range(cur_ipc // prune_ipc)):
                mask = torch.ones((args.num_classes, cur_ipc), dtype=torch.bool)
                mask[c, prune_ipc*idx:prune_ipc*(idx + 1)] = False
                mask = torch.ones_like(label_syn, dtype=torch.bool)
                mask[cur_ipc*c+prune_ipc*idx:cur_ipc*c+prune_ipc*(idx+1)] = False

                image_syn_loocv = image_syn[mask]
                label_syn_loocv = label_syn[mask]

                acc_full_test_list = []
                for it_eval in range(args.num_eval // 4):
                    net_eval = get_network_closure()
                    _, _, acc_full_test = evaluate_synset(it_eval, net_eval, image_syn_loocv, label_syn_loocv, testloader, args, 
                                                          verbose=False, fast=True)
                    acc_full_test_list.append(acc_full_test)
                acc_full_test_avg = np.mean(acc_full_test_list)

                if acc_full_test_avg > max_test:
                    max_test = acc_full_test_avg
                    prune_mask = mask
                # print('-'*20)
                # print(acc_full_test_avg)
                # print(mask)
            # print('='*20)
            # print(prune_mask)
            image_syn = image_syn[prune_mask]
            label_syn = label_syn[prune_mask]

        pruned_synset = [image_syn.requires_grad_(True), label_syn]
    else:
        logging.info(f'ERROR: invalid prune method: {args.prune_method}')

    logging.info('---> Avg Test Acc After Pruning: {:.4f}'.format(max_test))
    return pruned_synset