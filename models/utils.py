import pdb
import time
import logging
import torch
import torch.nn as nn
from models.networks import MLP, MLP_TINY, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP, CharCNN
import optimizer_cust



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

def get_default_charcnn_setting():
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    max_length, input_dim = 1014, len(alphabet)
    return max_length, input_dim


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


def get_network(model, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    max_length, input_dim = get_default_charcnn_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    if model == 'MLP_TINY':
        net = MLP_TINY(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none')
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')
    elif model == 'CharCNNSmall':
        net = CharCNN(input_length=max_length, n_classes=num_classes,
                      input_dim=input_dim,
                      n_conv_filters=256, n_fc_neurons=1024)
    elif model == 'CharCNNLarge':
        net = CharCNN(input_length=max_length, n_classes=num_classes,
                      input_dim=input_dim, 
                      n_conv_filters=1024, n_fc_neurons=2048)
    else:
        net = None
        exit('DC error: unknown model')

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net



def get_optimizer(parameters, opt_name, lr, weight_decay, rho, momentum):
    base_optimizer = optimizer_cust.SGD
    if opt_name == 'sgd':
        optimizer = base_optimizer(parameters, lr=lr, momentum=momentum)
    if opt_name == 'sam':
        optimizer = optimizer_cust.SAM(parameters, base_optimizer, rho=rho, adaptive=False,
                                       lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == 'asam':
        optimizer = optimizer_cust.SAM(parameters, base_optimizer, rho=rho, adaptive=True,
                                       lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == 'sam-rand':
        optimizer = optimizer_cust.SAM(parameters, base_optimizer, rho=rho, adaptive=False, rand=True,
                                       lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == 'asam-rand':
        optimizer = optimizer_cust.SAM(parameters, base_optimizer, rho=rho, adaptive=True, rand=True,
                                       lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == 'lbfgs':
        optimizer = torch.optim.LBFGS(parameters, lr=lr)
    optimizer.zero_grad()
    return optimizer



def get_lr_schedule(optimizer, opt_name, lr, Epoch):
    if opt_name == 'sgd':
        lr_schedule = optimizer_cust.StepLR_SGD(optimizer, lr=lr, total_epochs=Epoch)
    elif opt_name in ['sam', 'asam', 'sam-rand', 'asam-rand']:
        lr_schedule = optimizer_cust.StepLR_SAM(optimizer, lr=lr, total_epochs=Epoch)
    return lr_schedule
    


def copy_model(src, dst):
    for p, pp in zip(dst.parameters(), src.parameters()):
        p.data.copy_(pp.data)



def distance_wb(gwr, gws, dist_dropout=0):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0 ## TODO under dev ## here changed to tensor(0)
    if dist_dropout > 0:
        dout_mask = (torch.cuda.FloatTensor(size=gwr.shape).uniform_() > dist_dropout).float()
        gwr *= dout_mask
        gws *= dout_mask
    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).cuda()

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            try:
                dis += distance_wb(gwr, gws, dist_dropout=args.dist_dropout)
            except:
                dis += distance_wb(gwr, gws, dist_dropout=0)

    elif args.dis_metric == 'mse':
        assert not args.dist_dropout, 'not implemented for mse'
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        assert not args.dist_dropout, 'not implemented for cos'
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis



def theta_matching(net1, net2, args):
    net1_param = list(net1.parameters())
    net1_param_flat = torch.cat([p.flatten() for p in net1_param], dim=0)
    
    net2_param = list(net2.parameters())
    net2_param_flat = torch.cat([p.flatten() for p in net2_param], dim=0)

    l2 = (net1_param_flat - net2_param_flat).norm()
    dist = match_loss(net1_param, net2_param, args)
    norm1 = net1_param_flat.norm()
    norm2 = net2_param_flat.norm()
    return l2, dist, norm1, norm2



def compute_match_loss(imgs_real, labs_real, imgs_syn, labs_syn,
                        net, net_parameters, criterion, args, perturb=False):
    """ compute the gradient match loss for X, X~ """
    assert isinstance(net_parameters, list)
    output_real = net(imgs_real)
    loss_real = criterion(output_real, labs_real)
    if perturb:
        gw_real = torch.autograd.grad(loss_real, net_parameters, create_graph=True)
    else:
        gw_real = torch.autograd.grad(loss_real, net_parameters)
        gw_real = list((_.detach().clone() for _ in gw_real))
    
    output_syn = net(imgs_syn)
    loss_syn = criterion(output_syn, labs_syn)
    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
    loss = match_loss(gw_syn, gw_real, args)
    return loss



def cross_entropy_loss_cust(args):
    '''
    input: softmaxed output
    '''
    class xent:
        def __init__(self, args):
            self.logsoftmax = nn.LogSoftmax(dim=-1)
            self.torch_xent = nn.CrossEntropyLoss(reduction='mean')
            self.args = args
            try:
                self.normalize = (self.args.label == 'soft_norm')
                self.softmax   = (self.args.label == 'soft_sm')
            except:
                self.normalize = False
                self.softmax = False
        
        def __call__(self, output, target, weight=None):
            if len(target.shape) > 1: # soft-label
                if self.normalize:
                    target /= target.sum(dim=-1).view(-1, 1)
                if self.softmax:
                    target = torch.softmax(target, dim=-1)
                return torch.mean(torch.sum(- target * self.logsoftmax(output), 1))
            else:
                # all_loss = self.torch_xent(output, target)
                # print(all_loss)
                # weight /= weight.sum(dim=-1)
                # weight = weight
                # print(all_loss * weight)
                # return (all_loss * weight).sum()
                return self.torch_xent(output, target)

        def cuda(self):
            self.logsoftmax = self.logsoftmax.cuda()
            self.torch_xent = self.torch_xent.cuda()
            return self

    return xent(args)

# def soft_cross_entropy(pred, soft_targets):
#     """A method for calculating cross entropy with soft targets"""
#     logsoftmax = nn.LogSoftmax()
#     return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def tensor_list_add(tl1, tl2):
    if tl1 is None: # first addition
        return tl2
    else:
        assert len(tl1) == len(tl2)
        return [t1 + t2 for t1, t2 in zip(tl1, tl2)]


def tensor_list_div(tl, val):
    return [t / val for t in tl]