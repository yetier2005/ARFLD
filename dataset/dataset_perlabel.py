import numpy as np
import torch
import os


class PerLabelDatasetNonIID():
    def __init__(self, dst_train, classes, channel, args): # images: n x c x h x w tensor
        self.images_all = []
        labels_all = []
        self.indices_class = {c: [] for c in classes}

        self.images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            lab = lab.item()
            if lab not in classes:
                continue
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).cuda()
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        # for c in range(num_classes):
        #     logging.info('class c = %d: %d real images'%(c, len(self.indices_class[c])))

        # for ch in range(channel):
        #     logging.info('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(self.images_all[:, ch]), torch.std(self.images_all[:, ch])))

    def __len__(self):
        return self.images.shape[0]

    def get_random_images(self, n): # get n random images
        idx_shuffle = np.random.permutation(range(self.images_all.shape[0]))[:n]
        return self.images_all[idx_shuffle]
    
    def get_images(self, c, n, avg=False): # get n random images from class c
        if not avg:
            if len(self.indices_class[c]) >= n:
                idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            else:
                sampled_idx = np.random.choice(self.indices_class[c], n-len(self.indices_class[c]), replace=True)
                idx_shuffle = np.concatenate((self.indices_class[c], sampled_idx), axis=None)
            return self.images_all[idx_shuffle]
        else:
            sampled_imgs = []
            for _ in range(n):
                if len(self.indices_class[c])>=5:
                    idx = np.random.choice(self.indices_class[c], 5, replace=False)
                else:
                    idx = np.random.choice(self.indices_class[c], 5, replace=True)
                sampled_imgs.append(torch.mean(self.images_all[idx], dim=0, keepdim=True))
            sampled_imgs = torch.cat(sampled_imgs, dim=0).cuda()
            return sampled_imgs

#### code for fetch per-label images
class PerLabelDataset():
    def __init__(self, dst_train, num_classes, channel, args): # images: n x c x h x w tensor
        self.images_all = []
        labels_all = []
        self.indices_class = [[] for c in range(num_classes)] # (num_classes, data_idx)

        self.images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).cuda()
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        self.pseudo_class = [[] for c in range(10)]


        # for c in range(num_classes):
        #     logging.info('class c = %d: %d real images'%(c, len(self.indices_class[c])))

        # for ch in range(channel):
        #     logging.info('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(self.images_all[:, ch]), torch.std(self.images_all[:, ch])))

    def __len__(self):
        return self.images_all.shape[0]
    
    def get_images(self, c, n): # get n random images from class c
        if len(self.indices_class[c]) >= n:
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        else:
            sampled_idx = np.random.choice(self.indices_class[c], n-len(self.indices_class[c]), replace=True)
            idx_shuffle = np.concatenate((self.indices_class[c], sampled_idx), axis=None)
        return self.images_all[idx_shuffle]

    def get_random_images(self, n): # get n random images
        idx_shuffle = np.random.permutation(range(self.images_all.shape[0]))[:n]
        return self.images_all[idx_shuffle]

    def get_images_with_pseudo_labels(self, c, n): # get n random images based on pseudo labels
        if len(self.pseudo_class[c]) >= n:
            idx_shuffle = np.random.permutation(self.pseudo_class[c])[:n]
        else:
            sampled_idx = np.random.choice(self.pseudo_class[c], n-len(self.pseudo_class[c]), replace=True)
            idx_shuffle = np.concatenate((self.pseudo_class[c], sampled_idx), axis=None)
        return self.images_all[idx_shuffle]


class PerLabelLargeDataset(): # load images on the fly
    def __init__(self, dst_train, num_classes, channel, args): # images: n x c x h x w tensor
        self.dst_train = dst_train
        self.num_classes = num_classes
        self.args = args

        self.targets = torch.tensor(self.dst_train.targets if hasattr(self.dst_train, 'targets') else self.dst_train.labels)
        self.perlabel_loaders = []
        #### TODO: DELETE THIS 
        print("!!!!WARNING!!!!\n"*3)
        print("in PerLabelLargeDataset much much smaller batch size")
        # batch_size = num_classes*args.batch_real
        batch_size = 8
        self.loader = torch.utils.data.DataLoader(self.dst_train, batch_size=batch_size, num_workers=32)

    def get_init_images(self, ipc):
        images = []
        image_inds = []
        for c in range(self.num_classes):
            target_idx = torch.where(self.targets == c)[0]
            ind = np.random.choice(target_idx, size=ipc, replace=False)
            image_inds.append(ind)
        image_inds = np.concatenate(image_inds, axis=0)
        images = [self.dst_train[idx] for idx in image_inds]
        return images


#### code for fetch per-label nlp (only for tiny dataset)
class PerLabelNLPDataset():
    def __init__(self, dst_train, num_classes, args): # texts: batch_size x seq_len x num_char tensor
        self.texts_all = []
        labels_all = []
        self.indices_class = [[] for c in range(num_classes)] # (num_classes, data_idx)

        preprocessed_text_path = os.path.join(args.data_path, 'nlp', f'{args.dataset}_?.pth')
        if os.path.exists(preprocessed_text_path.replace('?', 'texts')):
            print('loading from preprocessed data...')
            self.texts_all = torch.load(preprocessed_text_path.replace('?', 'texts'))
            self.indices_class = torch.load(preprocessed_text_path.replace('?', 'indices_class'))
        else:
            ## TODO under dev ##
            # self.texts_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
            self.texts_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(10000)]
            self.texts_all = torch.cat(self.texts_all, dim=0)
            
            # labels_all = [dst_train[i][1] for i in range(len(dst_train))]
            labels_all = [dst_train[i][1] for i in range(10000)]
            for i, lab in enumerate(labels_all):
                self.indices_class[lab].append(i)
            
            torch.save(self.texts_all,  preprocessed_text_path.replace('?', 'texts'))
            torch.save(self.indices_class, preprocessed_text_path.replace('?', 'indices_class'))

        self.texts_all = self.texts_all.cuda()
        
        for c in range(num_classes):
            print('class c = %d: %d real texts'%(c, len(self.indices_class[c])))

    def __len__(self):
        return self.images.shape[0]
    
    def get_texts(self, c, n): # get n random images from class c
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        return self.images_all[idx_shuffle]


class PerLabelLargeNLPDataset(): # load texts on the fly
    def __init__(self, dst_train, num_classes, args): # texts: batch_size x seq_len x num_char tensor
        self.dst_train = dst_train
        self.num_classes = num_classes
        self.args = args
        
        batch_size = num_classes*args.batch_real
        self.loader = torch.utils.data.DataLoader(self.dst_train, batch_size=batch_size, shuffle=True, num_workers=4)
        self.loader_iter = iter(self.loader)

    def get_texts(self, n):
        texts = torch.tensor([])
        targets = torch.LongTensor([])
        while texts.shape[0] < n:
            try:
                text, target = next(self.loader_iter)
            except:
                self.loader_iter = iter(self.loader)
                text, target = next(self.loader_iter)
                
            texts = torch.cat([texts, text], dim=0)
            targets = torch.cat([targets, target], dim=0)
        texts = texts[:n].cuda()
        targets = targets[:n].cuda()
        return texts, targets