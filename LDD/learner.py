import torchvision.transforms.functional as TF
from module.memory import MemoryWrapLayer
from module.loss import GeneralizedCELoss
from module.models import dic_models
from module.models2 import dic_models_2
from module.resnets_vision import dic_models_3
from data.util import get_dataset, IdxDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from util import EMA
from module.utils import dic_functions
from config import *
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
set_seed = dic_functions['set_seed']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

write_to_file = dic_functions['write_to_file']

class trainer():

    def __init__(self, args):
        self.run_type = args.run_type
        self.loader_config = dataloader_confg[args.dataset_in]
        if args.dataset_in == 'CMNIST':
            print("[DATASET][CMNIST]")
            self.dataset_in = 'ColoredMNIST-Skewed0.05-Severity4'
        elif args.dataset_in == 'CIFAR':
            print("[DATASET][CIFAR]")
            self.dataset_in = 'CorruptedCIFAR10-Type1-Skewed0.05-Severity2'
        else:
            print("[DATASET][CELEBA]")
            self.dataset_in = args.dataset_in
        self.model_in = args.model_in
        self.train_samples = args.train_samples
        self.bias_ratio = args.bias_ratio
        self.reduce = args.reduce
        self.target_attr_idx = 0
        self.bias_attr_idx = 1
        if 'CelebA' in self.dataset_in:
            self.target_attr_idx = 9
            self.bias_attr_idx = 20
            self.num_classes = 2
        else:
            self.num_classes = 10
    
    def concat_dummy(self,z):
        def hook(model, input, output):
            z.append(output.squeeze())
            return torch.cat((output, torch.zeros_like(output)), dim=1)
        return hook

    def rotation_ssl_data(self, images, labels):
        labels = torch.zeros(len(labels))
        images_90 = TF.rotate(images, 90)
        labels_90 = torch.ones(len(labels))
        images_180 = TF.rotate(images, 180)
        labels_180 = torch.ones(len(labels))*2
        images_270 = TF.rotate(images, 270)
        labels_270 = torch.ones(len(labels))*3
        images = torch.cat((images, images_90, images_180, images_270), dim=0)
        labels = torch.cat((labels, labels_90, labels_180, labels_270), dim=0)
        images = images
        labels = labels
        return images, labels

    def store_results(self, test_accuracy, test_accuracy_epoch, test_cheat):
        write_to_file('results_text/'+self.run_type+'_'+self.dataset_in.split('-')[0]+'_'+str(self.train_samples)+'_'+str(self.bias_ratio)+'.txt','[Best Test Accuracy]'+str(test_accuracy)+"[Final Epoch Test Accuracy]"+str(test_accuracy_epoch)+ '[Best Cheat Test Accuracy]'+str(test_cheat))

    def datasets(self):
        self.train_dataset = get_dataset(
            self.dataset_in,
            data_dir='/home/user/datasets/debias',
            dataset_split="train",
            transform_split="train",)
        self.test_dataset = get_dataset(
            self.dataset_in,
            data_dir='/home/user/datasets/debias',
            dataset_split="eval",
            transform_split="eval",)
        self.valid_dataset = get_dataset(
            self.dataset_in,
            data_dir='/home/user/datasets/debias',
            dataset_split="train",
            transform_split="train",
            add = True
        )
    
    def reduce_data(self):
        indices_train_biased = self.train_dataset.attr[:,self.target_attr_idx] == self.train_dataset.attr[:,self.bias_attr_idx]
        indices_train_biased = indices_train_biased.nonzero().squeeze()

        nums_train_biased = []
        for i in range(self.num_classes):
            nums_train_biased.append(np.random.choice(indices_train_biased[self.train_dataset.attr[indices_train_biased,self.target_attr_idx] == i], int((1-self.bias_ratio) * self.train_samples/self.num_classes) , replace=False))
        nums_train_biased = np.concatenate(nums_train_biased)


        indices_train_unbiased = self.train_dataset.attr[:,self.target_attr_idx] != self.train_dataset.attr[:,self.bias_attr_idx]
        indices_train_unbiased = indices_train_unbiased.nonzero().squeeze()
        
        nums_train_unbiased = []
        for i in range(self.num_classes):
            nums_train_unbiased.append(np.random.choice(indices_train_unbiased[self.train_dataset.attr[indices_train_unbiased,self.target_attr_idx] == i], int(self.bias_ratio * self.train_samples/self.num_classes) , replace=False))
        
        nums_train_unbiased = np.concatenate(nums_train_unbiased)

        nums_train = np.concatenate((nums_train_biased, nums_train_unbiased))

        nums_valid_unbiased = []
        while len(nums_valid_unbiased) < 1000:
            i = np.random.randint(0, len(self.valid_dataset))
            if self.valid_dataset.attr[i,self.target_attr_idx] != self.valid_dataset.attr[i,self.bias_attr_idx] and i not in nums_train:
                nums_valid_unbiased.append(i)
        nums_valid_unbiased = np.array(nums_valid_unbiased)

        if self.dataset_in == 'CelebA':
            self.train_dataset.data = [self.train_dataset.data[index] for index in nums_train]
            self.train_dataset.attr = self.train_dataset.attr[nums_train]
            self.valid_dataset.data = [self.valid_dataset.data[index] for index in nums_valid_unbiased]
            self.valid_dataset.attr = self.valid_dataset.attr[nums_valid_unbiased]

        else:
            self.valid_dataset.attr = self.valid_dataset.attr[nums_valid_unbiased]
            self.valid_dataset.data = self.valid_dataset.data[nums_valid_unbiased]
            self.valid_dataset.__len__ = 1000
            self.valid_dataset.query_attr = self.valid_dataset.attr[:, torch.arange(2)]
            
            self.train_dataset.attr = self.train_dataset.attr[nums_train]
            self.train_dataset.data = self.train_dataset.data[nums_train]
            self.train_dataset.__len__ = self.train_samples
            self.train_dataset.query_attr = self.train_dataset.attr[:, torch.arange(2)]
        del indices_train_biased, indices_train_unbiased, nums_train_biased, nums_train_unbiased, nums_train, nums_valid_unbiased


    def dataloaders(self):

        print("[Size of the Dataset]["+str(len(self.train_dataset))+"]")
        print("[Conflicting Samples in Training Data]["+str(len(self.train_dataset.attr[self.train_dataset.attr[:,self.target_attr_idx] != self.train_dataset.attr[:,self.bias_attr_idx]]))+"]")
        print("[Conflicting Samples in Validation Data]["+str(len(self.valid_dataset.attr[self.valid_dataset.attr[:,self.target_attr_idx] != self.valid_dataset.attr[:,self.bias_attr_idx]]))+"]")
        print("[Conflicting Samples in Test Data]["+str(len(self.test_dataset.attr[self.test_dataset.attr[:,self.target_attr_idx] != self.test_dataset.attr[:,self.bias_attr_idx]]))+"]")

        print("[Number of samples in each class]")
        for i in range(self.num_classes):
            print("[Class "+str(i)+"]")
            print("[Training Data]["+str(len(self.train_dataset.attr[self.train_dataset.attr[:,self.target_attr_idx] == i]))+"]")

        self.train_target_attr = self.train_dataset.attr[:, self.target_attr_idx]
        self.train_bias_attr = self.train_dataset.attr[:, self.bias_attr_idx]

        self.train_dataset = IdxDataset(self.train_dataset)
        self.valid_dataset = IdxDataset(self.valid_dataset)    
        self.test_dataset = IdxDataset(self.test_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            **self.loader_config['train'],)

        self.test_loader = DataLoader(
            self.test_dataset,
            **self.loader_config['test'],)

        self.valid_loader = DataLoader(
            self.valid_dataset,
            **self.loader_config['valid'],)
        
        if 'MW' in self.run_type:
            self.mem_loader = DataLoader(
                self.train_dataset,
                **self.loader_config['memory'],)
            
    def models(self):
        if 'MW' in self.run_type:
            try:
                self.model_d = dic_models[self.model_in+'_MW'](self.num_classes).to(device)
                self.model_b = dic_models[self.model_in](self.num_classes).to(device)
            except:
                try:
                    self.model_d = dic_models_2[self.model_in+'_MW'](self.num_classes).to(device)
                    self.model_b = dic_models_2[self.model_in](self.num_classes).to(device)
                except:
                    self.model_d = dic_models_3[self.model_in+'_MW'](self.num_classes).to(device)
                    self.model_b = dic_models_3[self.model_in](self.num_classes).to(device)
            self.model_d.fc = MemoryWrapLayer(self.model_d.fc.in_features*2, self.num_classes).to(device)
            self.model_b.fc = nn.Linear(self.model_b.fc.in_features*2, self.num_classes).to(device)
        else:
            try:
                self.model_d = dic_models[self.model_in](self.num_classes).to(device)
                self.model_b = dic_models[self.model_in](self.num_classes).to(device)
            except:
                try:
                    self.model_d = dic_models_2[self.model_in](self.num_classes).to(device)
                    self.model_b = dic_models_2[self.model_in](self.num_classes).to(device)
                except:
                    self.model_d = dic_models_3[self.model_in](self.num_classes).to(device)
                    self.model_b = dic_models_3[self.model_in](self.num_classes).to(device)
            self.model_d.fc = nn.Linear(self.model_d.fc.in_features*2, self.num_classes).to(device)
            self.model_b.fc = nn.Linear(self.model_b.fc.in_features*2, self.num_classes).to(device)
        if 'rotation' in self.run_type:
             self.model_mlp = nn.Sequential(nn.Linear(self.model_d.fc.in_features//2, 4),).to(device)

        print("[MODEL]["+self.model_in+"]")
    
    def optimizers(self):
        if 'MNIST' in self.dataset_in:
            self.optimizer_b = torch.optim.Adam(self.model_b.parameters(),lr= 0.002, weight_decay=0.0)
            self.optimizer_d = torch.optim.Adam(self.model_d.parameters(),lr= 0.002, weight_decay=0.0)
            self.schedulerd = MultiStepLR(self.optimizer_d, milestones=[300], gamma=0.5)
            self.schedulerb = MultiStepLR(self.optimizer_b, milestones=[300], gamma=0.5)
            self.epochs = 200
            self.lambda_swap_align = 10.0
            self.lambda_dis_align = 10.0
            self.lambda_swap_ = 1.0
            self.loss_rotate_param = 0.5
        elif self.dataset_in == 'CelebA':
            self.optimizer_b = torch.optim.Adam(self.model_b.parameters(),lr= 1e-4, weight_decay=1e-4)
            self.optimizer_d = torch.optim.Adam(self.model_d.parameters(),lr= 1e-4, weight_decay=1e-4)
            self.schedulerd = MultiStepLR(self.optimizer_d, milestones=[300], gamma=0.5)
            self.schedulerb = MultiStepLR(self.optimizer_b, milestones=[300], gamma=0.5)
            self.epochs = 200
            self.lambda_swap_align = 2.0
            self.lambda_dis_align = 2.0
            self.lambda_swap_ = 1.0
            self.loss_rotate_param = 2.0
        else:
            self.optimizer_b = torch.optim.SGD(self.model_b.parameters(),lr= 0.1, weight_decay=5e-4, momentum = 0.9, nesterov = True)
            self.optimizer_d = torch.optim.SGD(self.model_d.parameters(),lr= 0.1, weight_decay=5e-4, momentum = 0.9, nesterov = True)
            self.schedulerd = MultiStepLR(self.optimizer_d, milestones=[150,225], gamma=0.1)
            self.schedulerb = MultiStepLR(self.optimizer_b, milestones=[150,225], gamma=0.1)
            self.epochs = 300
            self.lambda_swap_align = 5.0
            self.lambda_dis_align = 5.0
            self.lambda_swap_ = 1.0
            self.loss_rotate_param = 2.0
        if 'rotation' in self.run_type:
            self.optimizer_mlp = torch.optim.Adam(self.model_mlp.parameters(), lr=0.001)
        print("[OPTIMIZER]["+str(self.optimizer_d)+"]")

        print("[EPOCHS]["+str(self.epochs)+"]")
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        self.criterion = criterion.to(device)
        bias_criterion = GeneralizedCELoss()
        self.bias_criterion = bias_criterion.to(device)

        self.sample_loss_ema_b = EMA(torch.LongTensor(self.train_target_attr), num_classes=self.num_classes, alpha=0.9)
        self.sample_loss_ema_d = EMA(torch.LongTensor(self.train_target_attr), num_classes=self.num_classes, alpha=0.9)
    
    def train_LDD(self):
        scaler = GradScaler()
        test_accuracy = -1.0
        test_cheat = -1.0
        test_accuracy_epoch = -1.0
        valid_accuracy_best = -1.0

        evaluate_accuracy = dic_functions['LDD']

        for epoch in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                data = data.to(device)
                attr = attr.to(device)
                
                label = attr[:, self.target_attr_idx]
                try:
                    z_b = []
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    with autocast():
                        _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                    
                    z_d = []
                    hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_d))
                    with autocast():
                        _ = self.model_d(data)
                    hook_fn.remove()
                    z_d = z_d[0]
                    
                    if epoch == 1 and ix == 0:
                        print("[Average Pool layer Selected]")
                except:
                    z_b = []
                    hook_fn = self.model_b.layer4.register_forward_hook(self.concat_dummy(z_b))
                    with autocast():
                        _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                    
                    z_d = []
                    hook_fn = self.model_d.layer4.register_forward_hook(self.concat_dummy(z_d))
                    with autocast():
                        _ = self.model_d(data)
                    hook_fn.remove()
                    z_d = z_d[0]
                    if epoch == 1 and ix == 0:
                        print("[Layer 4 Selected]")
                
                z_conflict = torch.cat((z_d, z_b.detach()), dim=1)
                z_align = torch.cat((z_d.detach(), z_b), dim=1)
                
                with autocast():
                    pred_conflict = self.model_d.fc(z_conflict)
                    pred_align = self.model_b.fc(z_align)
                    loss_dis_conflict = self.criterion(pred_conflict, label)
                    loss_dis_align = self.criterion(pred_align, label)
                
                loss_dis_conflict = loss_dis_conflict.detach()
                loss_dis_align = loss_dis_align.detach()
                
                self.sample_loss_ema_d.update(loss_dis_conflict, index)
                self.sample_loss_ema_b.update(loss_dis_align, index)
                
                loss_dis_conflict = self.sample_loss_ema_d.parameter[index].clone().detach()
                loss_dis_align = self.sample_loss_ema_b.parameter[index].clone().detach()
                
                loss_dis_conflict = loss_dis_conflict.to(device)
                loss_dis_align = loss_dis_align.to(device)
                
                for c in range(self.num_classes):
                    class_index = torch.where(label == c)[0].to(device)
                    max_loss_conflict = self.sample_loss_ema_d.max_loss(c)
                    max_loss_align = self.sample_loss_ema_b.max_loss(c)
                    loss_dis_conflict[class_index] /= max_loss_conflict
                    loss_dis_align[class_index] /= max_loss_align
                
                loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)  
                loss_weight = loss_weight.to(device)
                
                with autocast():
                    loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight           
                    loss_dis_align = self.bias_criterion(pred_align, label)
                
                if epoch >= 30:
                    indices = np.random.permutation(z_b.size(0))
                    z_b_swap = z_b[indices]         # z tilde
                    label_swap = label[indices]     # y tilde
                    z_mix_conflict = torch.cat((z_d, z_b_swap.detach()), dim=1)
                    z_mix_align = torch.cat((z_d.detach(), z_b_swap), dim=1)
                    with autocast():
                        pred_mix_conflict = self.model_d.fc(z_mix_conflict)
                        pred_mix_align = self.model_b.fc(z_mix_align)
                        loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight     
                        loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)                               
                    lambda_swap = self.lambda_swap_    
                else:
                    loss_swap_conflict = torch.tensor([0]).float()
                    loss_swap_align = torch.tensor([0]).float()
                    lambda_swap = 0
                
                with autocast():
                    loss_dis  = loss_dis_conflict.mean() + self.lambda_dis_align * loss_dis_align.mean()                # Eq.2 L_dis
                    loss_swap = loss_swap_conflict.mean() + self.lambda_swap_align * loss_swap_align.mean()             # Eq.3 L_swap
                    loss = loss_dis + lambda_swap * loss_swap 
                self.optimizer_d.zero_grad()
                self.optimizer_b.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_d)
                scaler.step(self.optimizer_b)
                scaler.update()
            self.schedulerb.step()
            self.schedulerd.step()
            
            train_accuracy_epoch = evaluate_accuracy(self.model_b, self.model_d, self.train_loader, self.target_attr_idx, device)
            prev_valid_accuracy = valid_accuracy_best
            valid_accuracy_epoch = evaluate_accuracy(self.model_b, self.model_d, self.valid_loader, self.target_attr_idx, device)
            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)
            
            print("[Epoch "+str(epoch)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")
            
            test_accuracy_epoch = evaluate_accuracy(self.model_b, self.model_d, self.test_loader, self.target_attr_idx, device)
            
            test_cheat = max(test_cheat, test_accuracy_epoch)
            
            print("[Test Accuracy cheat][%.4f]"%test_cheat)
            
            if valid_accuracy_best > prev_valid_accuracy:
                test_accuracy = test_accuracy_epoch
            
            print('[Best Test Accuracy]', test_accuracy)
        
        return test_accuracy, test_accuracy_epoch, test_cheat

    def train_LDD_MW(self):
        scaler = GradScaler()
        test_accuracy = -1.0
        test_cheat = -1.0
        test_accuracy_epoch = -1.0
        valid_accuracy_best = -1.0
        mem_iter = None
        evaluate_accuracy = dic_functions['LDD_MW']

        for epoch in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                data = data.to(device)
                attr = attr.to(device)
                label = attr[:, self.target_attr_idx]

                try:
                    indexm, datam, labelm = next(mem_iter)
                except:
                    mem_iter = iter(self.mem_loader)
                    indexm, datam, labelm = next(mem_iter)
                
                datam = datam.to(device)

                try:
                    z_b = []
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    with autocast():
                        _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                    
                    z_d = []
                    hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_d))
                    with autocast():
                        _ = self.model_d(data, data)
                    hook_fn.remove()
                    z_d = z_d[0]

                    z_bm = []
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_bm))
                    with autocast():
                        _ = self.model_b(datam)
                    hook_fn.remove()
                    z_bm = z_bm[0]
                    
                    z_dm = []
                    hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_dm))
                    with autocast():
                        _ = self.model_d(datam, datam)
                    hook_fn.remove()
                    z_dm = z_dm[0]

                    if epoch == 1 and ix == 0:
                        print("[Average Pool layer Selected]")
                except:
                    z_b = []
                    hook_fn = self.model_b.layer4.register_forward_hook(self.concat_dummy(z_b))
                    with autocast():
                        _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                    
                    z_d = []
                    hook_fn = self.model_d.layer4.register_forward_hook(self.concat_dummy(z_d))
                    with autocast():
                        _ = self.model_d(data)
                    hook_fn.remove()
                    z_d = z_d[0]

                    z_bm = []
                    hook_fn = self.model_b.layer4.register_forward_hook(self.concat_dummy(z_bm))
                    with autocast():
                        _ = self.model_b(datam)
                    hook_fn.remove()
                    z_bm = z_bm[0]
                    
                    z_dm = []
                    hook_fn = self.model_d.layer4.register_forward_hook(self.concat_dummy(z_dm))
                    with autocast():
                        _ = self.model_d(datam, datam)
                    hook_fn.remove()
                    z_dm = z_dm[0]

                    if epoch == 1 and ix == 0:
                        print("[Layer 4 Selected]")
                
                z_conflict = torch.cat((z_d, z_b.detach()), dim=1)
                z_align = torch.cat((z_d.detach(), z_b), dim=1)
                mem_features = torch.cat((z_dm, z_bm.detach()), dim=1)

                with autocast():
                    pred_conflict = self.model_d.fc(z_conflict, mem_features)
                    pred_align = self.model_b.fc(z_align)
                    loss_dis_conflict = self.criterion(pred_conflict, label)
                    loss_dis_align = self.criterion(pred_align, label)
                
                loss_dis_conflict = loss_dis_conflict.detach()
                loss_dis_align = loss_dis_align.detach()
                
                self.sample_loss_ema_d.update(loss_dis_conflict, index)
                self.sample_loss_ema_b.update(loss_dis_align, index)
                
                loss_dis_conflict = self.sample_loss_ema_d.parameter[index].clone().detach()
                loss_dis_align = self.sample_loss_ema_b.parameter[index].clone().detach()
                
                loss_dis_conflict = loss_dis_conflict.to(device)
                loss_dis_align = loss_dis_align.to(device)
                
                for c in range(self.num_classes):
                    class_index = torch.where(label == c)[0].to(device)
                    max_loss_conflict = self.sample_loss_ema_d.max_loss(c)
                    max_loss_align = self.sample_loss_ema_b.max_loss(c)
                    loss_dis_conflict[class_index] /= max_loss_conflict
                    loss_dis_align[class_index] /= max_loss_align
                
                loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)  
                loss_weight = loss_weight.to(device)
                
                with autocast():
                    loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight           
                    loss_dis_align = self.bias_criterion(pred_align, label)
                
                if epoch >= 30:
                    indices = np.random.permutation(z_b.size(0))
                    z_b_swap = z_b[indices]         # z tilde
                    label_swap = label[indices]     # y tilde
                    z_mix_conflict = torch.cat((z_d, z_b_swap.detach()), dim=1)
                    z_mix_align = torch.cat((z_d.detach(), z_b_swap), dim=1)
                    with autocast():
                        pred_mix_conflict = self.model_d.fc(z_mix_conflict, mem_features)
                        pred_mix_align = self.model_b.fc(z_mix_align)
                        loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight     
                        loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)                               
                    lambda_swap = self.lambda_swap_    
                else:
                    loss_swap_conflict = torch.tensor([0]).float()
                    loss_swap_align = torch.tensor([0]).float()
                    lambda_swap = 0
                
                with autocast():
                    loss_dis  = loss_dis_conflict.mean() + self.lambda_dis_align * loss_dis_align.mean()                # Eq.2 L_dis
                    loss_swap = loss_swap_conflict.mean() + self.lambda_swap_align * loss_swap_align.mean()             # Eq.3 L_swap
                    loss = loss_dis + lambda_swap * loss_swap 
                self.optimizer_d.zero_grad()
                self.optimizer_b.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_d)
                scaler.step(self.optimizer_b)
                scaler.update()
            self.schedulerb.step()
            self.schedulerd.step()
            
            train_accuracy_epoch = evaluate_accuracy(self.model_b, self.model_d, self.train_loader, self.mem_loader,self.target_attr_idx, device)
            prev_valid_accuracy = valid_accuracy_best
            valid_accuracy_epoch = evaluate_accuracy(self.model_b, self.model_d, self.valid_loader, self.mem_loader, self.target_attr_idx, device)
            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)
            
            print("[Epoch "+str(epoch)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")
            
            test_accuracy_epoch = evaluate_accuracy(self.model_b, self.model_d, self.test_loader, self.mem_loader, self.target_attr_idx, device)
            
            test_cheat = max(test_cheat, test_accuracy_epoch)
            
            print("[Test Accuracy cheat][%.4f]"%test_cheat)
            
            if valid_accuracy_best > prev_valid_accuracy:
                test_accuracy = test_accuracy_epoch
            
            print('[Best Test Accuracy]', test_accuracy)
        
        return test_accuracy, test_accuracy_epoch, test_cheat

    def train_LDD_rotation(self):
        test_accuracy = -1.0
        test_cheat = -1.0
        test_accuracy_epoch = -1.0
        valid_accuracy_best = -1.0

        evaluate_accuracy = dic_functions['LDD']

        for epoch in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                label = attr[:, self.target_attr_idx]
                data_rot, label_rot = self.rotation_ssl_data(data, label)
                data = data.to(device)
                label = label.to(device)
                data_rot = data_rot.to(device)
                label_rot = label_rot.to(device)
                

                try:
                    z_b = []
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                    
                    z_d = []
                    hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_d))
                    _ = self.model_d(data)
                    hook_fn.remove()
                    z_d = z_d[0]
                    
                    if epoch == 1 and ix == 0:
                        print("[Average Pool layer Selected]")
                except:
                    z_b = []
                    hook_fn = self.model_b.layer4.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                    
                    z_d = []
                    hook_fn = self.model_d.layer4.register_forward_hook(self.concat_dummy(z_d))
                    _ = self.model_d(data)
                    hook_fn.remove()
                    z_d = z_d[0]
                    if epoch == 1 and ix == 0:
                        print("[Layer 4 Selected]")
                
                z_conflict = torch.cat((z_d, z_b.detach()), dim=1)
                z_align = torch.cat((z_d.detach(), z_b), dim=1)

                '''
                Rotation Starts
                '''

                try:
                    z_r = []
                    hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_r))
                    _ = self.model_d(data_rot)
                    hook_fn.remove()
                    z_r = z_r[0]
                except:
                    z_r = []
                    hook_fn = self.model_d.layer4.register_forward_hook(self.concat_dummy(z_r))
                    _ = self.model_d(data_rot)
                    hook_fn.remove()
                    z_r = z_r[0]
                z_r = self.model_mlp(z_r)
                loss_rotate = self.criterion(z_r, label_rot.long()).mean()


                '''
                Rotation Ends
                '''

                pred_conflict = self.model_d.fc(z_conflict)
                pred_align = self.model_b.fc(z_align)
                loss_dis_conflict = self.criterion(pred_conflict, label)
                loss_dis_align = self.criterion(pred_align, label)
                
                loss_dis_conflict = loss_dis_conflict.detach()
                loss_dis_align = loss_dis_align.detach()
                
                self.sample_loss_ema_d.update(loss_dis_conflict, index)
                self.sample_loss_ema_b.update(loss_dis_align, index)
                
                loss_dis_conflict = self.sample_loss_ema_d.parameter[index].clone().detach()
                loss_dis_align = self.sample_loss_ema_b.parameter[index].clone().detach()
                
                loss_dis_conflict = loss_dis_conflict.to(device)
                loss_dis_align = loss_dis_align.to(device)
                
                for c in range(self.num_classes):
                    class_index = torch.where(label == c)[0].to(device)
                    max_loss_conflict = self.sample_loss_ema_d.max_loss(c)
                    max_loss_align = self.sample_loss_ema_b.max_loss(c)
                    loss_dis_conflict[class_index] /= max_loss_conflict
                    loss_dis_align[class_index] /= max_loss_align
                
                loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)  
                loss_weight = loss_weight.to(device)
                
                loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight           
                loss_dis_align = self.bias_criterion(pred_align, label)
                
                if epoch >= 30:
                    indices = np.random.permutation(z_b.size(0))
                    z_b_swap = z_b[indices]         # z tilde
                    label_swap = label[indices]     # y tilde
                    z_mix_conflict = torch.cat((z_d, z_b_swap.detach()), dim=1)
                    z_mix_align = torch.cat((z_d.detach(), z_b_swap), dim=1)
                    pred_mix_conflict = self.model_d.fc(z_mix_conflict)
                    pred_mix_align = self.model_b.fc(z_mix_align)
                    loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight     
                    loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)                               
                    lambda_swap = self.lambda_swap_    
                else:
                    loss_swap_conflict = torch.tensor([0]).float()
                    loss_swap_align = torch.tensor([0]).float()
                    lambda_swap = 0
                
                loss_dis  = loss_dis_conflict.mean() + self.lambda_dis_align * loss_dis_align.mean()                # Eq.2 L_dis
                loss_swap = loss_swap_conflict.mean() + self.lambda_swap_align * loss_swap_align.mean()             # Eq.3 L_swap
                loss = loss_dis + lambda_swap * loss_swap + loss_rotate * self.loss_rotate_param
                self.optimizer_d.zero_grad()
                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_d.step()
                self.optimizer_b.step()
            self.schedulerb.step()
            self.schedulerd.step()
            
            train_accuracy_epoch = evaluate_accuracy(self.model_b, self.model_d, self.train_loader, self.target_attr_idx, device)
            prev_valid_accuracy = valid_accuracy_best
            valid_accuracy_epoch = evaluate_accuracy(self.model_b, self.model_d, self.valid_loader, self.target_attr_idx, device)
            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)
            
            print("[Epoch "+str(epoch)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")
            
            test_accuracy_epoch = evaluate_accuracy(self.model_b, self.model_d, self.test_loader, self.target_attr_idx, device)
            
            test_cheat = max(test_cheat, test_accuracy_epoch)
            
            print("[Test Accuracy cheat][%.4f]"%test_cheat)
            
            if valid_accuracy_best > prev_valid_accuracy:
                test_accuracy = test_accuracy_epoch
            
            print('[Best Test Accuracy]', test_accuracy)
        
        return test_accuracy, test_accuracy_epoch, test_cheat    

    def get_results(self, seed):
        set_seed(seed)
        print('[Training][{}]'.format(self.run_type))
        self.datasets()
        if self.reduce:
            self.reduce_data()
        self.dataloaders()
        self.models()
        self.optimizers()
        if self.run_type == 'LDD':
            a,b,c = self.train_LDD()
            self.store_results(a,b,c)
        elif self.run_type == 'LDD_MW':
            a,b,c = self.train_LDD_MW()
            self.store_results(a,b,c)
        elif self.run_type == 'LDD_rotation':
            a,b,c = self.train_LDD_rotation()
            self.store_results(a,b,c)
        else:
            print('Invalid run type')
            return