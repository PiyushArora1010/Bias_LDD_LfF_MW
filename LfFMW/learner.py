from module.loss import GeneralizedCELoss,EMA
from module.models import dic_models
from module.models2 import dic_models_2
from data.util import get_dataset, IdxDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from module.utils import dic_functions
from config import dataloader_confg
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from module.resnets_vision import dic_models_3
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
        self.seed = args.seed
        set_seed(self.seed)
        self.target_attr_idx = 0
        self.bias_attr_idx = 1
        if 'CelebA' in self.dataset_in:
            self.target_attr_idx = 9
            self.bias_attr_idx = 20
            self.num_classes = 2
        else:
            self.num_classes = 10
    
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
        nums_train_biased = np.random.choice(indices_train_biased, int(self.train_samples - self.bias_ratio * self.train_samples) , replace=False)
        indices_train_unbiased = self.train_dataset.attr[:,self.target_attr_idx] != self.train_dataset.attr[:,self.bias_attr_idx]
        indices_train_unbiased = indices_train_unbiased.nonzero().squeeze()
        nums_train_unbiased = np.random.choice(indices_train_unbiased, int(self.bias_ratio * self.train_samples) , replace=False)
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
        
        print("[Size of the Dataset]["+str(len(self.train_dataset))+"]")
        print("[Conflicting Samples in Training Data]["+str(len(self.train_dataset.attr[self.train_dataset.attr[:,self.target_attr_idx] != self.train_dataset.attr[:,self.bias_attr_idx]]))+"]")
        print("[Conflicting Samples in Validation Data]["+str(len(self.valid_dataset.attr[self.valid_dataset.attr[:,self.target_attr_idx] != self.valid_dataset.attr[:,self.bias_attr_idx]]))+"]")
        print("[Conflicting Samples in Test Data]["+str(len(self.test_dataset.attr[self.test_dataset.attr[:,self.target_attr_idx] != self.test_dataset.attr[:,self.bias_attr_idx]]))+"]")
        
        self.train_target_attr = self.train_dataset.attr[:, self.target_attr_idx]
        self.train_bias_attr = self.train_dataset.attr[:, self.bias_attr_idx]

        self.train_dataset = IdxDataset(self.train_dataset)
        self.valid_dataset = IdxDataset(self.valid_dataset)    
        self.test_dataset = IdxDataset(self.test_dataset)

    def dataloaders(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            **self.loader_config['train'],)

        self.test_loader = DataLoader(
            self.test_dataset,
            **self.loader_config['test'],)

        self.valid_loader = DataLoader(
            self.valid_dataset,
            **self.loader_config['valid'],)
        
        if 'approach' in self.run_type:
            self.mem_loader = DataLoader(
                self.train_dataset,
                **self.loader_config['memory'],)
            self.mem_loader_ = DataLoader(
                self.train_dataset,
                **self.loader_config['memory'],)

        elif 'MW' in self.run_type:
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
        elif 'approach' in self.run_type:
            try:
                self.model_d = dic_models[self.model_in+'_approach'](self.num_classes).to(device)
                self.model_b = dic_models[self.model_in](self.num_classes).to(device)
            except:
                try:
                    self.model_d = dic_models_2[self.model_in+'_approach'](self.num_classes).to(device)
                    self.model_b = dic_models_2[self.model_in](self.num_classes).to(device)
                except:
                    self.model_d = dic_models_3[self.model_in+'_approach'](self.num_classes).to(device)
                    self.model_b = dic_models_3[self.model_in](self.num_classes).to(device)
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
        print("[MODEL]["+self.model_in+"]")
    
    def optimizers(self):
        if 'MNIST' in self.dataset_in:
            self.optimizer_b = torch.optim.Adam(self.model_b.parameters(),lr= 0.002, weight_decay=0.0)
            self.optimizer_d = torch.optim.Adam(self.model_d.parameters(),lr= 0.002, weight_decay=0.0)
            self.schedulerd = MultiStepLR(self.optimizer_d, milestones=[300], gamma=0.5)
            self.schedulerb = MultiStepLR(self.optimizer_b, milestones=[300], gamma=0.5)
            self.epochs = 200
        elif self.dataset_in == 'CelebA':
            self.optimizer_b = torch.optim.Adam(self.model_b.parameters(),lr= 1e-4, weight_decay=1e-4)
            self.optimizer_d = torch.optim.Adam(self.model_d.parameters(),lr= 1e-4, weight_decay=1e-4)
            self.schedulerd = MultiStepLR(self.optimizer_d, milestones=[300], gamma=0.5)
            self.schedulerb = MultiStepLR(self.optimizer_b, milestones=[300], gamma=0.5)
            self.epochs = 200
        else:
            self.optimizer_b = torch.optim.SGD(self.model_b.parameters(),lr= 0.1, weight_decay=5e-4, momentum = 0.9, nesterov = True)
            self.optimizer_d = torch.optim.SGD(self.model_d.parameters(),lr= 0.1, weight_decay=5e-4, momentum = 0.9, nesterov = True)
            self.schedulerd = MultiStepLR(self.optimizer_d, milestones=[150,225], gamma=0.1)
            self.schedulerb = MultiStepLR(self.optimizer_b, milestones=[150,225], gamma=0.1)
            self.epochs = 300
        
        print("[OPTIMIZER]["+str(self.optimizer_d)+"]")

        print("[EPOCHS]["+str(self.epochs)+"]")
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        self.criterion = criterion.to(device)
        bias_criterion = GeneralizedCELoss()
        self.bias_criterion = bias_criterion.to(device)

        self.sample_loss_ema_b = EMA(torch.LongTensor(self.train_target_attr), alpha=0.7)
        self.sample_loss_ema_d = EMA(torch.LongTensor(self.train_target_attr), alpha=0.7)

        if 'approach' in self.run_type:
            self.sample_loss_ema_b_mem = EMA(torch.LongTensor(self.train_target_attr), alpha=0.7)
            self.sample_loss_ema_d_mem = EMA(torch.LongTensor(self.train_target_attr), alpha=0.7)
    
    def train_LfF(self):
        scaler = GradScaler()
        test_accuracy = -1.0
        test_cheat = -1.0
        test_accuracy_epoch = -1.0
        valid_accuracy_best = -1.0

        evaluate_accuracy = dic_functions['LfF LfF_Rotation']

        for step in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                data = data.to(device)
                attr = attr.to(device)
                
                label = attr[:, self.target_attr_idx]
                bias_label = attr[:, self.bias_attr_idx]
                
                with autocast():
                    logit_b = self.model_b(data)
                    logit_d = self.model_d(data)
                    loss_b = self.criterion(logit_b, label)
                    loss_d = self.criterion(logit_d, label)
                
                loss_b = loss_b.cpu().detach()
                loss_d = loss_d.cpu().detach()
                
                self.sample_loss_ema_b.update(loss_b, index)
                self.sample_loss_ema_d.update(loss_d, index)
                
                loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
                loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()
                
                label_cpu = label.cpu()
                
                for c in range(self.num_classes):
                    class_index = np.where(label_cpu == c)[0]
                    max_loss_b = self.sample_loss_ema_b.max_loss(c)
                    max_loss_d = self.sample_loss_ema_d.max_loss(c)
                    loss_b[class_index] /= max_loss_b
                    loss_d[class_index] /= max_loss_d
                
                loss_weight = loss_b / (loss_b + loss_d + 1e-8)
                loss_weight = loss_weight.to(device)
                with autocast():
                    loss_b_update = self.bias_criterion(logit_b, label)
                    loss_d_update = self.criterion(logit_d, label) * loss_weight
                    loss = loss_b_update.mean() + loss_d_update.mean()
                
                self.optimizer_b.zero_grad()
                self.optimizer_d.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_b)
                scaler.step(self.optimizer_d)
                scaler.update()

            self.schedulerb.step()
            self.schedulerd.step()
            
            train_accuracy_epoch = evaluate_accuracy(self.model_d, self.train_loader, self.target_attr_idx, device)
            prev_valid_accuracy = valid_accuracy_best
            valid_accuracy_epoch = evaluate_accuracy(self.model_d, self.valid_loader, self.target_attr_idx, device)
            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)
            
            print("[Epoch "+str(step)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")
            
            test_accuracy_epoch = evaluate_accuracy(self.model_d, self.test_loader, self.target_attr_idx, device)
            
            test_cheat = max(test_cheat, test_accuracy_epoch)
            
            print("[Test Accuracy cheat][%.4f]"%test_cheat)
            
            if valid_accuracy_best > prev_valid_accuracy:
                test_accuracy = test_accuracy_epoch
            
            print('[Best Test Accuracy]', test_accuracy)
        
        return test_accuracy, test_accuracy_epoch, test_cheat
    
    def train_MW_LfF(self):
        scaler = GradScaler()
        test_accuracy = -1.0
        test_cheat = -1.0
        test_accuracy_epoch = -1.0
        valid_accuracy_best = -1.0
        memory_loader_iter = iter(self.mem_loader)
        evaluate_accuracy = dic_functions['MW_LfF MW_LfF_Rotation MW']

        for step in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                data = data.to(device)
                attr = attr.to(device)

                try:
                    mem_index, mem_data, mem_attr = next(memory_loader_iter)
                except:
                    memory_loader_iter = iter(self.mem_loader)
                    mem_index, mem_data, mem_attr = next(memory_loader_iter)

                mem_data = mem_data.to(device)

                label = attr[:, self.target_attr_idx]
                bias_label = attr[:, self.bias_attr_idx]
                
                with autocast():
                    logit_b = self.model_b(data)
                    logit_d, content_weights = self.model_d(data, mem_data, True)
                    loss_b = self.criterion(logit_b, label)
                    loss_d = self.criterion(logit_d, label)
                
                loss_b = loss_b.cpu().detach()
                loss_d = loss_d.cpu().detach()

                self.sample_loss_ema_b.update(loss_b, index)
                self.sample_loss_ema_d.update(loss_d, index)
                
                loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
                loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()
                
                label_cpu = label.cpu()
                
                for c in range(self.num_classes):
                    class_index = np.where(label_cpu == c)[0]
                    max_loss_b = self.sample_loss_ema_b.max_loss(c)
                    max_loss_d = self.sample_loss_ema_d.max_loss(c)
                    loss_b[class_index] /= max_loss_b
                    loss_d[class_index] /= max_loss_d
                
                loss_weight = loss_b / (loss_b + loss_d + 1e-8)
                loss_weight = loss_weight.to(device)
                with autocast():
                    loss_b_update = self.bias_criterion(logit_b, label)
                    loss_d_update = self.criterion(logit_d, label) * loss_weight
                    loss = loss_b_update.mean() + loss_d_update.mean()
                
                self.optimizer_b.zero_grad()
                self.optimizer_d.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_b)
                scaler.step(self.optimizer_d)
                scaler.update()

            self.schedulerb.step()
            self.schedulerd.step()
            
            train_accuracy_epoch = evaluate_accuracy(self.model_d, self.train_loader, self.mem_loader, self.target_attr_idx, device)
            prev_valid_accuracy = valid_accuracy_best
            valid_accuracy_epoch = evaluate_accuracy(self.model_d, self.valid_loader, self.mem_loader, self.target_attr_idx, device)
            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)
            
            print("[Epoch "+str(step)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")
            
            test_accuracy_epoch = evaluate_accuracy(self.model_d, self.test_loader, self.mem_loader, self.target_attr_idx, device)
            
            test_cheat = max(test_cheat, test_accuracy_epoch)
            
            print("[Test Accuracy cheat][%.4f]"%test_cheat)
            
            if valid_accuracy_best > prev_valid_accuracy:
                test_accuracy = test_accuracy_epoch
            
            print('[Best Test Accuracy]', test_accuracy)
        
        return test_accuracy, test_accuracy_epoch, test_cheat

    def train_simple(self):
        scaler = GradScaler()
        test_accuracy = -1.0
        test_cheat = -1.0
        test_accuracy_epoch = -1.0
        valid_accuracy_best = -1.0

        evaluate_accuracy = dic_functions['Simple']

        for step in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                data = data.to(device)
                attr = attr.to(device)
                
                label = attr[:, self.target_attr_idx]
                bias_label = attr[:, self.bias_attr_idx]
                
                with autocast():
                    logit_d = self.model_d(data)
                    loss = torch.mean(self.criterion(logit_d, label))
                
                self.optimizer_d.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_d)
                scaler.update()
            self.schedulerd.step()
            
            train_accuracy_epoch = evaluate_accuracy(self.model_d, self.train_loader, self.target_attr_idx, device)
            prev_valid_accuracy = valid_accuracy_best
            valid_accuracy_epoch = evaluate_accuracy(self.model_d, self.valid_loader, self.target_attr_idx, device)
            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)
            
            print("[Epoch "+str(step)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")
            
            test_accuracy_epoch = evaluate_accuracy(self.model_d, self.test_loader, self.target_attr_idx, device)
            
            test_cheat = max(test_cheat, test_accuracy_epoch)
            
            print("[Test Accuracy cheat][%.4f]"%test_cheat)
            
            if valid_accuracy_best > prev_valid_accuracy:
                test_accuracy = test_accuracy_epoch
            
            print('[Best Test Accuracy]', test_accuracy)
        
        return test_accuracy, test_accuracy_epoch, test_cheat

    def train_MW(self):
        scaler = GradScaler()
        test_accuracy = -1.0
        test_cheat = -1.0
        test_accuracy_epoch = -1.0
        valid_accuracy_best = -1.0
        mem_iter = None
        evaluate_accuracy = dic_functions['MW_LfF MW_LfF_Rotation MW']

        for step in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                data = data.to(device)
                attr = attr.to(device)

                try:
                    _, datam, _ = next(mem_iter)
                except:
                    mem_iter = iter(self.mem_loader)
                    _, datam, _ = next(mem_iter)
                
                datam = datam.to(device)
                label = attr[:, self.target_attr_idx]
                bias_label = attr[:, self.bias_attr_idx]
                
                with autocast():
                    logit_d = self.model_d(data, datam)
                    loss = torch.mean(self.criterion(logit_d, label))
                
                self.optimizer_d.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_d)
                scaler.update()
            
            self.schedulerd.step()
            
            train_accuracy_epoch = evaluate_accuracy(self.model_d, self.train_loader, self.mem_loader, self.target_attr_idx, device)
            prev_valid_accuracy = valid_accuracy_best
            valid_accuracy_epoch = evaluate_accuracy(self.model_d, self.valid_loader, self.mem_loader, self.target_attr_idx, device)
            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)
            
            print("[Epoch "+str(step)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")
            
            test_accuracy_epoch = evaluate_accuracy(self.model_d, self.test_loader, self.mem_loader, self.target_attr_idx,  device)
            
            test_cheat = max(test_cheat, test_accuracy_epoch)
            
            print("[Test Accuracy cheat][%.4f]"%test_cheat)
            
            if valid_accuracy_best > prev_valid_accuracy:
                test_accuracy = test_accuracy_epoch
            
            print('[Best Test Accuracy]', test_accuracy)
        
        return test_accuracy, test_accuracy_epoch, test_cheat

    def train_approach(self):
        
        scaler = GradScaler()
        test_accuracy = -1.0
        test_cheat = -1.0
        test_accuracy_epoch = -1.0
        valid_accuracy_best = -1.0
        memory_loader_iter = None
        memory_loader_iter_ = None
        evaluate_accuracy = dic_functions['approach']

        for step in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                data = data.to(device)
                attr = attr.to(device)

                try:
                    indexm, mem_data, mem_attr = next(memory_loader_iter)
                    _, mem_data_, _ = next(memory_loader_iter_)
                except:
                    memory_loader_iter = iter(self.mem_loader)
                    memory_loader_iter_ = iter(self.mem_loader_)
                    indexm, mem_data, mem_attr = next(memory_loader_iter)
                    _, mem_data_, _ = next(memory_loader_iter_)

                mem_data = mem_data.to(device)
                mem_data_ = mem_data_.to(device)

                label = attr[:, self.target_attr_idx]
                labelm = mem_attr[:, self.target_attr_idx]
                labelm = labelm.to(device)
                bias_label = attr[:, self.bias_attr_idx]
              
                # print(torch.unique(labelm))
                # print(torch.unique(label))
                '''
                Memory Update
                '''
                with autocast():
                    logit_b_mem = self.model_b(mem_data)
                    logit_d_mem = self.model_d(mem_data, mem_data_, 'Error', False)
                    loss_b_mem = self.criterion(logit_b_mem, labelm)
                    loss_d_mem = self.criterion(logit_d_mem, labelm)
                
                # logit_b_mem = self.model_b(mem_data)
                # logit_d_mem = self.model_d(mem_data, mem_data_, 'Error', False)
                # loss_b_mem = self.criterion(logit_b_mem, labelm)
                # loss_d_mem = self.criterion(logit_d_mem, labelm)
                
                loss_b_mem = loss_b_mem.cpu().detach()
                loss_d_mem = loss_d_mem.cpu().detach()

                self.sample_loss_ema_b_mem.update(loss_b_mem, indexm)
                self.sample_loss_ema_d_mem.update(loss_d_mem, indexm)
                loss_b_mem = self.sample_loss_ema_b_mem.parameter[indexm].clone().detach()
                loss_d_mem = self.sample_loss_ema_d_mem.parameter[indexm].clone().detach()

                label_cpu_mem = labelm.cpu()

                for c in range(self.num_classes):
                    class_index = np.where(label_cpu_mem == c)[0]
                    max_loss_b_mem = self.sample_loss_ema_b_mem.max_loss(c)
                    max_loss_d_mem = self.sample_loss_ema_d_mem.max_loss(c)
                    loss_b_mem[class_index] /= max_loss_b_mem
                    loss_d_mem[class_index] /= max_loss_d_mem

                loss_weight_mem = loss_b_mem / (loss_b_mem + loss_d_mem + 1e-8)
                loss_weight_mem = loss_weight_mem.detach()
                loss_weight_mem = loss_weight_mem.to(device)
                '''
                Memory Update ends
                '''
                with autocast():
                    logit_b = self.model_b(data)
                    logit_d = self.model_d(data, mem_data, loss_weight_mem, True)
                    loss_b = self.criterion(logit_b, label)
                    loss_d = self.criterion(logit_d, label)
                
                # logit_b = self.model_b(data)
                # logit_d = self.model_d(data, mem_data, loss_weight_mem, True)
                # loss_b = self.criterion(logit_b, label)
                # loss_d = self.criterion(logit_d, label)

                loss_b = loss_b.cpu().detach()
                loss_d = loss_d.cpu().detach()

                self.sample_loss_ema_b.update(loss_b, index)
                self.sample_loss_ema_d.update(loss_d, index)
                
                loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
                loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()
                
                label_cpu = label.cpu()
                
                for c in range(self.num_classes):
                    class_index = np.where(label_cpu == c)[0]
                    max_loss_b = self.sample_loss_ema_b.max_loss(c)
                    max_loss_d = self.sample_loss_ema_d.max_loss(c)
                    loss_b[class_index] /= max_loss_b
                    loss_d[class_index] /= max_loss_d
                
                loss_weight = loss_b / (loss_b + loss_d + 1e-8)
                loss_weight = loss_weight.to(device)

                with autocast():
                    loss_b_update = self.bias_criterion(logit_b, label)
                    loss_d_update = self.criterion(logit_d, label) * loss_weight
                    loss = loss_b_update.mean() + loss_d_update.mean()
                
                # loss_b_update = self.bias_criterion(logit_b, label)
                # loss_d_update = self.criterion(logit_d, label) * loss_weight
                # loss = loss_b_update.mean() + loss_d_update.mean()

                self.optimizer_b.zero_grad()
                self.optimizer_d.zero_grad()

                scaler.scale(loss).backward()
                
                scaler.step(self.optimizer_b)
                scaler.step(self.optimizer_d)
                
                scaler.update()
                # loss.backward()
                # self.optimizer_b.step()
                # self.optimizer_d.step()

            self.schedulerb.step()
            self.schedulerd.step()
            
            train_accuracy_epoch = evaluate_accuracy(self.model_d, self.model_b, self.train_loader, self.mem_loader, self.mem_loader_,self.target_attr_idx, self.sample_loss_ema_b_mem, self.sample_loss_ema_d_mem, device)
            prev_valid_accuracy = valid_accuracy_best
            valid_accuracy_epoch = evaluate_accuracy(self.model_d, self.model_b, self.valid_loader, self.mem_loader,self.mem_loader_, self.target_attr_idx,self.sample_loss_ema_b_mem, self.sample_loss_ema_d_mem, device)
            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)
            
            print("[Epoch "+str(step)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")
            
            test_accuracy_epoch = evaluate_accuracy(self.model_d, self.model_b, self.test_loader, self.mem_loader, self.mem_loader_,self.target_attr_idx,self.sample_loss_ema_b_mem, self.sample_loss_ema_d_mem, device)
            
            test_cheat = max(test_cheat, test_accuracy_epoch)
            
            print("[Test Accuracy cheat][%.4f]"%test_cheat)
            
            if valid_accuracy_best > prev_valid_accuracy:
                test_accuracy = test_accuracy_epoch
            
            print('[Best Test Accuracy]', test_accuracy)
        
        return test_accuracy, test_accuracy_epoch, test_cheat

    def get_results(self):
        print('[Training][{}]'.format(self.run_type))
        self.datasets()
        self.reduce_data()
        self.dataloaders()
        self.models()
        self.optimizers()
        if self.run_type == 'simple':
            a,b,c = self.train_simple()
            self.store_results(a,b,c)
        elif self.run_type == 'MW':
            a,b,c = self.train_MW()
            self.store_results(a,b,c)
        elif self.run_type == 'MW_LfF':
            a,b,c = self.train_MW_LfF()
            self.store_results(a,b,c)
        elif self.run_type == 'LfF':
            a,b,c = self.train_LfF()
            self.store_results(a,b,c)
        elif self.run_type == 'approach':
            a,b,c = self.train_approach()
            self.store_results(a,b,c)
        else:
            print('Invalid run type')
            return