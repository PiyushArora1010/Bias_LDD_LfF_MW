import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
import numpy as np
import random
from numpy.random import RandomState

def evaluate_accuracy_LfF_mem(mw_model, test_loader, memory_loader, target_attr_idx, device):
  mw_model.eval()
  mw_correct = 0
  mem_iter = iter(memory_loader)
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)
        try:
            indexm,memory_input,_ = next(mem_iter)
        except:
            mem_iter = iter(memory_loader)
            indexm,memory_input,_ = next(mem_iter)
        memory_input = memory_input.to(device)

        mw_outputs = mw_model(data, memory_input)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy

def evaluate_accuracy_LfF(mw_model, test_loader, target_attr_idx, device, param = -1):
  mw_model.eval()
  mw_correct = 0
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)

        mw_outputs  = mw_model(data)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy

def evaluate_rotation(model, test_loader, device, param1 = -1, param2 = -1):
  model.eval()
  correct = 0
  with torch.no_grad():
    for _, images, labels in test_loader:
        labels = torch.zeros(len(labels))
        images_90 = TF.rotate(images, 90)
        labels_90 = torch.ones(len(labels))
        images_180 = TF.rotate(images, 180)
        labels_180 = torch.ones(len(labels))*2
        images_270 = TF.rotate(images, 270)
        labels_270 = torch.ones(len(labels))*3
        images = torch.cat((images, images_90, images_180, images_270), dim=0)
        labels = torch.cat((labels, labels_90, labels_180, labels_270), dim=0)
        images = images.to(device)
        labels = labels.to(device)
        del images_90, images_180, images_270, labels_90, labels_180, labels_270

        outputs  = model(images)
        pred = outputs.data.max(1, keepdim=True)[1]

        correct += pred.eq(labels.data.view_as(pred)).sum().item()
  accuracy = 100.*(torch.true_divide(correct,len(test_loader.dataset)*4)).item()
  model.train()
  return accuracy

def evaluate_accuracy_simple(mw_model, test_loader, target_attr_idx, device, param1 = -1):
  mw_model.eval()
  mw_correct = 0
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)

        mw_outputs  = mw_model(data)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy

criterion = nn.CrossEntropyLoss(reduction = 'none')

def weights_loss(model_d, model_b, datam, labelm, datam1, sample_loss_ema_b_mem, sample_loss_ema_d_mem):
        logit_b_mem = model_b(datam)
       
        logit_d_mem = model_d(datam, datam1,'Error',False )
      
        loss_b_mem = criterion(logit_b_mem, labelm).cpu().detach()
        loss_d_mem = criterion(logit_d_mem, labelm).cpu().detach()

        num_classes = len(np.unique(labelm.cpu().detach().numpy()))
        label_cpu_mem = labelm.cpu()
        for c in range(num_classes):
            class_index = np.where(label_cpu_mem == c)[0]
            max_loss_b_mem = sample_loss_ema_b_mem.max_loss(c)
            max_loss_d_mem = sample_loss_ema_d_mem.max_loss(c)
            loss_b_mem[class_index] /= max_loss_b_mem
            loss_d_mem[class_index] /= max_loss_d_mem

        loss_weight_mem = loss_b_mem / (loss_b_mem + loss_d_mem + 1e-8)
        loss_weight_mem = loss_weight_mem.detach()
        return loss_weight_mem

def evaluate_accuracy_approach(mw_model, model_b, test_loader, memory_loader, mem_loader_1, target_attr_idx, sample_loss_ema_b_mem, sample_loss_ema_d_mem,device):
  mw_model.eval()
  mw_correct = 0

  mem_iter = iter(memory_loader)
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)
        try:
            _,data_m_1, _ = next(mem_iter_)
        except:
            mem_iter_ = iter(mem_loader_1)
            _,data_m_1, _ = next(mem_iter_)
        try:
            indexm,memory_input,attrm = next(mem_iter)
            _,data_m_1, _ = next(mem_iter_)
        except:
            mem_iter = iter(memory_loader)
            indexm,memory_input,attrm = next(mem_iter)
            mem_iter_ = iter(mem_loader_1)
            _,data_m_1, _ = next(mem_iter_)
        data_m_1 = data_m_1.to(device)
        labelm = attrm[:,target_attr_idx].to(device)
        memory_input = memory_input.to(device)
        weights_mul = weights_loss(mw_model, model_b, memory_input, labelm, data_m_1, sample_loss_ema_b_mem, sample_loss_ema_d_mem)
        weights_mul = weights_mul.to(device)
        mw_outputs  = mw_model(data,memory_input, weights_mul, True)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy

class MultiDimAverageMeter(object):
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )
        
    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()

def evaluate_LfF_classwise(model, data_loader, attr_dims, target_attr_idx, bias_attr_idx, device):
        model.eval()
        acc = 0
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for index, data, attr in data_loader:
            label = attr[:, target_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

            attr = attr[:, [target_attr_idx, bias_attr_idx]]

            attrwise_acc_meter.add(correct.cpu(), attr.cpu())

        accs = attrwise_acc_meter.get_mean()

        model.train()

        return accs

def set_seed(seed: int) -> RandomState:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state


def write_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text)
        f.write('\n')

dic_functions = {
    'MW_LfF MW_LfF_Rotation MW': evaluate_accuracy_LfF_mem,
    'LfF LfF_Rotation': evaluate_accuracy_LfF,
    'Rotation': evaluate_rotation,
    'Simple': evaluate_accuracy_simple,
    'approach': evaluate_accuracy_approach,
    'set_seed': set_seed,
    'write_to_file': write_to_file,
    'class_LfF': evaluate_LfF_classwise
}