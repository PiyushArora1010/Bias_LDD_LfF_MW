import torch
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
    'set_seed': set_seed,
    'write_to_file': write_to_file
}