import torch
import argparse
from module.utils import dic_functions
from learner import trainer

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("--run_type", default="LDD", help="Run Type")
parser.add_argument("--dataset_in",default="CMNIST", help="Name of the Dataset")
parser.add_argument("--model_in", default="resnet18_C", help="Name of the model")
parser.add_argument("--batch_size", default=250, help="Batch size")
parser.add_argument("--train_samples", default=1000, type=int,help="Number of training samples")
parser.add_argument("--bias_ratio", default=0.03, type = float,help="Bias ratio")
parser.add_argument("--seed", default=0, type = int,help="Seed")
args = parser.parse_args()

set_seed = dic_functions['set_seed']
set_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

write_to_file = dic_functions['write_to_file']

run = trainer(args)

run.get_results()