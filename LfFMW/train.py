import torch
import os
import argparse
from module.utils import dic_functions
from learner import trainer

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("--run_type", default="approach", help="Run Type")
parser.add_argument("--dataset_in",default="CMNIST", help="Name of the Dataset")
parser.add_argument("--model_in", default="resnet18_C", help="Name of the model")
parser.add_argument("--train_samples", default=1000, type=int,help="Number of training samples")
parser.add_argument("--bias_ratio", default=0.03, type = float,help="Bias ratio")
parser.add_argument("--runs", default=10, type=int, help="Number of runs")
args = parser.parse_args()

set_seed = dic_functions['set_seed']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

write_to_file = dic_functions['write_to_file']

run = trainer(args)

for run_num in range(args.runs):
    run.get_results(run_num)