import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import torch
from pymongo import MongoClient
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import argparse
import os

from src.subset_iterator import SubsetIterator
from src.utils import test
from src.models import mnist, cifar10

parser = argparse.ArgumentParser(description='Test the model')

parser.add_argument('--dataset',
                    help='Dataset to use for training (default: mnist).',
                    choices=['mnist', 'cifar10'],
                    default='mnist')
parser.add_argument('--nb_test',
                    type=int,
                    help='Number of samples to use for testing '
                         '(default: 2000)',
                    default=2000)
parser.add_argument('--include_list',
                    help='Classes to include in the training process, '
                         'separated by commas (default: all).',
                    default='all')
parser.add_argument('--robust',
                    action='store_true',
                    help='Whether to use the robust trained '
                         'model or the normal model. '
                         'Either robust or distilled could be '
                         'activated, but not both (default: False).',
                    default=False)
parser.add_argument('--distilled',
                    action='store_true',
                    help='Whether to use the distilled model or the '
                         'normal model. Either robust or distilled could be '
                         'activated, but not both (default: False).',
                    default=False)
parser.add_argument('--use_cuda',
                    action='store_true',
                    help='Use cuda if available (default: True).',
                    default=True)

args = parser.parse_args()

use_cuda = args.use_cuda and torch.cuda.is_available()
dataset = args.dataset
nb_test = args.nb_test
include_list = list(map(int, args.include_list.split(','))) \
    if args.include_list is not 'all' else range(10)
robust = args.robust
distilled = args.distilled

if robust and distilled:
    raise ValueError("Either robust or distilled could be active, not both. "
                     "They can be both disabled.")

with MongoClient('localhost', 27017) as client:
    db = client['sec-evals']
    collection = db['models']
    print({
            'dataset': dataset,
            'include_list': args.include_list,
            'distilled': distilled,
            'robust': robust})
    try:
        teacher_model_params = collection.find({
            'dataset': dataset,
            'include_list': args.include_list,
            'distilled': distilled,
            'robust': robust}).next()
    except StopIteration:
        raise ValueError("Pretrained model not found. Please train "
                         "model with the same parameters, using the "
                         "script `train_base_model.py` and the "
                         "same dataset and include_list as the "
                         "desired distilled model.")
    else:
        logging.info("Loading stored model: {}".format(teacher_model_params['_id']))

model = None
if dataset == 'mnist':
    model = mnist(pretrained=False, n_hiddens=[256, 256], n_class=len(include_list))
elif dataset == 'cifar10':
    model = cifar10(pretrained=False, n_channel=128, num_classes=len(include_list))

if teacher_model_params is not None:
    model.load_state_dict(torch.load(teacher_model_params['model_path']))

kwargs_data = {
    'nb_train': 1,
    'nb_test': nb_test,
    'include_list': include_list,
    'batch_size': 10,
}

device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

train_loader, valid_loader, test_loader = \
    SubsetIterator(**kwargs_data).get_train_valid_test(valid_size=0,
                                                       dataset=dataset)

loss_fn = torch.nn.CrossEntropyLoss()
acc = test(model, device, test_loader, 0, loss_fn)
print(acc)
