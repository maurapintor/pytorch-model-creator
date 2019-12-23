import logging

from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
import torch
import argparse

# local imports
from src.regularization import train_regularized
from src.subset_iterator import SubsetIterator
from src.utils import train, test
from src.models import mnist, cifar10

parser = argparse.ArgumentParser(description='Train base model.')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    help='Number of epochs for training (default: 10).')
parser.add_argument('--dataset',
                    help='Dataset to use for training (default: mnist).',
                    choices=['mnist', 'cifar10'],
                    default='mnist')
parser.add_argument('--nb_train',
                    help='Number of samples to use for training '
                         '(default: 10000).',
                    type=int,
                    default=10000)
parser.add_argument('--nb_test',
                    type=int,
                    help='Number of samples to use for testing '
                         '(default: 2000)',
                    default=2000)
parser.add_argument('--include_list',
                    help='Classes to include in the training process, '
                         'separated by commas (default: all).',
                    default='all')
parser.add_argument('--batch_size',
                    help='Batch size to use for training (default: 128).',
                    type=int,
                    default=128)
parser.add_argument('--lr',
                    help='Learning rate to use for training (default: 0.01).',
                    type=float,
                    default=0.01)
parser.add_argument('--scheduler_steps',
                    help='Learning rate scheduler steps, separated by '
                         'commas (default: None)',
                    default=None)
parser.add_argument('--scheduler_gamma',
                    help='Learning rate decay to apply at the milestones '
                         'defined in scheduler_steps (default: 0).',
                    type=float,
                    default=0)
parser.add_argument('--weight_decay',
                    help='Weight decay to use as regularization (default: 0).',
                    type=int,
                    default=0)
parser.add_argument('--momentum',
                    help='Momentum to use during training (default: 0).',
                    type=int,
                    default=0)
parser.add_argument('--robust',
                    default=False,
                    action='store_true',
                    help='Whether or not to train the model with double '
                         'backprop, regularizing the gradient size wrt the '
                         'input (default: False).')
parser.add_argument('--use_cuda',
                    action='store_true',
                    help='Use cuda if available (default: True).',
                    default=True)
parser.add_argument('--num_workers',
                    type=int,
                    default=0,
                    help='Number of additional workers '
                         'to spawn for the training process '
                         '(default: 0).')
parser.add_argument('--output_file',
                    default=None,
                    help='Name of the output file. The file will be stored '
                         'in the directory `pretrained_models`.')

args = parser.parse_args()

use_cuda = args.use_cuda and torch.cuda.is_available()
num_workers = args.num_workers
epochs = args.epochs
dataset = args.dataset
nb_train = args.nb_train
nb_test = args.nb_test
include_list = list(map(int, args.include_list.split(','))) \
    if args.include_list is not 'all' else range(10)
batch_size = args.batch_size
lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
scheduler_steps = args.scheduler_steps.split(',') \
    if args.scheduler_steps is not None else None
scheduler_gamma = args.scheduler_gamma \
    if args.scheduler_gamma is not 0 else None
robust = args.robust
output_file = args.output_file if args.output_file is not None else dataset

device = torch.device("cuda" if use_cuda and torch.cuda.is_available()
                      else "cpu")

dataset_kwargs = {
    'nb_train': nb_train, 'nb_test': nb_test,
    'include_list': include_list,
    'batch_size': batch_size,
}

train_loader, valid_loader, test_loader = \
    SubsetIterator(**dataset_kwargs) \
        .get_train_valid_test(valid_size=0,
                              dataset=dataset)

if dataset == 'mnist':
    model = mnist(pretrained=False,
                  n_hiddens=[256, 256]).to(device)
elif dataset == 'cifar10':
    model = cifar10(pretrained=False, n_channel=3).to(device)

optim_kwargs = {'lr': lr, 'momentum': momentum}

optimizer = optim.SGD(model.parameters(), **optim_kwargs)
if scheduler_steps is not None and scheduler_gamma is not 0:
    scheduler = MultiStepLR(optimizer,
                            milestones=scheduler_steps,
                            gamma=scheduler_gamma)
else:
    scheduler = None

loss_fn = nn.CrossEntropyLoss()
for epoch in range(1, epochs + 1):
    logging.info("Training epoch {}/{}".format(epoch, epochs + 1))
    if robust is False:
        train(model, device, train_loader, optimizer, epoch, loss_fn)
    else:
        train_regularized(model, device, train_loader, optimizer, epoch, loss_fn)
    if scheduler:
        scheduler.step(epoch)
    test(model, device, test_loader, epoch, loss_fn)

# todo lambda_const
torch.save(model.state_dict(), "pretrained_models/{}.pt".format(output_file))
