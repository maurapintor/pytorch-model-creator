import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import argparse

from src.subset_iterator import SubsetIterator
from src.utils import test
from src.models import mnist


def get_teacher_output(teacher_model, device, data_loader):
    teacher_model.eval()
    teacher_outputs = torch.zeros((len(data_loader.dataset),
                                   len(data_loader.dataset.classes)), device=device)
    for i, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        teacher_outputs[i * len(data):(i + 1) * (len(data)), :] = teacher_model(data).data
    return teacher_outputs


def loss_fn_distill(outputs, labels, teacher_outputs, alpha, T):
    loss = torch.softmax(teacher_outputs / T, dim=1) * (alpha) + \
              nn.functional.cross_entropy(outputs, labels) * (1. - alpha)

    return loss


def train_kd(model, device, teacher_outputs, optimizer, loss_fn_kd, data_loader, alpha, T):
    model.train()
    with tqdm(total=len(data_loader)) as t:
        for i, (data, labels) in enumerate(data_loader):
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            teacher_output = teacher_outputs[i * len(data):(i + 1) * (len(data)), :]
            loss = loss_fn_kd(output, labels, teacher_output, alpha, T)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:05.3f}'.format(loss.item()))
            t.update()


parser = argparse.ArgumentParser(description='Train the distilled model.')

parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    help='Number of epochs for training (default: 10).')
parser.add_argument('--nb_train',
                    help='Number of samples to use for training '
                         '(default: 60000).',
                    type=int,
                    default=60000)
parser.add_argument('--nb_test',
                    type=int,
                    help='Number of samples to use for testing '
                         '(default: 10000)',
                    default=10000)
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
                    type=float,
                    default=0)
parser.add_argument('--momentum',
                    help='Momentum to use during training (default: 0).',
                    type=float,
                    default=0)
parser.add_argument('--nesterov',
                    default=False,
                    action='store_true',
                    help='Nesterov for SGD (default: False).')
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
                         'in the directory `pretrained_models`. Will be saved '
                         'with the suffix "distilled". ')
parser.add_argument('--alpha',
                    type=float,
                    default=0.1,
                    help='Weight for the teacher loss. If alpha is equal '
                         'to 1, the loss is computed using only the soft labels '
                         'coming from the teacher. If alpha is 0, '
                         'the loss is computed with hard-labels only. If the '
                         'value of alpha is between 0 and 1, the loss will '
                         'incorporate both terms. (default: 0.1)')
parser.add_argument('--temperature',
                    type=float,
                    default=1,
                    help='Temperature of the softmax for the output scores '
                         'of the teacher model (default: 1).')
parser.add_argument('--teacher',
                    type=str,
                    required=True,
                    help='Path of the pretrained teacher parameters.')

args = parser.parse_args()

use_cuda = args.use_cuda and torch.cuda.is_available()
num_workers = args.num_workers
epochs = args.epochs
dataset = "mnist"
nb_train = args.nb_train
nb_test = args.nb_test
include_list = list(map(int, args.include_list.split(','))) \
    if args.include_list is not 'all' else range(10)
batch_size = args.batch_size
lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
nesterov = args.nesterov
scheduler_steps = args.scheduler_steps.split(',') \
    if args.scheduler_steps is not None else None
scheduler_gamma = args.scheduler_gamma \
    if args.scheduler_gamma != 0 else None
alpha = args.alpha
temperature = args.temperature
output_file = args.output_file if args.output_file is not None else dataset
model_path = args.teacher

teacher_model_params = {'model_path': model_path}

kwargs_optim_dist = {
    'lr': lr,
}

device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


model = None
if dataset == 'mnist':
    model = mnist().to(device)

if teacher_model_params is not None:
    model.load_state_dict(torch.load(teacher_model_params['model_path']))

kwargs_data = {
    'nb_train': nb_train,
    'nb_test': nb_test,
    'include_list': include_list,
    'batch_size': batch_size,
}


train_loader, valid_loader, test_loader = \
    SubsetIterator(**kwargs_data).get_train_valid_test(valid_size=0,
                                                       dataset=dataset,
                                                       num_workers=num_workers)

teacher_outputs = get_teacher_output(model, device, train_loader)

distilled = None
if dataset == 'mnist':
    distilled = mnist().to(device)

optim_kwargs = {'lr': lr, 'momentum': momentum, 'nesterov': nesterov}

optimizer = optim.SGD(distilled.parameters(), **optim_kwargs)
if scheduler_steps is not None and scheduler_gamma != 0:
    scheduler = MultiStepLR(optimizer,
                            milestones=scheduler_steps,
                            gamma=scheduler_gamma)
else:
    scheduler = None

loss_fn = nn.CrossEntropyLoss()
acc = test(model, device, test_loader, 0, loss_fn)
for e in range(1, epochs + 1):
    train_kd(distilled, device, teacher_outputs, optimizer, loss_fn_distill, train_loader, alpha, temperature)
    if scheduler:
        scheduler.step(e)
    acc = test(distilled, device, test_loader, e, loss_fn)

model_name = "pretrained_models/{}_distilled.pt".format(output_file)
torch.save(distilled.state_dict(), model_name)
