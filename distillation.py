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


def get_teacher_output(teacher_model, device, data_loader):
    teacher_model.eval()
    teacher_outputs = torch.zeros((len(data_loader.dataset),
                                   len(data_loader.dataset.classes)), device=device)
    for i, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        teacher_outputs[i * len(data):(i + 1) * (len(data)), :] = teacher_model(data).data
    return teacher_outputs


def loss_fn_kd(outputs, labels, teacher_outputs, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss()(torch.log_softmax(outputs / T, dim=1),
                             torch.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              nn.functional.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


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
                         'incorporate both terms.')
parser.add_argument('--temperature',
                    type=float,
                    default=1,
                    help='Temperature of the softmax for the output scores '
                         'of the teacher model (default: 1).')

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
alpha = args.alpha
temperature = args.temperature
output_file = args.output_file if args.output_file is not None else dataset

teacher_model_params = None
with MongoClient('localhost', 27017) as client:
    db = client['sec-evals']
    collection = db['models']

    try:
        teacher_model_params = collection.find({
            'dataset': dataset,
            'include_list': args.include_list}).next()
    except StopIteration:
        raise ValueError("Pretrained model not found. Please train "
                         "model with the same parameters, using the "
                         "script `train_base_model.py` and the "
                         "same dataset and include_list as the "
                         "desired distilled model.")
    else:
        logging.info("Loading stored model: {}".format(teacher_model_params['_id']))


kwargs_optim_dist = {
    'lr': lr,
}

model = None
if dataset == 'mnist':
    model = mnist(pretrained=False, n_hiddens=[256, 256], n_class=len(include_list))
elif dataset == 'cifar10':
    model = cifar10(pretrained=False, n_channel=128, num_classes=len(include_list))

if teacher_model_params is not None:
    model.load_state_dict(torch.load(teacher_model_params['model_path']))

kwargs_data = {
    'nb_train': nb_train,
    'nb_test': nb_test,
    'include_list': include_list,
    'batch_size': batch_size,
}

device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

train_loader, valid_loader, test_loader = \
    SubsetIterator(**kwargs_data).get_train_valid_test(valid_size=0,
                                                       dataset=dataset,
                                                       num_workers=num_workers)

teacher_outputs = get_teacher_output(model, device, train_loader)

distilled = None
if dataset == 'mnist':
    distilled = mnist(n_hiddens=[256, 256], n_class=len(include_list)).to(device)
elif dataset == 'cifar10':
    distilled = cifar10(n_channel=128, num_classes=len(include_list)).to(device)

optim_kwargs = {'lr': lr, 'momentum': momentum}

optimizer = optim.SGD(distilled.parameters(), **optim_kwargs)
if scheduler_steps is not None and scheduler_gamma is not 0:
    scheduler = MultiStepLR(optimizer,
                            milestones=scheduler_steps,
                            gamma=scheduler_gamma)
else:
    scheduler = None

loss_fn = nn.CrossEntropyLoss()
acc = test(model, device, test_loader, 0, loss_fn)
for e in range(1, epochs + 1):
    train_kd(distilled, device, teacher_outputs, optimizer, loss_fn_kd, train_loader, alpha, temperature)
    if scheduler:
        scheduler.step(e)
    acc = test(distilled, device, test_loader, e, loss_fn)

model_name = "pretrained_models/{}_distilled.pt".format(output_file)
torch.save(distilled.state_dict(), model_name)

# store arguments as mongodb object
args_dict = args.__dict__
args_dict['model_path'] = os.path.abspath(model_name)
args_dict['distilled'] = True
args_dict['final_acc'] = acc
# robust model can be trained with the other script
args_dict['robust'] = False

with MongoClient('localhost', 27017) as client:
    db = client['sec-evals']
    collection = db['models']

    # check if another model exists
    try:
        already_stored = collection.find({
            'dataset': dataset,
            'include_list': args.include_list,
            'distilled': True}).next()
    except StopIteration:
        already_stored = False

    if already_stored is False:
        model_id = collection.insert_one(args_dict)
    else:
        if already_stored and already_stored['final_acc'] < acc:
            # replace model
            logging.info("Found already stored model with lower accuracy. "
                         "Removed old model in favor of the newly "
                         "trained.")
            model_id = collection.replace_one({'_id': already_stored['_id']},
                                              args_dict)
        else:
            # keep only best model
            logging.info("Found already stored model with better accuracy. "
                         "Keeping best result.")
            model_id = already_stored['_id']

logging.info("Model stored: {}".format(model_id))

