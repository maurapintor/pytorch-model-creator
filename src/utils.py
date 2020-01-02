from torchvision.models.resnet import BasicBlock
from tqdm import tqdm
import torch
from torch import nn


def get_layers(model):
    for name, layer in model._modules.items():
        # If it is a sequential, don't return its name
        # but recursively register all it's module children
        if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock):
            yield from [(":".join([name, l]), m) for (l, m) in get_layers(layer)]
        else:
            yield (name, layer)


def train(model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            t.set_postfix(
                epoch='{}'.format(epoch),
                completed='[{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)),
                loss='{:.4f}'.format(loss.item()))
            t.update()
    return


def test(model, device, test_loader, epoch, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                test_loss += loss.item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                t.set_postfix(
                    epoch='{}'.format(epoch),
                    completed='[{}/{} ({:.0f}%)]'.format(
                        batch_idx * len(data),
                        len(test_loader.dataset),
                        100. * batch_idx / len(test_loader)),
                    loss='{:.4f}'.format(loss.item()))
                t.update()

        test_loss /= len(test_loader.dataset)

    print('\n[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)
