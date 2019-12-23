from tqdm import tqdm
import torch


def train_regularized(model, device, train_loader, optimizer,
                      epoch, loss_fn, lambda_const=0.1):
    model.train()
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            optimizer.zero_grad()
            output = model(data)
            partial_loss = loss_fn(output, target)
            grad = torch.autograd.grad(partial_loss, data, retain_graph=True)[0]
            loss = partial_loss + lambda_const * torch.norm(grad)
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
