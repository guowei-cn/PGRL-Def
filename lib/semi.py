"""Modified from https://github.com/YU1ut/MixMatch-pytorch.
"""
import numpy as np
import torch
from tqdm import tqdm

from lib.dataLoader import get_batch_val


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch

    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    return [torch.cat(v, dim=0) for v in xy]

def mixmatch_train(model, xloader, uloader, criterion, optimizer, epoch, num_classes, batch_id, train_iteration, device, writer=None, debugging_flag=False):
    # xiter = iter(xloader)
    # uiter = iter(uloader)

    model.train()
    # for param in model.lstm.parameters():
    #     param.requires_grad = True
    for batch_idx, ubatch in enumerate(tqdm(uloader)):
        batch_id += 1
        xbatch = get_batch_val(xloader)
        xinput, xtarget = xbatch[0], xbatch[1]
        uinput1, uinput2 = ubatch[0], ubatch[1]

        xbatch_size = xinput.size(0)
        ubatch_size = uinput1.shape[0]
        batch_size = ubatch_size
        repeat_num = int(ubatch_size/xbatch_size) + 1
        xinput, xtarget = xinput.repeat([repeat_num]+[1 for _ in range(len(xinput.shape)-1)])[:ubatch_size], xtarget.repeat([repeat_num*xtarget.shape[0]]+[1 for _ in range(len(xtarget.shape)-1)])[:ubatch_size]

        xtarget = torch.zeros(xtarget.shape[0], num_classes).scatter_(
            1, xtarget.view(-1, 1).long(), 1
        )
        xinput = xinput.to(device)
        xtarget = xtarget.to(device)
        uinput1 = uinput1.to(device)
        uinput2 = uinput2.to(device)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            # uoutput1 = model(uinput1)
            # uoutput2 = model(uinput2)
            uoutput = model(torch.cat([uinput1, uinput2]))
            uoutput1, uoutput2 = uoutput[:uinput1.shape[0]], uoutput[uinput1.shape[0]:]
            p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2
            temperature = 0.5
            pt = p ** (1 / temperature)
            utarget = pt / pt.sum(dim=1, keepdim=True)
            utarget = utarget.detach()

        # mixup
        all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
        all_target = torch.cat([xtarget, utarget, utarget], dim=0)
        l = np.random.beta(0.75, 0.75)
        l = max(l, 1 - l)
        idx = torch.randperm(all_input.size(0))
        input_a, input_b = all_input, all_input[idx]
        target_a, target_b = all_target, all_target[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logit = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logit.append(model(input))

        # put interleaved samples back
        logit = interleave(logit, batch_size)
        xlogit = logit[0]
        ulogit = torch.cat(logit[1:], dim=0)

        Lx, Lu, lambda_u = criterion(
            xlogit,
            mixed_target[:batch_size],
            ulogit,
            mixed_target[batch_size:],
            epoch + batch_idx / train_iteration,
        )
        loss = Lx + lambda_u * Lu
        if writer != None:
            writer.add_scalar('mixmatch/loss', loss.item(), batch_id)
            writer.add_scalar('mixmatch/Lx', Lx.item(), batch_id)
            writer.add_scalar('mixmatch/Lu', Lu.item(), batch_id)
            writer.add_scalar('mixmatch/lambda_u', lambda_u, batch_id)

        # print('loss: {}'.format(loss.item()))
        # print('Lx: {}'.format(Lx.item()))
        # print('Lu: {}'.format(Lu.item()))
        # print('lambda_u: {}'.format(lambda_u))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema_optimizer.step()
        if debugging_flag:
            break

    return batch_id
