# TODO merge naive and weighted loss.
import torch
import torch.nn.functional as F


def weighted_nll_loss(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.nll_loss(pred, label, reduction='none')
    return torch.sum(raw * weight)[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor


def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor

def sigmoid_kldiv_focal_loss(pred,
                            target,
                            weight,
                            temperature=4.,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean'):
    '''
    sigmoid KLDiv focal loss (added by huangchuanhong)
    '''
    def kldiv(p, q, temperature):
        kl = (q * ((q + 1e-10).log() - (p + 1e-10).log()) + (1 - q) * ((1 - q + 1e-10).log() - (1 - p + 1e-10).log())) * (temperature ** 2)
        return kl
    pred_t_sigmoid = F.sigmoid(pred / temperature)
    target_t_sigmoid = F.sigmoid(target / temperature)
    kl = kldiv(pred_t_sigmoid, target_t_sigmoid, temperature)
    pt = (1 - pred_t_sigmoid) * target_t_sigmoid + pred_t_sigmoid * (1 - target_t_sigmoid)
    weight = (alpha * target_t_sigmoid + (1 - alpha) * (1 - target_t_sigmoid)) * weight
    weight = weight * pt.pow(gamma)
    loss = kl * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
    return loss

def weighted_sigmoid_kldiv_focal_loss(pred,
                                      target,
                                      weight,
                                      temperature=4.,
                                      gamma=2.0,
                                      alpha=0.25,
                                      teacher_alpha=1,
                                      avg_factor=None,
                                      num_classes=80):
    '''
    For a KD loss, weighted as focal loss(added by huangchuanhong)
    :param target: the teacher pred
    '''
    if teacher_alpha == 0.:
        return torch.zeros([]).cuda()
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_kldiv_focal_loss(
        pred, target, weight, temperature=temperature, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor * teacher_alpha

def sigmoid_kldiv(pred,
                  target,
                  weight,
                  temperature=4.,
                  reduction='mean'):
    '''
    sigmoid KLDiv loss (added by huangchuanhong)
    '''
    def kldiv(p, q, temperature):
        kl = (q * ((q + 1e-10).log() - (p + 1e-10).log()) + (1 - q) * ((1 - q + 1e-10).log() - (1 - p + 1e-10).log())) * (temperature ** 2)
        return kl
    pred_t_sigmoid = F.sigmoid(pred / temperature)
    target_t_sigmoid = F.sigmoid(target / temperature)
    kl = kldiv(pred_t_sigmoid, target_t_sigmoid, temperature)
    loss = kl * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
    return loss

def weighted_sigmoid_kldiv(pred,
                           target,
                           weight,
                           temperature=4,
                           teacher_alpha=1,
                           avg_factor=None,
                           num_classes=80):
    '''
        For a KD loss(added by huangchuanhong)
        :param target: the teacher pred
        '''
    if teacher_alpha == 0.:
        return torch.zeros([]).cuda()
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_kldiv(
        pred, target, weight, temperature=temperature,
        reduction='sum')[None] / avg_factor * teacher_alpha


def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res

#def attention_loss(xs, ys, beta):
#    '''
#    added by huangchuanhong
#    '''
#    def at(x):
#        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
#    def at_loss(x, y):
#        return (at(x) - at(y)).pow(2).mean()
#    return sum([at_loss(x, y) for x, y in zip(xs, ys)]) * beta

#def attention_loss(x, y, beta):
#    '''
#    added by huangchuanhong
#    '''
#    if beta == 0.:
#        return torch.zeros([]).cuda(),
#    def at(x):
#        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
#    def at_loss(x, y):
#        return (at(x) - at(y)).pow(2).mean()
#    return at_loss(x, y) * beta, 

def attention_loss(x, y, beta):
    '''
    added by huangchuanhong
    '''
    if beta == 0.:
        return torch.zeros([]).cuda(),
    def at(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    def at_loss(x, y):
        return (at(x) - at(y)).pow(2).sum()
    return at_loss(x, y) * beta * 0.5,
