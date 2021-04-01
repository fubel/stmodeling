import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import SequentialSampler

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
from ops import plot_utils
import datasets_video

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    check_rootfolders()

    categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset,
                                                                                                       args.modality)
    num_class = len(categories)

    args.store_name = '_'.join(['STModeling', args.dataset, args.modality, args.arch, args.consensus_type,
                                'segment%d' % args.num_segments])
    print('storing name: ' + args.store_name)

    model = TSN(num_class, args)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation()

    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # best_prec1 = 0
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    print(model)
    cudnn.benchmark = True

    # Data loading code
    if ((args.modality != 'RGBDiff') | (args.modality != 'RGBFlow')):
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    elif args.modality == 'RGBFlow':
        data_length = args.num_motion

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   dataset=args.dataset,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3']),
                             isRGBFlow=(args.modality == 'RGBFlow')),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   dataset=args.dataset,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3']),
                             isRGBFlow=(args.modality == 'RGBFlow')),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.consensus_type == 'DNDF':
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, '%s.csv' % args.store_name), 'w')
    for epoch in range(args.start_epoch, args.epochs):
        if not args.consensus_type == 'DNDF':
            adjust_learning_rate(optimizer, epoch, args.lr_steps)

        history = {
            'accuracy': [0.5,0.6],
            'val_accuracy': [0.2,0.3],
            'loss': [0.9,0.8],
            'val_loss': [0.8,0.7]
        }
        model_details = {
            'backbone': args.arch,
            'transformer_arch': args.consensus_type,
            'lr': args.lr,
            'batch_size': args.batch_size
        }
        plot_utils.plot_statistics(history,model_details)
        
        # train for one epoch
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, log_training)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, val_acc , val_loss = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
        
        history['accuracy'].append(acc)
        history['loss'].append(loss)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)

        

def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # compute output
        output = model(input_var)
        if args.consensus_type == 'DNDF':
            loss = criterion(torch.log(output), target_var)
        else:
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            # print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()
        #return 
    return top1.avg,losses.avg


def validate(val_loader, model, criterion, iter, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            if args.consensus_type == 'DNDF':
                loss = criterion(torch.log(output), target_var)
            else:
                loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
            print(output)
            log.write(output + '\n')
            log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f' % (best_prec1)
    print(output_best)
    log.write(output + ' ' + output_best + '\n')
    log.flush()
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),
                        '%s/%s_best.pth.tar' % (args.root_model, args.store_name))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
