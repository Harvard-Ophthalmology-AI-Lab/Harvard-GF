import os
import argparse
import random
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.optim import *
import torch.nn.functional as F

from sklearn.metrics import *
from sklearn.model_selection import KFold

import sys
sys.path.append('.')

from src.modules import *
from src.data_handler import *
from src import logger
from src.class_balanced_loss import *
from typing import NamedTuple

from fairlearn.metrics import *


class Identity_Info(NamedTuple):
    no_of_classes: int = 2
    no_of_attr: int = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('--pretrained-weights', default='', type=str)

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--data_dir', default='./results', type=str)
parser.add_argument('--model_type', default='./results', type=str)
parser.add_argument('--task', default='cls', type=str, help='cls | md | tds')
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--loss_type', default='bce', type=str)
parser.add_argument('--progression_outcome', default='', type=str)
parser.add_argument('--modality_types', default='rnflt', type=str, help='rnflt|bscans')
parser.add_argument('--fuse_coef', default=1.0, type=float)
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--attribute_type', default='race', type=str, help='race|gender')

                    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration, identity_Info=None):
    global device

    model.train()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []
    
    preds = []
    gts = []
    attrs = []
    datadirs = []

    # beta = 0.9999
    # gamma = 2.0
    # samples_per_cls = [2,3,1,2,2]
    # loss_type = "sigmoid"

    preds_by_attr = [ [] for _ in range(identity_Info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(identity_Info.no_of_attr) ]
    t1 = time.time()
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            input = input.to(device)
            target = target.to(device)

            pred = model(input).squeeze(1)

            loss = criterion(pred, target)
            
            pred_prob = torch.sigmoid(pred.detach())
            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attrs.append(attr.detach().cpu().numpy())

            for j, x in enumerate(attr.detach().cpu().numpy()):
                preds_by_attr[x].append(pred_prob[j])
                gts_by_attr[x].append(target[j].item())

        loss_batch.append(loss.item())
        
        top1_accuracy = accuracy(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), topk=(1,))
        
        top1_accuracy_batch.append(top1_accuracy)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=0).astype(int)
    cur_auc = auc_score(preds, gts)
    acc = accuracy(preds, gts, topk=(1,))

    pred_labels = (preds >= 0.5).astype(float)
    dpd = demographic_parity_difference(gts,
                                pred_labels,
                                sensitive_features=attrs)
    dpr = demographic_parity_ratio(gts,
                                pred_labels,
                                sensitive_features=attrs)
    eod = equalized_odds_difference(gts,
                                pred_labels,
                                sensitive_features=attrs)
    eor = equalized_odds_ratio(gts,
                                pred_labels,
                                sensitive_features=attrs)

    torch.cuda.synchronize()
    t2 = time.time()

    print(f"train ====> epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f} time: {t2 - t1:.4f}")

    preds_by_attr_tmp = []
    gts_by_attr_tmp = []
    aucs_by_attr = []
    for one_attr in np.unique(attrs).astype(int):
        preds_by_attr_tmp.append(preds[attrs == one_attr])
        gts_by_attr_tmp.append(gts[attrs == one_attr])
        aucs_by_attr.append(auc_score(preds[attrs == one_attr], gts[attrs == one_attr]))
        print(f'{one_attr}-attr auc: {aucs_by_attr[-1]:.4f}')

    t1 = time.time()

    return np.mean(loss_batch), acc, cur_auc, preds, gts, attrs, [preds_by_attr_tmp, gts_by_attr_tmp, aucs_by_attr], [acc, dpd, dpr, eod, eor]
    

def validation(model, criterion, optimizer, validation_dataset_loader, epoch, result_dir=None, identity_Info=None):
    global device

    model.eval()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []

    preds = []
    gts = []
    attrs = []
    datadirs = []

    preds_by_attr = [ [] for _ in range(identity_Info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(identity_Info.no_of_attr) ]

    with torch.no_grad():
        for i, (input, target, attr) in enumerate(validation_dataset_loader):
            input = input.to(device)
            target = target.to(device)
            
            pred = model(input).squeeze(1)

            loss = criterion(pred, target)

            pred_prob = torch.sigmoid(pred.detach())
            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attrs.append(attr.detach().cpu().numpy())
            
            for j, x in enumerate(attr.detach().cpu().numpy()):
                preds_by_attr[x].append(pred_prob[j])
                gts_by_attr[x].append(target[j].item())
            

            loss_batch.append(loss.item())

            top1_accuracy = accuracy(pred.cpu().numpy(), target.cpu().numpy(), topk=(1,)) 
        
            top1_accuracy_batch.append(top1_accuracy)
        
    loss = np.mean(loss_batch)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=0).astype(int)
    cur_auc = auc_score(preds, gts)
    acc = accuracy(preds, gts, topk=(1,))

    pred_labels = (preds >= 0.5).astype(float)
    dpd = demographic_parity_difference(gts,
                                pred_labels,
                                sensitive_features=attrs)
    dpr = demographic_parity_ratio(gts,
                                pred_labels,
                                sensitive_features=attrs)
    eod = equalized_odds_difference(gts,
                                pred_labels,
                                sensitive_features=attrs)
    eor = equalized_odds_ratio(gts,
                                pred_labels,
                                sensitive_features=attrs)

    print(f"test <==== epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f}")

    preds_by_attr_tmp = []
    gts_by_attr_tmp = []
    aucs_by_attr = []
    for one_attr in np.unique(attrs).astype(int):
        preds_by_attr_tmp.append(preds[attrs == one_attr])
        gts_by_attr_tmp.append(gts[attrs == one_attr])
        aucs_by_attr.append(auc_score(preds[attrs == one_attr], gts[attrs == one_attr]))
        print(f'{one_attr}-attr auc: {aucs_by_attr[-1]:.4f}')
    
    return loss, acc, cur_auc, preds, gts, attrs, [preds_by_attr_tmp, gts_by_attr_tmp, aucs_by_attr], [acc, dpd, dpr, eod, eor]


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    logger.log(f'===> random seed: {args.seed}')
    
    logger.configure(dir=args.result_dir, log_suffix='train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    trn_dataset = EyeFair(os.path.join(args.data_dir, 'train'), modality_type=args.modality_types, task=args.task, resolution=args.image_size, attribute_type=args.attribute_type)
    tst_dataset = EyeFair(os.path.join(args.data_dir, 'test'), modality_type=args.modality_types, task=args.task, resolution=args.image_size, attribute_type=args.attribute_type)

    logger.log(f'trn patients {len(trn_dataset)} with {len(trn_dataset)} samples, val patients {len(tst_dataset)} with {len(tst_dataset)} samples')

    train_dataset_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    validation_dataset_loader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    _, samples_per_attr = get_num_by_group(train_dataset_loader)
    print(f'group information:')
    logger.log(f'group information:')
    print(samples_per_attr)
    logger.log(samples_per_attr)
    idt_info = Identity_Info()

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    lastep_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'last_{args.perf_file}')

    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            acc_head_str = ', '.join([f'acc_class{x}' for x in range(len(samples_per_attr))])
            auc_head_str = ', '.join([f'auc_class{x}' for x in range(len(samples_per_attr))])
            with open(best_global_perf_file, 'w') as f:
                f.write(f'epoch, es_acc, acc, {acc_head_str}, es_auc, auc, {auc_head_str}, dpd, dpr, eod, eor, path\n')
        if not os.path.exists(lastep_global_perf_file):
            acc_head_str = ', '.join([f'acc_class{x}' for x in range(len(samples_per_attr))])
            auc_head_str = ', '.join([f'auc_class{x}' for x in range(len(samples_per_attr))])
            with open(lastep_global_perf_file, 'w') as f:
                f.write(f'epoch, es_acc, acc, {acc_head_str}, es_auc, auc, {auc_head_str}, dpd, dpr, eod, eor, path\n')

    if args.task == 'md':
        out_dim = 1
        criterion = nn.MSELoss()
        predictor_head = nn.Identity()
    elif args.task == 'cls': 
        out_dim = 1
        criterion = nn.BCEWithLogitsLoss()
        predictor_head = nn.Sigmoid()
    elif args.task == 'tds': 
        out_dim = 52
        criterion = nn.MSELoss()
        predictor_head = nn.Identity()

    if args.modality_types == 'ilm' or args.modality_types == 'rnflt':
        in_dim = 1
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
    elif 'bscan' in args.modality_types:
        in_dim = 200
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
    elif args.modality_types == 'rnflt+ilm':
        in_dim = 2
        model = OphBackbone(model_type=args.model_type, in_dim=in_dim, coef=args.fuse_coef)
    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.0, 0.1), weight_decay=args.weight_decay)
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    start_epoch = 0
    best_top1_accuracy = 0.

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)

        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    total_iteration = len(trn_dataset)//args.batch_size

    best_auc_groups = None
    best_acc_groups = None
    best_pred_gt_by_attr = None
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min
    best_ep = 0
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_auc, trn_preds, trn_gts, trn_attrs, trn_pred_gt_by_attrs, trn_other_metrics = train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration, identity_Info=idt_info)
        test_loss, test_acc, test_auc, tst_preds, tst_gts, tst_attrs, tst_pred_gt_by_attrs, tst_other_metrics = validation(model, criterion, optimizer, validation_dataset_loader, epoch, identity_Info=idt_info)
        scheduler.step()

        trn_acc_groups = []
        trn_auc_groups = []
        for i_group in range(len(trn_pred_gt_by_attrs[0])):
            trn_acc_groups.append(accuracy(trn_pred_gt_by_attrs[0][i_group], trn_pred_gt_by_attrs[1][i_group], topk=(1,))) 
            trn_auc_groups.append(auc_score(trn_pred_gt_by_attrs[0][i_group], trn_pred_gt_by_attrs[1][i_group]))
        
        acc_groups = []
        auc_groups = []
        for i_group in range(len(tst_pred_gt_by_attrs[0])):
            acc_groups.append(accuracy(tst_pred_gt_by_attrs[0][i_group], tst_pred_gt_by_attrs[1][i_group], topk=(1,))) 
            auc_groups.append(auc_score(tst_pred_gt_by_attrs[0][i_group], tst_pred_gt_by_attrs[1][i_group]))
        es_acc = equity_scaled_accuracy(tst_preds, tst_gts, tst_attrs)
        es_auc = equity_scaled_AUC(tst_preds, tst_gts, tst_attrs)

        if best_auc <= test_auc:
            best_auc = test_auc
            best_acc = test_acc
            best_ep = epoch
            best_pred_gt_by_attr = tst_pred_gt_by_attrs
            best_tst_other_metrics = tst_other_metrics
            best_acc_groups = acc_groups
            best_auc_groups = auc_groups
            best_es_acc = es_acc
            best_es_auc = es_auc

            state = {
            'epoch': epoch,# zero indexing
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scaler_state_dict' : scaler.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'train_auc': train_auc,
            'test_auc': test_auc
            }
            
        print(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        for i_attr in range(len(best_pred_gt_by_attr[-1])):
            print(f'---- best AUC at {i_attr}-attr {best_pred_gt_by_attr[-1][i_attr]:.4f} at epoch {best_ep}')
            logger.log(f'---- best AUC at {i_attr}-attr {best_pred_gt_by_attr[-1][i_attr]:.4f} at epoch {best_ep}')
    
        if args.result_dir is not None:
            np.savez(os.path.join(args.result_dir, f'pred_gt_ep{epoch:03d}.npz'), 
                        val_pred=tst_preds, val_gt=tst_gts, val_attr=tst_attrs)


        logger.logkv('epoch', epoch)
        logger.logkv('trn_loss', round(train_loss,4))
        logger.logkv('trn_acc', round(train_acc,4))
        logger.logkv('trn_auc', round(train_auc,4))
        logger.logkv('trn_acc', round(trn_other_metrics[0],4))
        logger.logkv('trn_dpd', round(trn_other_metrics[1],4))
        logger.logkv('trn_dpr', round(trn_other_metrics[2],4))
        logger.logkv('trn_eod', round(trn_other_metrics[3],4))
        logger.logkv('trn_eor', round(trn_other_metrics[4],4))
        for i_group in range(len(trn_acc_groups)):
            logger.logkv(f'trn_acc_class{i_group}', round(trn_acc_groups[i_group],4))
        for i_group in range(len(trn_auc_groups)):
            logger.logkv(f'trn_auc_class{i_group}', round(trn_auc_groups[i_group],4))

        logger.logkv('val_loss', round(test_loss,4))
        logger.logkv('val_acc', round(test_acc,4))
        logger.logkv('val_auc', round(test_auc,4))
        logger.logkv('val_es_acc', round(es_acc,4))
        logger.logkv('val_es_auc', round(es_auc,4))
        logger.logkv('val_acc', round(tst_other_metrics[0],4))
        logger.logkv('val_dpd', round(tst_other_metrics[1],4))
        logger.logkv('val_dpr', round(tst_other_metrics[2],4))
        logger.logkv('val_eod', round(tst_other_metrics[3],4))
        logger.logkv('val_eor', round(tst_other_metrics[4],4))
        for i_group in range(len(acc_groups)):
            logger.logkv(f'val_acc_class{i_group}', round(acc_groups[i_group],4))
        for i_group in range(len(auc_groups)):
            logger.logkv(f'val_auc_class{i_group}', round(auc_groups[i_group],4))
        logger.dumpkvs()

        if (epoch == args.epochs-1) and (args.perf_file != ''):
            if os.path.exists(lastep_global_perf_file):
                with open(lastep_global_perf_file, 'a') as f:
                    acc_head_str = ', '.join([f'{x:.4f}' for x in acc_groups])
                    auc_head_str = ', '.join([f'{x:.4f}' for x in auc_groups])
                    path_str = f'{args.result_dir}'
                    f.write(f'{best_ep}, {es_acc:.4f}, {best_acc:.4f}, {acc_head_str}, {es_auc:.4f}, {test_auc:.4f}, {auc_head_str}, {tst_other_metrics[1]:.4f}, {tst_other_metrics[2]:.4f}, {tst_other_metrics[3]:.4f}, {tst_other_metrics[4]:.4f}, {path_str}\n')

    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):
            with open(best_global_perf_file, 'a') as f:
                acc_head_str = ', '.join([f'{x:.4f}' for x in best_acc_groups])
                auc_head_str = ', '.join([f'{x:.4f}' for x in best_auc_groups])
                path_str = f'{args.result_dir}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_es_acc:.4f}, {best_acc:.4f}, {acc_head_str}, {best_es_auc:.4f}, {best_auc:.4f}, {auc_head_str}, {best_tst_other_metrics[1]:.4f}, {best_tst_other_metrics[2]:.4f}, {best_tst_other_metrics[3]:.4f}, {best_tst_other_metrics[4]:.4f}, {path_str}\n')

    os.rename(args.result_dir, f'{args.result_dir}_{args.seed}_auc{best_auc:.4f}')