import os

import tqdm
import yaml
import json
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch
import clip
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold, KFold
import logging
from torch.utils.tensorboard import SummaryWriter

import argparse
from dataset import WikipediaDataset
import utils

from model import MatchingModel

from shutil import copyfile


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=40, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--data_dir', default='data', help='Root dir for data')
    # parser.add_argument('--crop_size', default=224, type=int,
    #                     help='Size of an image crop as the CNN input.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=200, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--test_step', default=100000000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/test',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads model, optimizer, scheduler')
    parser.add_argument('--load_model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")
    parser.add_argument('--cross_validation', action='store_true', help='Enables cross validation')
    # parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')

    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load the train dataframe, and associate samples to folds
    train_df = utils.create_train_pd(opt.data_dir)
    num_folds = config['dataset']['n-folds']
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for k, (train_idx, valid_idx) in enumerate(kfold.split(X=train_df, y=train_df['language'])):
        train_df.loc[valid_idx, 'Fold'] = k

    if opt.cross_validation:
        logging.info('Using {} folds'.format(num_folds))
        for fold in tqdm.trange(num_folds):
            train(opt, config, train_df, fold=fold)
    else:
        # train using fold 0 as validation fold
        train(opt, config, train_df, fold=0)

def train(opt, config, data_df, fold=0):
    # torch.cuda.set_enabled_lms(True)
    # if (torch.cuda.get_enabled_lms()):
    #     torch.cuda.set_limit_lms(11000 * 1024 * 1024)
    #     print('[LMS=On limit=' + str(torch.cuda.get_limit_lms()) + ']')

    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')
    experiment_path = tb_logger.get_logdir()

    # Dump configuration to experiment path
    copyfile(opt.config, os.path.join(experiment_path, 'config.yaml'))

    # Load datasets and create dataloaders
    _, clip_transform = clip.load(config['image-model']['model-name'])
    tokenizer = AutoTokenizer.from_pretrained(config['text-model']['model-name'])

    x_train, x_valid = data_df.query(f"Fold != {fold}"), data_df.query(f"Fold == {fold}")
    train_dataset = WikipediaDataset(x_train, tokenizer, max_length=80, split='trainval', transforms=clip_transform)
    val_dataset = WikipediaDataset(x_valid, tokenizer, max_length=80, split='trainval', transforms=clip_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, num_workers=opt.workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['bs'], shuffle=False, num_workers=opt.workers)

    # Construct the model
    model = MatchingModel(config)
    if torch.cuda.is_available():
        model.cuda()

    # Construct the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    # if not config['text-model']['fine-tune'] and not config['image-model']['fine-tune']:
    #     optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if 'textual_module' not in n and 'visual_module' not in n], lr=config['training']['lr'])
    # else:
    #     if config['dataset']['task'] == 3:
    #         optimizer = torch.optim.Adam([
    #             {'params': [p for n, p in model.named_parameters() if 'textual_module' not in n and 'visual_module' not in n]},
    #             {'params': model.textual_module.parameters(), 'lr': config['training']['pretrained-modules-lr']},
    #             {'params': model.visual_module.parameters(), 'lr': config['training']['pretrained-modules-lr']}]
    #             , lr=config['training']['lr'])
    #     elif config['dataset']['task'] == 1:
    #         optimizer = torch.optim.Adam([
    #             {'params': [p for n, p in model.named_parameters() if
    #                         'textual_module' not in n and 'visual_module' not in n]},
    #             {'params': model.textual_module.parameters(), 'lr': config['training']['pretrained-modules-lr']}]
    #             , lr=config['training']['lr'])
    # LR scheduler
    scheduler_name = config['training']['scheduler']
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=config['training']['gamma'], milestones=config['training']['milestones'])
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError('{} scheduler is not available'.format(scheduler_name))


    # # optionally resume from a checkpoint
    start_epoch = 0
    # if opt.resume or opt.load_model:
    #     filename = opt.resume if opt.resume else opt.load_model
    #     if os.path.isfile(filename):
    #         print("=> loading checkpoint '{}'".format(filename))
    #         checkpoint = torch.load(filename, map_location='cpu')
    #         model.load_state_dict(checkpoint['model'], strict=False)
    #         if torch.cuda.is_available():
    #             model.cuda()
    #         if opt.resume:
    #             start_epoch = checkpoint['epoch']
    #             # best_rsum = checkpoint['best_rsum']
    #             optimizer.load_state_dict(checkpoint['optimizer'])
    #             if checkpoint['scheduler'] is not None and not opt.reinitialize_scheduler:
    #                 scheduler.load_state_dict(checkpoint['scheduler'])
    #             # Eiters is used to show logs as the continuation of another
    #             # training
    #             model.Eiters = checkpoint['Eiters']
    #             print("=> loaded checkpoint '{}' (epoch {})"
    #                   .format(opt.resume, start_epoch))
    #         else:
    #             print("=> loaded only model from checkpoint '{}'"
    #                   .format(opt.load_model))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(opt.resume))

    model.train()

    # Train loop
    mean_loss = 0
    progress_bar = tqdm.trange(start_epoch, opt.num_epochs)
    progress_bar.set_description('Train')

    for epoch in progress_bar:
        for it, data in enumerate(tqdm.tqdm(train_dataloader)):
            global_iteration = epoch * len(train_dataloader) + it

            # forward the model
            optimizer.zero_grad()

            loss = model(*data)
            loss.backward()

            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            mean_loss += loss.item()

            if global_iteration % opt.log_step == 0:
                mean_loss /= opt.log_step
                progress_bar.set_postfix(dict(loss='{:.2}'.format(mean_loss)))
                mean_loss = 0

            tb_logger.add_scalar("Training/Epoch", epoch, global_iteration)
            tb_logger.add_scalar("Training/Loss", loss.item(), global_iteration)
            tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], global_iteration)

            if False: #global_iteration % opt.val_step == 0:    # TODO: validation
                # validate (using different thresholds)
                metrics = validate(val_dataloader, model, classes, thresholds=[0.3, 0.5, 0.8])
                tb_logger.add_scalars("Validation/F1", metrics, global_iteration)
                print(metrics)
                # progress_bar.set_postfix(dict(macroF1='{:.2}'.format(metrics['macroF1_thr=0.5']), microF1='{:.2}'.format(metrics['microF1_thr=0.5'])))

                # save best model
                if metrics['macroF1_thr=0.3'] + metrics['microF1_thr=0.3'] > best_f1:
                    print('Saving best model...')
                    checkpoint = {
                        'cfg': config,
                        'epoch': epoch,
                        'model': model.joint_processing_module.state_dict() if not config['text-model']['fine-tune'] and not config['image-model']['fine-tune'] else model.state_dict()}
                        # 'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict()}
                    latest = os.path.join(experiment_path, 'model_best_fold{}.pt'.format(val_fold))
                    torch.save(checkpoint, latest)
                    best_f1 = metrics['macroF1_thr=0.3'] + metrics['microF1_thr=0.3']

        scheduler.step()


def validate(val_dataloader, model, classes_list, thresholds=[0.3, 0.5, 0.8]):
    model.eval()

    # TODO!!
    # predictions = []
    # metrics = {}
    # progress_bar = tqdm.tqdm(thresholds)
    # progress_bar.set_description('Validation')
    # for thr in progress_bar:
    #     for it, (image, text, text_len, labels, ids) in enumerate(val_dataloader):
    #         if torch.cuda.is_available():
    #             image = image.cuda() if image is not None else None
    #             text = text.cuda()
    #             labels = labels.cuda()
    #         with torch.no_grad():
    #             pred_classes = model(image, text, text_len, inference_threshold=thr)
    #
    #         for id, labels in zip(ids, pred_classes):    # loop over every element of the batch
    #             predictions.append({'id': id, 'labels': labels})
    #
    #     macro_f1, micro_f1 = evaluate(predictions, val_dataloader.dataset.targets, classes_list)
    #     metrics['macroF1_thr={}'.format(thr)] = macro_f1
    #     metrics['microF1_thr={}'.format(thr)] = micro_f1

    model.train()
    return metrics

if __name__ == '__main__':
    main()