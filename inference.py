import torch
import argparse
import logging
logging.basicConfig(level = logging.INFO)

from torch.utils.data import DataLoader
import tqdm
from transformers import AutoTokenizer
import clip
import csv
import pandas as pd
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

import evaluation
import utils
from dataset import WikipediaDataset, collate_fn_without_nones
from model import MatchingModel

SCORES_CACHE = 'cached_scores.npy'


def knn_search_from_cached_scores(npts, topk=5):
    fp = np.memmap(SCORES_CACHE, dtype='float32', mode='r', shape=(npts, npts))

    ordered = np.argsort(fp, axis=1)[:, ::-1]
    ordered = ordered[:, :topk]
    results = ordered.tolist()
    return results


def exhaustive_knn_search(query_feats, caption_feats, topk=5, batch_size=5, gpu=False, cache_scores=False):
    npts = query_feats.shape[0]
    if cache_scores and not os.path.exists(SCORES_CACHE):
        logging.info('Caching scores into {}'.format(SCORES_CACHE))
        fp = np.memmap(SCORES_CACHE, dtype='float32', mode='w+', shape=(npts, npts))

    num_batches = (npts // batch_size) + 1 if npts % batch_size != 0 else (npts // batch_size)
    pbar = tqdm.trange(num_batches)
    pbar.set_description('Top k')
    results = []
    if gpu:
        caption_feats = caption_feats.cuda()
    for index in pbar:
        # Get query image-url
        query = query_feats[index * batch_size: (index + 1) * batch_size]
        if gpu:
            query = query.cuda()
        # Compute scores
        d = torch.mm(query, caption_feats.T)  # query_batch_size x npts

        ranks = torch.argsort(d, dim=1, descending=True)
        ordered = ranks[:, :topk]    # get top k
        ordered = ordered.tolist()
        results.extend(ordered)

        if cache_scores:
            fp[index * batch_size: (index + 1) * batch_size, :] = d.cpu().numpy()

    if cache_scores:
        fp.flush()
    return results


def linear_assignment_search(npts, topk=5):
    scores = np.memmap(SCORES_CACHE, dtype='float32', mode='c', shape=(npts, npts))
    result = np.empty((npts, topk))
    # scores = np.load(SCORES_CACHE)
    for i in tqdm.trange(topk):
        row_ind, col_ind = linear_sum_assignment(scores, maximize=True)
        scores[row_ind, col_ind] = 0
        result[row_ind, i] = col_ind
    result = result.tolist()
    return result


def compute_recall(items, topk):
    """
    Text->Images (Image Search)
    Images: (N, K) matrix of images
    Captions: (N, K) matrix of captions
    """
    if topk < 10:
        logging.warning('Computing recall with topk={}. Not all the recall values are significative!'.format(topk))
    npts = len(items)

    ranks = np.zeros(npts)
    pbar = tqdm.trange(npts)
    pbar.set_description('Validation')
    for index in pbar:

        ordered = np.array(items[index])
        a = np.where(ordered == index)
        if len(a[0]) == 0:
            ranks[index] = topk + 1
        else:
            ranks[index] = a[0][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return {'r1':r1, 'r5': r5, 'r10':r10}


def main(opt):
    checkpoint = torch.load(opt.checkpoint, map_location='cpu')
    config = checkpoint['cfg']

    if opt.set == 'val':
        logging.info('Validating! Using subfolder {}'.format(opt.train_subfolder))
        n_samples = config['dataset']['val-samples']
    elif opt.set == 'test':
        logging.info('Testing!')
        n_samples = 92366

    if opt.linear_assignment:
        assert opt.enable_cached_scores and os.path.exists(SCORES_CACHE)
        logging.info('Using cached scores. Linear assignment search')
        result_indexes = linear_assignment_search(n_samples, opt.k)
    else:
        if opt.enable_cached_scores and os.path.exists(SCORES_CACHE):
            logging.info('Using cached scores. Standard search')
            result_indexes = knn_search_from_cached_scores(n_samples, opt.k)
        else:
            if opt.set == 'test':
                df = utils.create_test_pd(opt.data_dir)
            elif opt.set == 'val':
                df = utils.create_train_pd(opt.data_dir, subfolder=opt.train_subfolder)
                df = df.sample(frac=1, random_state=42)
                all_idxs = np.arange(len(df))
                val_samples = config['dataset']['val-samples']
                logging.info('Using {} samples for validating'.format(val_samples))
                valid_idx = all_idxs[:val_samples]
                df = df.loc[valid_idx]
            # test_df = test_df[:1200]

            # Load datasets and create dataloaders
            _, clip_transform = clip.load(config['image-model']['model-name'])
            tokenizer = AutoTokenizer.from_pretrained(config['text-model']['model-name'])

            test_dataset = WikipediaDataset(df, tokenizer, max_length=80, split=opt.set, transforms=clip_transform,
                                            training_img_cache=opt.img_cache,
                                            include_images=not config['image-model']['disabled'])

            test_dataloader = DataLoader(test_dataset, batch_size=opt.bs, shuffle=False,
                                         num_workers=opt.workers, collate_fn=collate_fn_without_nones)

            # Construct the model
            model = MatchingModel(config)
            if torch.cuda.is_available():
                model.cuda()
            logging.info('Model initialized')

            model.load_state_dict(checkpoint['model'], strict=True)
            logging.info('Checkpoint loaded')
            model.eval()

            query_feats, caption_feats, _ = evaluation.encode_data(model, test_dataloader)
            result_indexes = exhaustive_knn_search(query_feats, caption_feats, topk=opt.k, gpu=True, cache_scores=opt.enable_cached_scores)

    if opt.set == 'test':
        # if in test mode, output files for the submission
        if opt.output_indexes:
            logging.info('Saving indexes in {}'.format(opt.output_indexes))
            if not os.path.exists(opt.output_indexes):
                os.makedirs(opt.output_indexes)
            chunks_size = 1000
            num_chunks = opt.k // chunks_size
            for i in tqdm.trange(num_chunks):
                b = i * chunks_size
                e = (i + 1) * chunks_size
                fname = os.path.join(opt.output_indexes, 'indexes_{}_{}'.format(b, e) + '.csv')
                result_indexes_cut = [o[b:e] for o in result_indexes]
                with open(fname, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(result_indexes_cut)

        if opt.submission_file:
            # convert ids to captions
            pbar = tqdm.tqdm(result_indexes)
            pbar.set_description('Assemble final table')
            final_dframes = [] # pd.DataFrame()
            for i, topk_idxs in enumerate(pbar):
                topk_idxs = topk_idxs[:5]   # only the top-5 results are evaluated
                captions_df = df.iloc[topk_idxs]
                captions_df['id'] = [i] * len(topk_idxs)
                final_dframes.append(captions_df) # = pd.concat([final_df, captions_df])

            final_df = pd.concat(final_dframes)
            final_df = final_df[["id", "caption_title_and_reference_description"]]
            final_df.to_csv(opt.submission_file, index=False)
    else:
        # validate on the val set
        metrics = compute_recall(result_indexes, topk=opt.k)
        print(metrics)
    logging.info('DONE')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    parser.add_argument('--data_dir', default='data', help='Root dir for data')
    parser.add_argument('--submission_file', default=None, help='File name for submission (.csv)')
    parser.add_argument('--output_indexes', default=None, help='Path to folder where indexes csv files are stored')
    parser.add_argument('--k', type=int, default=1000, help='k for k-nn search')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--img_cache', type=str, default=None, help='Path to images')
    parser.add_argument('--set', type=str, default='test', choices=['test', 'val'], help='Which set to use for inference')
    parser.add_argument('--linear_assignment', action='store_true', help='Enable linear assignment')
    parser.add_argument('--disable_cached_scores', action='store_true', help='Disables loading of the cached scores')
    parser.add_argument('--train_subfolder', type=str, default='full', help='Which train feather files are used (for constructing the validation set')
    parser.add_argument('--bs', type=int, default=64)

    opt = parser.parse_args()
    opt.enable_cached_scores = not opt.disable_cached_scores
    print(opt)

    main(opt)