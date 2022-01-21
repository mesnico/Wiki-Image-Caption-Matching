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
from scipy import sparse as sp
import json

import evaluation
import utils
from dataset import WikipediaDataset, collate_fn_without_nones
from model import MatchingModel

# SCORES_CACHE = 'cached_scores.npz'  # sparse
# IDXS_CACHE = 'cached_indexes.npy'  # dense


def get_cached_files(set):
    cache_dir = 'cached_scores' if opt.scores_dir is None else opt.scores_dir  # TODO: opt is accessed as a global variable :(
    scores_file = 'cached_scores_{}.npz'.format(set)
    indexes_file = 'cached_indexes_{}.npy'.format(set)
    return os.path.join(cache_dir, scores_file), os.path.join(cache_dir, indexes_file)	# scores, indexes 


def knn_search_from_cached_scores(npts, set, topk=5):
    indexes = np.load(get_cached_files(set)[1])
    indexes = indexes[:, :topk]
    return indexes.tolist()

    # inds, _ = utils.sparse_topk(scores, k=topk, reverse=True, return_mat=False)
    # df = pd.DataFrame(zip(*inds))
    # lol = df.groupby(0)[1].apply(list).tolist()
    # return lol


def exhaustive_knn_search(query_feats, caption_feats, set, topk=5, batch_size=5, gpu=False, cache_scores=False, top_scores=1000):
    score_file, index_file = get_cached_files(set)
    npts = query_feats.shape[0]
    if cache_scores and not os.path.exists(score_file):
        logging.info('Caching scores into {}'.format(score_file))
        cached_scores = sp.lil_matrix((npts, npts), dtype='float32')
        cached_indexes = np.empty((npts, top_scores))

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

        scores, ranks = torch.sort(d, dim=1, descending=True)
        ordered = ranks[:, :topk]    # get top k
        ordered = ordered.tolist()
        results.extend(ordered)

        if cache_scores:
            for i in range(d.shape[0]):
                internal_idx = index * batch_size + i
                r = ranks[i, :top_scores].cpu().numpy()
                s = scores[i, :top_scores]

                cached_scores[internal_idx, r] = s.cpu().numpy() # save only the first 'top_scores' indexes
                cached_indexes[internal_idx, :] = r

    if cache_scores:
        sp.save_npz(score_file, cached_scores.tocsr())
        np.save(index_file, cached_indexes)
    return results


def linear_assignment_search(set, topk=5, top_scores=1000, use_rank=False, score_override=None, index_override=None):
    limit = 0 # now it is disabled # TODO: this limit is empirically found for making a complete graph matching
    score_file, index_file = get_cached_files(set)
    if score_override is not None:
        scores = score_override
        indexes = index_override
    else:
        indexes = np.load(index_file).astype(int)
    npts = indexes.shape[0]
    if score_override is None:
        if not use_rank:
            scores = sp.load_npz(score_file)
        else:
            scores = sp.lil_matrix((npts, scores.shape[1]), dtype='float32')
            for i, ind in enumerate(indexes):
                scores[i, ind] = np.log(1001) - np.log(np.arange(1, 1001))
            scores = scores.tocsr()

    if top_scores < indexes.shape[1]:
        dense_scores = scores.todense()
        if top_scores < limit:
            np.put_along_axis(dense_scores, indexes[:, top_scores:limit], -1, axis=1)	# make assignment impossible (-1) but not zero, otherwise too much sparsity makes assignment impossible
            np.put_along_axis(dense_scores, indexes[:, limit:], 0, axis=1)
        else:
            np.put_along_axis(dense_scores, indexes[:, top_scores:], 0, axis=1)
        # dense_scores[indexes[:, top_scores:]] = 0
        #dense_scores = np.take(dense_scores, indexes[:, top_scores:], axis=1)
        scores = sp.csr_matrix(dense_scores)

    result = np.empty((npts, topk))
    # scores = np.load(SCORES_CACHE)
    for i in tqdm.trange(topk):
        row_ind, col_ind = sp.csgraph.min_weight_full_bipartite_matching(scores, maximize=True)
        scores[row_ind, col_ind] = 0 #scores[row_ind, col_ind] - 100
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


import pdb
def compute_indexes_stats(opt):
    score_file, index_file = get_cached_files(opt.set)
    index = np.load(index_file)
    flattened = index[:, :5].flatten()
    unique, counts = np.unique(flattened, return_counts=True)
    # print(np.asarray((unique, counts)).T)
    # max = np.max(counts)
    # max_index = np.argmax(counts)
    # print('Max = {}, at index {}'.format(max, max_index))
    sort_idxs = np.argsort(-counts)
    sorted_counts = counts[sort_idxs]
    sorted_unique = unique[sort_idxs]
    p = np.asarray((sorted_unique, sorted_counts)).T
    print(p[:10 * 5])
    queries = []
    for i, idx in enumerate(sorted_unique[:10 * 5]):
       query = np.where(flattened == idx)[0]
       query = query // 5
       queries.extend(query.tolist())

    queries = np.unique(np.array(queries))

    print('Num queries: {}'.format(len(queries)))
    #print('Num elements: {}'.format(len(queries) * 5))
    chosen_ids = sorted_unique[:len(queries)] #index[queries, :5]
    print('Num elems: {}'.format(len(chosen_ids)))
    # pdb.set_trace()
    # print(p[:20, :])
    return queries, chosen_ids  # row_ind, col_ind


def compute_la_stats(opt):
    score_file, index_file = get_cached_files(opt.set)
    assert os.path.exists(score_file)
    logging.info('Computing stats on linear assignment for different values of top_scores')
    rows = []
    top_k_scores = list(range(300, 1000, 50))
    for t in tqdm.tqdm(top_k_scores):
        result_indexes = linear_assignment_search(opt.set, opt.k, top_scores=t)
        metrics = compute_recall(result_indexes, topk=opt.k)
        row = {'top_scores': t, 'r1': metrics['r1'], 'r5': metrics['r5']}
        rows.append(row)
        print(row)
    with open(opt.compute_la_stats, 'w') as f:
        json.dump(rows, f)


def create_val_df(config, opt):
    df = utils.create_train_pd(opt.data_dir, subfolder=opt.train_subfolder)
    df = df.sample(frac=1, random_state=42)
    all_idxs = np.arange(len(df))
    val_samples = config['dataset']['val-samples']
    logging.info('Using {} samples for validating'.format(val_samples))
    valid_idx = all_idxs[:val_samples]
    df = df.loc[valid_idx]
    return df


def main(opt):
    checkpoint = torch.load(opt.checkpoint, map_location='cpu')
    config = checkpoint['cfg']
    score_file, index_file = get_cached_files(opt.set)
    df = None

    if opt.set == 'val':
        logging.info('Validating! Using subfolder {}'.format(opt.train_subfolder))
        n_samples = config['dataset']['val-samples']
    elif opt.set == 'test':
        logging.info('Testing!')
        n_samples = 92366

    #compute_indexes_stats(opt)
    #quit()

    if opt.linear_assignment:
        assert opt.enable_cached_scores and os.path.exists(score_file)
        logging.info('Using cached scores. Linear assignment search')
        result_indexes = linear_assignment_search(opt.set, opt.k, top_scores=1000, use_rank=False)
    else:
        if opt.enable_cached_scores and os.path.exists(score_file):
            logging.info('Using cached scores. Standard search')
            result_indexes = knn_search_from_cached_scores(n_samples, opt.set, opt.k)
            if opt.linear_assignment_enhancement:
                logging.info('Using linear assignment to enhance standard search')
                la_row, la_col = compute_indexes_stats(opt)
                scores = sp.load_npz(score_file)
                indexes = np.load(index_file)
                scores = scores[la_row, :]
                indexes = indexes[la_row]
                result_indexes_aug = linear_assignment_search(opt.set, opt.k, top_scores=1000, use_rank=False, index_override=indexes, score_override=scores)
                # merge
                assert len(result_indexes_aug) == len(la_row)
                for r, ind in zip(result_indexes_aug, la_row):
                    r = [int(k) for k in r]
                    result_indexes[ind] = r
        else:
            if opt.set == 'test':
                df = utils.create_test_pd(opt.data_dir, opt.test_subfolder)
            elif opt.set == 'val':
                df = create_val_df(config, opt)
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
            result_indexes = exhaustive_knn_search(query_feats, caption_feats, opt.set, topk=opt.k, gpu=True, cache_scores=opt.enable_cached_scores)

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
            if df is None:
                df = utils.create_test_pd(opt.data_dir, opt.test_subfolder)
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
        if opt.print_example_results:
            if df is None:
                df = create_val_df(config, opt)
            for i, query_res in enumerate(result_indexes[:200]):
                gt_row = df.iloc[i]
                print('[{}]: Query: {}'.format(i, gt_row["image_url"]))
                print('[{}]: GT: {}'.format(i, gt_row["caption_title_and_reference_description"]))
                print(' - Found:')
                for res in query_res:
                    res = int(res)
                    retrieved_item = df.iloc[res]
                    print('- - [{}]: {} - {}'.format(res, retrieved_item["image_url"], retrieved_item["caption_title_and_reference_description"]))

                print('------')

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
    parser.add_argument('--linear_assignment_enhancement', action='store_true', help='Use linear assignment only for enhancing the standard search')
    parser.add_argument('--compute_la_stats', type=str, default=None, help='If a json file is specified, this is where where la stats are saved')
    parser.add_argument('--disable_cached_scores', action='store_true', help='Disables loading of the cached scores')
    parser.add_argument('--train_subfolder', type=str, default='full', help='Which train feather files are used (for constructing the validation set')
    parser.add_argument('--test_subfolder', type=str, default='original',
                        help='Which test subfolder to use')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--scores_dir', type=str, default='cached_scores', help='Directory where score cache files are placed')
    parser.add_argument('--print_example_results', action='store_true', help='Print example results. Only works when set=val')

    opt = parser.parse_args()
    opt.enable_cached_scores = not opt.disable_cached_scores
    print(opt)

    if opt.compute_la_stats is not None:
        compute_la_stats(opt)
    else:
        main(opt)
