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

import evaluation
import utils
from dataset import WikipediaDataset, collate_fn_without_nones
from model import MatchingModel

def exhaustive_knn_search(query_feats, caption_feats, topk=5, batch_size=5, gpu=False):
    npts = query_feats.shape[0]
    num_batches = (npts // batch_size) + 1
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

        ordered = torch.argsort(d, dim=1, descending=True)
        ordered = ordered[:, :topk]    # get top k
        ordered = ordered.tolist()
        results.extend(ordered)

    return results

def main(opt):
    checkpoint = torch.load(opt.checkpoint, map_location='cpu')
    config = checkpoint['cfg']
    test_df = utils.create_test_pd(opt.data_dir)
    # test_df = test_df[:1200]

    # Load datasets and create dataloaders
    _, clip_transform = clip.load(config['image-model']['model-name'])
    tokenizer = AutoTokenizer.from_pretrained(config['text-model']['model-name'])

    test_dataset = WikipediaDataset(test_df, tokenizer, max_length=80, split='test', transforms=clip_transform,
                                     training_img_cache=opt.img_cache, include_images=not config['image-model']['disabled'])

    test_dataloader = DataLoader(test_dataset, batch_size=config['training']['bs'], shuffle=False,
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
    result_indexes = exhaustive_knn_search(query_feats, caption_feats, topk=opt.k, gpu=True)

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
            result_indexes = [o[b:e] for o in result_indexes]
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(result_indexes)

    if opt.submission_file:
        # convert ids to captions
        pbar = tqdm.tqdm(result_indexes)
        pbar.set_description('Assemble final table')
        final_dframes = [] # pd.DataFrame()
        for i, topk_idxs in enumerate(pbar):
            topk_idxs = topk_idxs[:5]   # only the top-5 results are evaluated
            captions_df = test_df.iloc[topk_idxs]
            captions_df['id'] = [i] * len(topk_idxs)
            final_dframes.append(captions_df) # = pd.concat([final_df, captions_df])

        final_df = pd.concat(final_dframes)
        final_df = final_df[["id", "caption_title_and_reference_description"]]
        final_df.to_csv(opt.submission_file, index=False)
    print('DONE')

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

    opt = parser.parse_args()
    print(opt)

    main(opt)