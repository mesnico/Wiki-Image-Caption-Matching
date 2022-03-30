import datetime
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from tqdm.auto import tqdm

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    # classifies all the pairs of filenames and captions as either matching or non-matching
    # classification scores are used to keep the top 5 most matching captions
    os.makedirs('output', exist_ok=True)
    os.makedirs('scores', exist_ok=True)

    prefilter_filename = sys.argv[1]

    if prefilter_filename.find('test') >= 0:
        data_source = 'test'
    else:
        data_source = 'validation'

    prefilter_name = Path(prefilter_filename).name
    prefilter_name = prefilter_name[:prefilter_name.find(f'_{data_source}')]

    os.makedirs(f'scores/by_idx_{prefilter_name}_{data_source}', exist_ok=True)

    top_idxs = list()
    top_scores = list()

    print('Data source:', data_source)
    print('Prefilter:', prefilter_name)

    if os.name == 'nt':
        use_multiprocessing_for_evaluation = False
    else:
        use_multiprocessing_for_evaluation = True

    model_args = ClassificationArgs(eval_batch_size=1024,
                                    use_multiprocessing_for_evaluation=use_multiprocessing_for_evaluation)
    if data_source == 'validation':
        model = ClassificationModel('auto', 'roberta_classifier/checkpoint-900000', args=model_args)
    else:
        model = ClassificationModel('auto', 'roberta_classifier/checkpoint-934867-epoch-1', args=model_args)

    df = pd.read_csv(f'data/{data_source}.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    # use_tran = True
    use_tran = False

    if use_tran:
        df = pd.read_csv(f'data/{data_source}_en_tran.tsv', sep='\t')
        filenames_tran = df['image_url']
    else:
        filenames_tran = filenames

    k_start = None
    k_end = 200
    # k_start = 0
    # k_end = 1000
    all_candidates = list()
    with open(prefilter_filename, mode='tr', encoding='utf-8') as input_file:
        for line in input_file:
            all_candidates.append(sorted([int(token) for token in line.split(',')[k_start:k_end]]))

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    # use_tran = True
    use_tran = False

    if use_tran:
        df = pd.read_csv(f'data/{data_source}_caption_list_en_tran.csv')
        captions_tran = df['caption_title_and_reference_description']
        tran_flag = '_tran'
    else:
        captions_tran = captions
        tran_flag = ''

    to_keep = 1000
    to_match = 100

    for idx, (filename, candidates) in tqdm(enumerate(zip(filenames_tran, all_candidates)), total=len(filenames)):
        pairs = [[filename, caption] for caption in captions_tran[candidates]]
        predictions, raw_outputs = model.predict(pairs)
        top = np.argsort(raw_outputs[:, 1])[-to_keep:]
        local_top_scores = list()
        local_top_idxs = list()
        for top_idx in reversed(top):
            local_top_scores.append(raw_outputs[top_idx, 1])
            local_top_idxs.append(candidates[top_idx])
        if len(top_idxs) == len(filenames):
            duplicate = list()
            for pos, local_idx in enumerate(local_top_idxs):
                if local_idx in top_idxs[idx]:
                    duplicate.append(pos)
            for pos in reversed(duplicate):
                local_top_idxs.pop(pos)
                local_top_scores.pop(pos)
            group_idxs = top_idxs[idx] + local_top_idxs
            group_scores = top_scores[idx] + local_top_scores
            group_best = list(reversed(np.argsort(group_scores)[-to_keep:]))
            top_idxs[idx] = [group_idxs[best] for best in group_best]
            top_scores[idx] = [group_scores[best] for best in group_best]
        else:
            top_scores.append(local_top_scores)
            top_idxs.append(local_top_idxs)
        with open(f'scores/by_idx_{prefilter_name}_{data_source}/{idx}.txt', mode='tw', encoding='utf-8') as output_file:
            print(','.join(f'{id}:{score}' for id, score in zip(top_idxs[idx], top_scores[idx])),file=output_file)
        top_idxs[idx] = top_idxs[idx][:to_match]
        top_scores[idx] = top_scores[idx][:to_match]

    results = list()
    for idx, row in enumerate(top_idxs):
        for top_idx in row[:to_match]:
            results.append((idx, captions[top_idx]))

    time_str = f'{datetime.datetime.now():%Y-%m-%d-%H-%M}'

    output_filename = f'scores/roberta_classifier_{prefilter_name}_{data_source}{tran_flag}_{time_str}_scores.csv'
    df = pd.DataFrame(top_scores)
    df.to_csv(output_filename, index=False)

    output_filename = f'scores/roberta_classifier_{prefilter_name}_{data_source}{tran_flag}_{time_str}_idxs.csv'
    df = pd.DataFrame(top_idxs)
    df.to_csv(output_filename, index=False)

    output_filename = f'output/roberta_classifier_{prefilter_name}_{data_source}{tran_flag}_{time_str}.csv'
    df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
    df.to_csv(output_filename, index=False)
