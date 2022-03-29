import datetime
from functools import partial

import numpy as np
import pandas as pd
from Levenshtein import ratio
from tqdm.auto import tqdm

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    # trivial baseline: matches filenames and caption by using the levenshtein distance

    # data_source = 'test'
    data_source = 'validation'

    df = pd.read_csv(f'data/{data_source}.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    # use_tran = True
    use_tran = False

    if use_tran:
        df = pd.read_csv(f'data/{data_source}_en_tran.tsv', sep='\t')
        tran_filenames = df['image_url']

        df = pd.read_csv(f'data/{data_source}_caption_list_en_tran.csv')
        tran_captions = df['caption_title_and_reference_description']
        tran_flag = '_tran'
    else:
        tran_filenames = filenames
        tran_captions = captions
        tran_flag = ''

    # prefilter = 0
    prefilter = 200
    # prefilter = 1000

    if prefilter:
        to_match = prefilter
    else:
        to_match = 5

    results = list()
    results_sub = list()
    for idx in tqdm(range(len(filenames))):
        filename = tran_filenames[idx]
        partial_ratio = partial(ratio, filename)
        sims = list(map(partial_ratio, [caption.strip('"') for caption in tran_captions]))
        top_idxs = np.argsort(sims)[-to_match:]
        # if prefilter:
        if True:
            results.append(top_idxs)
        # else:
            for top_idx in top_idxs[::-1]:
                results_sub.append((idx, captions[top_idx]))

    # if prefilter:
    if True:
        with open(
                f'output/prefilter_levenshtein_{data_source}{tran_flag}_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv',
                mode='wt', encoding='utf-8') as output_file:
            for row in results:
                print(','.join((str(idx) for idx in row)), file=output_file)
    # else:
        df = pd.DataFrame(results_sub, columns=['id', 'caption_title_and_reference_description'])
        df.to_csv(f'output/levenshtein_{data_source}{tran_flag}_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv',
                  index=False)
