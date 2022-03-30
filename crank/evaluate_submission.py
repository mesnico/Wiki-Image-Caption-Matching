import sys

import numpy as np
import pandas as pd

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    submission_filename = sys.argv[1]

    if submission_filename.find('test') >= 0:
        data_source = 'test'
    else:
        data_source = 'validation'

    df = pd.read_csv(f'data/{data_source}.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    submission = pd.read_csv(submission_filename)

    ks = (1,5,10)

    for k in ks:
        last_id = -1
        den_count = 0
        total_rel = 0
        found = 0
        for row in submission.iterrows():
            id = row[1]['id']
            if last_id != id:
                if last_id != -1:
                    total_rel += rel
                    den_count += 1
                id_count = 1
                rel = 0
            else:
                id_count += 1
            if captions.iloc[id] == row[1]['caption_title_and_reference_description'] and id_count<=k:
                found += 1
                rel += 1 / np.log2(id_count + 1)

            last_id = id

        total_rel += rel
        den_count += 1

        print(f'Retrieved@{k} {found}/{den_count}')
        print(f'nDCG{k} = {total_rel / den_count}')
