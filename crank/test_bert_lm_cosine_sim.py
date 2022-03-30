import datetime

import numpy as np
import pandas as pd
from simpletransformers.language_representation import RepresentationModel
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot
from tqdm.auto import tqdm

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    # this test uses the bert masked language model to embed filenames and captions
    # and then computes the cosine similarity among all pairs, keeping the top 5 most
    # similar captions for every filename

    # data_source = 'test'
    data_source = 'validation'

    model = RepresentationModel('bert', 'bert_lm/checkpoint-912250-epoch-1')

    df = pd.read_csv(f'data/{data_source}.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    to_match = 10

    # prefilter = 0
    prefilter = 200
    # prefilter = 1000

    if prefilter:
        to_match = prefilter

    encoded_filenames = model.encode_sentences(filenames, combine_strategy='mean')

    encoded_captions = model.encode_sentences(captions, combine_strategy='mean')

    print(encoded_filenames.shape, encoded_captions.shape)

    # code from sklearn's cosine similarity function has been copied here to avoid
    # doing normalization of the same vectors on every call
    encoded_filenames, encoded_captions = check_pairwise_arrays(encoded_filenames, encoded_captions)
    encoded_filenames_normalized = normalize(encoded_filenames, copy=True)
    encoded_captions_normalized = normalize(encoded_captions, copy=True)

    # must compute cosine similarities in batches, keeping track of the top 5 most similar results
    # otherwise it would require to much memory
    batch_size = 100
    top_values = None
    top_idxs = None
    for idx in tqdm(range(0,len(encoded_captions),batch_size)):
        # cosine similarity of all filenames against a batch of captions
        batch_sims = safe_sparse_dot(encoded_filenames_normalized,
                                     encoded_captions_normalized[idx:idx + batch_size].T,
                                     dense_output=True)
        if top_values is not None:
            # updating the top 5 most similar captions
            batch_idxs = np.asarray([list(range(batch_sims.shape[1]))] * batch_sims.shape[0]) + idx
            sims_pool = np.hstack((top_values, batch_sims))
            idxs_pool = np.hstack((top_idxs, batch_idxs))
            top = np.argpartition(sims_pool, max(-to_match,-sims_pool.shape[1]))[:, -to_match:]
            top_values = np.take_along_axis(sims_pool, top, axis=1)
            top_idxs = np.take_along_axis(idxs_pool, top, axis=1)
        else:
            top = np.argpartition(batch_sims,max(-to_match,-batch_sims.shape[1]))[:, -to_match:]
            top_values = np.take_along_axis(batch_sims, top, axis=1)
            top_idxs = top + idx

    # if prefilter:
    if True:
        with open(f'output/prefilter_bert_lm_cosine_similarity_{data_source}_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv', mode='wt', encoding='utf-8') as output_file:
            for row in top_idxs:
                print(','.join((str(idx) for idx in row[::-1])), file=output_file)
    # else:
        results = list()
        order = np.argsort(top_values, axis=1)
        for row_idx, idxs in enumerate(order):
            for idx in idxs[::-1]:
                print(row_idx, idx, top_idxs[row_idx, idx], top_values[row_idx, idx], filenames[row_idx], '|',
                      captions[top_idxs[row_idx, idx]])
                results.append((row_idx, captions[top_idxs[row_idx, idx]]))

        output_filename = f'output/bert_lm_cosine_similarity_{data_source}_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv'
        df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
        df.to_csv(output_filename, index=False)
