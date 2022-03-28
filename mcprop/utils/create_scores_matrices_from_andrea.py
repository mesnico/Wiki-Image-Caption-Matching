import numpy as np
from scipy import sparse as sp
import os
import tqdm
from scipy.stats import rankdata

in_folder = 'scores_0_1000'
out_folder = 'andrea_scores_cache'

score_file = 'cached_scores_test.npz'
index_file = 'cached_indexes_test.npy'

use_rank = 'log-tiebreak'

if __name__ == '__main__':
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # read txt files
    txt_files = os.listdir(in_folder)
    npts = len(txt_files)

    # initialize the sparse matrix
    scores_mat = sp.lil_matrix((npts, npts), dtype='float32')
    indexes_mat = np.empty((npts, 1000), dtype='int32')

    for fname in tqdm.tqdm(txt_files):
        with open(os.path.join(in_folder, fname), 'r') as f:
            stream = f.readlines()[0]   # only one line of text in these files
        tuples = stream.split(',')
        idxs, scores = zip(*[t.split(':') for t in tuples])
        idxs = [int(i) for i in idxs]
        scores = np.array([float(s) for s in scores])
        if use_rank is not None:
            if use_rank == 'linear':
                scores = 1000 - np.arange(1000)
            elif use_rank == 'log':
                scores = np.log(1001) - np.log(np.arange(1, 1001))
            elif use_rank == 'log-tiebreak':
                ranks = rankdata(-scores)
                scores = np.log(1001) - np.log(ranks)

            # convert scores to rank and override model scores
            # np.log(1002) - np.log(np.arange(2, 1002))

        row_index = int(os.path.splitext(fname)[0])
        scores_mat[row_index, idxs] = scores
        indexes_mat[row_index] = np.array(idxs)

    # dump on files
    sp.save_npz(os.path.join(out_folder, score_file), scores_mat.tocsr())
    np.save(os.path.join(out_folder, index_file), indexes_mat)

    print('DONE')
