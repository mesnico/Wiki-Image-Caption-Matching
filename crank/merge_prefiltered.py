import datetime
import sys
from collections import defaultdict

from tqdm.auto import tqdm

if __name__ == '__main__':
    filenames = sys.argv[1:]

    candidates = defaultdict(list)
    for filename in filenames:
        all_candidates = list()
        with open(filename, mode='tr', encoding='utf-8') as input_file:
            for idx, line in enumerate(input_file):
                candidates[idx].append([int(token) for token in line.split(',')])

    k = 1000

    selected = list()
    for idx_candidates in tqdm(candidates.values()):
        idx_selected = list()
        rr = 0
        step = 0
        while len(idx_selected) < k:
            new_candidate = idx_candidates[rr][step]
            if new_candidate not in idx_selected:
                idx_selected.append(new_candidate)
            rr += 1
            if rr >= len(idx_candidates):
                rr = 0
                step += 1
        selected.append(idx_selected)

    with open(f'output/prefilter_merged_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv', mode='wt',
              encoding='utf-8') as output_file:
        for row in selected:
            print(','.join((str(idx) for idx in row)), file=output_file)
