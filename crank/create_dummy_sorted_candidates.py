import random

import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':
    count = len(pd.read_csv('data/test.tsv', sep='\t'))
    print(count)

    # random_sample = True
    random_sample = False

    k = 500
    start = 700

    if random_sample:
        with open(f'data/random_candidates_test.csv', mode='tw', encoding='utf-8') as output_file:
            population = list(range(count))
            for _ in tqdm(range(count)):
                candidates = random.choices(population,k=k)
                print(*candidates, sep=',', file=output_file)
    else:
        with open(f'data/range_candidates_{start}_{k}_test.csv', mode='tw', encoding='utf-8') as output_file:
            candidates = list(range(start,start+k))
            for _ in tqdm(range(count)):
                print(*candidates, sep=',', file=output_file)
