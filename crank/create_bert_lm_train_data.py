import pandas as pd
from tqdm.auto import tqdm

from train_transformer import url_to_filename

if __name__ == '__main__':
    # creates a training set to train a bert masked language model
    # every line is a filename (converted into a human readable form) concatenated with
    # its own caption from training data
    train_filenames = [f'train-0000{i}-of-00005.tsv' for i in range(5)]

    for i, train_filename in tqdm(enumerate(train_filenames), total=len(train_filenames)):
        data_df = pd.read_csv('data/' + train_filename, sep='\t', usecols=[2, 17])
        filenames = [url_to_filename(url) for url in data_df['image_url']]
        captions = data_df['caption_title_and_reference_description']

        with open('data/train_bert_lm.txt', mode='at', encoding='utf-8') as output_file:
            for filename, caption in tqdm(zip(filenames, captions), total=len(filenames)):
                print(filename, caption, file=output_file)
