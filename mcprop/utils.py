import pandas as pd
import glob
import os

def read_train_feather_files(feathers_file_root, drop_duplicates=False):
    feathers = glob.glob(feathers_file_root + '/train*')
    print(feathers)
    train_df = pd.DataFrame()
    for file in feathers:
        df = pd.read_feather(file)
        train_df = pd.concat([train_df, df])
    if drop_duplicates:
        print("Before removing duplicate rows:", train_df.shape)
        train_df = train_df.drop_duplicates()  # Drop duplicate rows if any
        print("After removing duplicate rows:", train_df.shape)
    else:
        print("Training dataframe shape:", train_df.shape)
    train_df = train_df.reset_index(drop=True)
    print("Null:", train_df.isnull().any().any())

    train_df.head()
    return train_df

def create_train_pd(data_root, subfolder='full'):
    train_dir = os.path.join(data_root, 'train', subfolder)
    train_pd = read_train_feather_files(train_dir)
    return train_pd

def create_test_pd(data_root, subfolder='original'):
    test_dir = os.path.join(data_root, 'test', subfolder)
    print('Using test folder {}'.format(test_dir))
    url_pd = pd.read_csv(os.path.join(test_dir, 'test.tsv'), sep='\t')
    captions_pd = pd.read_csv(os.path.join(test_dir, 'test_caption_list.csv'))
    assert len(url_pd) == len(captions_pd)
    test_pd = pd.concat([url_pd, captions_pd], axis=1)
    if subfolder == 'en_translated':
        # a bit hacky: rename the loaded one as 'img_url_translated', and load the image_url from the original.
        # otherwise, we do not have the image_urls to load the images
        test_pd.rename(columns={'image_url': 'image_url_translated', 'caption_title_and_reference_description': 'caption_title_and_reference_description_translated'}, inplace=True)
        orig_url_pd = pd.read_csv(os.path.join(data_root, 'test', 'original', 'test.tsv'), sep='\t')
        orig_captions_pd = pd.read_csv(os.path.join(data_root, 'test', 'original', 'test_caption_list.csv'))

        test_pd = pd.concat([test_pd, orig_url_pd, orig_captions_pd], axis=1)
    return test_pd