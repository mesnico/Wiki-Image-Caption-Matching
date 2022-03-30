import pandas as pd

if __name__ == '__main__':
    # creates validation data for the sentence pair classifier
    # gets data from the end of the training data, use it on a model that has not been trained on such data
    validation_data = list()
    validation_size = 10000
    data_df = pd.read_csv('data/train-00004-of-00005.tsv', sep='\t', usecols=[2, 17])

    data_df = data_df[-validation_size:]
    data_df.reset_index()['image_url'].to_csv('data/validation.tsv', sep='\t', index_label='id')

    data_df.reset_index()['caption_title_and_reference_description'].to_csv('data/validation_caption_list.csv',
                                                                            index=False)
