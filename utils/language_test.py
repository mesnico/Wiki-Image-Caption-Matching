import utils
from urllib.parse import unquote
import os
import fasttext

data_dir = 'data'
subfolder = 'downsampled'

if __name__ == '__main__':
    df = utils.create_train_pd(data_dir, subfolder=subfolder)
    df = df.sample(frac=1, random_state=42)
    lang_model = fasttext.load_model('lid.176.bin')

    for index, row in df.iterrows():
        language = row['language']
        caption = row['caption_title_and_reference_description']
        url = row['image_url']

        # clean image url
        url = url.rsplit('/', 1)
        url = url[0] if len(url) == 1 else url[1]
        url = unquote(url)
        url = url.replace('_', ' ')
        url = os.path.splitext(url)[0]
        url_prediction = lang_model.predict(url, k=3)[0]

        # clean caption
        caption = caption.replace("[SEP]", " ")
        caption_prediction = lang_model.predict(caption, k=3)[0]

        print('URL: {}; GT: {} - Predicted: {}'.format(url, language, url_prediction))
        print('Cap: {}; GT: {} - Predicted: {}'.format(caption, language, caption_prediction))
        print('-----')
        if index == 50:
            break
    # logging.info('Using {} samples for validating'.format(val_samples))
    # valid_idx = all_idxs[:val_samples]
    # df = df.loc[valid_idx]