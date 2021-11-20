import os
import logging

import random
from io import BytesIO

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from urllib import request
import uuid

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

def url_to_image(cache_path, img_url):   # TODO here, get hash of url and store the file in the cache_path
    if cache_path is not None:
        file_name = uuid.uuid5(uuid.NAMESPACE_URL, img_url)
        file_name = os.path.join(cache_path, '{}.jpg'.format(file_name))
        if os.path.exists(file_name):
            try:
                img = Image.open(file_name).convert("RGB")
                return img
            except:
                pass
    # file_name = str(uuid.uuid4())
    try:
        req = request.Request(img_url)
        req.add_header('User-Agent', 'abc-bot')
        response = request.urlopen(req)
        if cache_path is not None:
            # save on disk, on the cache path
            with open(file_name, 'wb') as f:
                f.write(response.read())
            img = Image.open(file_name).convert("RGB")
        else:
            # do not save on disk
            img = Image.open(BytesIO(response.read()))
        # os.remove(file_name)
        return img
    except:
        return None


class WikipediaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, transforms=None, training_img_cache='data/img_cache', split='train'):
        self.data = data.reset_index(drop=True)
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.split=split

        self.training_img_cache = training_img_cache
        if training_img_cache is not None and not os.path.exists(training_img_cache):
            os.makedirs(training_img_cache)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #         image_bytes = base64.b64decode(self.data[index]["b64_bytes"])
        #         img = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
        if torch.is_tensor(index):
            index = index.tolist()

        img = url_to_image(self.training_img_cache, self.data.at[index, "image_url"])
        if self.split != 'test':
            # while img is None:  # TODO: better way to handle missing images?
            #     # logging.warning('Image {} not existing. Choosing a random one.'.format(self.data.at[index, "image_url"]))
            #     index = random.randint(0, len(self.data) - 1)
            #     img = url_to_image(self.training_img_cache, self.data.at[index, "image_url"])
            if img is None:
                return None
        else:
            if img is None:
                raise FileNotFoundError('Image {} was not found. And now?'.format(self.data.at[index, "image_url"]))

        # img = np.array(img)
        # caption = random.choice(self.data.at[index, "caption_title_and_reference_description"])
        caption = self.data.at[index, "caption_title_and_reference_description"]
        caption = caption.replace("[SEP]", "</s>")  # sep token for xlm-roberta
        caption_inputs = self.tokenizer.encode_plus(
            caption,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        caption_ids = caption_inputs['input_ids']
        caption_mask = caption_inputs['attention_mask']

        url = self.data.at[index, "page_url"]
        url = url.rsplit('/', 1)[1]
        url_inputs = self.tokenizer.encode_plus(
            url,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        url_ids = url_inputs['input_ids']
        url_mask = url_inputs['attention_mask']

        if self.transforms:
            img = self.transforms(img)

        url_ids = torch.tensor(url_ids, dtype=torch.long)
        url_mask = torch.tensor(url_mask, dtype=torch.long)
        caption_ids = torch.tensor(caption_ids, dtype=torch.long)
        caption_mask = torch.tensor(caption_mask, dtype=torch.long)

        return img, url_ids, url_mask, caption_ids, caption_mask


def collate_fn_without_nones(batch):
    batch = [d for d in batch if d is not None]
    return default_collate(batch)


# testing
from transformers import AutoTokenizer
import utils
import clip

if __name__ == '__main__':
    feather_files_train_root = 'data'
    test_csv = 'data/test/test.tsv'

    train_pd = utils.create_train_pd('data', downsampled=False)
    #test_pd = utils.create_test_pd('data')

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    _, clip_transform = clip.load('ViT-B/32')

    if True:
        # train
        dataset = WikipediaDataset(train_pd, tokenizer, max_length=80, split='trainval', transforms=clip_transform, training_img_cache=None)
    else:
        dataset = WikipediaDataset(test_pd, tokenizer, max_length=80, split='test')

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn_without_nones)
    for d in dataloader:
        print(d)