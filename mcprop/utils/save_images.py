import gzip
import csv
import json
import pandas as pd
import os
import uuid
import base64

from os import cpu_count
from tqdm import tqdm
from multiprocessing.pool import Pool
from functools import partial

import argparse


def save_images(csv_file):
    df = pd.read_csv(os.path.join(opt.images_path, csv_file), compression='gzip', sep="\t")
    rows = len(df.index)
    for index, row in enumerate(df.iterrows()):
        url = row[1][0]
        base64_string = row[1][1]
        image_name = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
        
        decoded = base64.b64decode(base64_string)
        output_file = open(os.path.join(opt.output_path, image_name + ".jpg"), 'w', encoding="ISO-8859-1")
        output_file.write(decoded.decode("ISO-8859-1"))
        output_file.close()

        if index % 10000 == 0 and index > 0:
            print("Saved", index, "/", rows ,"images for csv", csv_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()  
    parser.add_argument('--images_path', default="../../data/image_data_train/image_pixels", type=str,
                        help='Path to the images_pixels folder')
    parser.add_argument('--output_path', default="../../data/image_data_train/images", type=str,
                        help='Path to the output folder')
    opt = parser.parse_args()
    print(opt)

    paths = os.listdir(opt.images_path)
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(save_images), paths):
                pbar.update()
