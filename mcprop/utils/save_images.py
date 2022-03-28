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

BASE_DIR = "../data/image_data_train/image_pixels"
OUTPUT_DIR = "../data/image_data_train/images"
def save_images(csv_file):
    df = pd.read_csv(os.path.join(BASE_DIR, csv_file), compression='gzip', sep="\t")
    rows = len(df.index)
    for index, row in enumerate(df.iterrows()):
        url = row[1][0]
        base64_string = row[1][1]
        image_name = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
        
        decoded = base64.b64decode(base64_string)
        output_file = open(os.path.join(OUTPUT_DIR, image_name + ".jpg"), 'w', encoding="ISO-8859-1")
        output_file.write(decoded.decode("ISO-8859-1"))
        output_file.close()

        if index % 10000 == 0 and index > 0:
            print("Saved", index, "/", rows ,"images for csv", csv_file)
     
paths = os.listdir(BASE_DIR)
with Pool(processes=cpu_count()-2) as p:
    with tqdm(total=len(paths)) as pbar:
        for v in p.imap_unordered(partial(save_images), paths):
            pbar.update()
