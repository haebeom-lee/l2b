"""
@author: Hayeon Lee
2020/02/19
Script for downloading, and reorganizing CUB few shot
Run this file as follows:
    python get_data.py
"""

import pickle
import os
import numpy as np
from tqdm import tqdm
import requests
import tarfile
from PIL import Image
import glob
import shutil
import pickle

def download_file(url, filename):
    """
    Helper method handling downloading large files from `url`
    to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename

if not os.path.exists("CUB_200_2011.tgz"):
    print("Downloading CUB_200_2011.tgz\n")
    download_file('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz','CUB_200_2011.tgz')
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")

if not os.path.exists("CUB_200_2011"):
    tarname = "CUB_200_2011.tgz"
    print("Untarring: {}".format(tarname))
    tar = tarfile.open(tarname)
    tar.extractall()
    tar.close()
    print("Removing original CUB_200_2011.tgz")
else:
    print("CUB_200_2011 folder already exists. Did not untarring twice\n")

print("Generate preprocessed valid.npy data")
with open('val_cls.pkl', 'rb') as f:
    data = pickle.load(f)
x_lst = [[] for _ in range(len(data))]
for c, x_per_cls in enumerate(tqdm(data)):
    for x_path in x_per_cls:
        img = Image.open(
            os.path.join('CUB_200_2011', 'images', x_path)).resize((84, 84))
        img = np.array(img)
        if img.shape == (84, 84, 3):
          x_lst[c].append(img / 255.0)
    x_lst[c] = np.array(x_lst[c])

np.save('valid.npy', np.array(x_lst))
print("Done")

print("Generate preprocessed test.npy data")
with open('test_cls.pkl', 'rb') as f:
    data = pickle.load(f)
x_lst = [[] for _ in range(len(data))]
for c, x_per_cls in enumerate(tqdm(data)):
    for x_path in x_per_cls:
        img = Image.open(
            os.path.join('CUB_200_2011', 'images', x_path)).resize((84, 84))
        img = np.array(img)
        if img.shape == (84, 84, 3):
          x_lst[c].append(img / 255.0)
    x_lst[c] = np.array(x_lst[c])
np.save('test.npy', np.array(x_lst))
print("Done")

# print("Removing original CUB_200_2011")
# os.remove('CUB_200_2011.tgz')
# shutil.rmtree('CUB_200_2011', ignore_errors=True)
# os.remove('attributes.txt')
