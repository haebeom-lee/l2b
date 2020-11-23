"""
@author: Hayeon Lee
2020/02/19
Script for downloading, and reorganizing CUB 
for few shot classification
Run this file as follows:
    python get_data.py
"""

import os
from tqdm import tqdm
import requests


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


if not os.path.exists("valid.npy"):
    print("Downloading valid.npy of cub\n")
    download_file('https://www.dropbox.com/s/2cv4lfdmaja0wq6/valid.npy?dl=1','valid.npy')
    print("Downloading done.\n")
else:
    print("valid.npy has already been downloaded. Did not download twice.\n")

if not os.path.exists("test.npy"):
    print("Downloading test.npy of cub\n")
    download_file('https://www.dropbox.com/s/1oxiehzuqvuil60/test.npy?dl=1','test.npy')
    print("Downloading done.\n")
else:
    print("test.npy has already been downloaded. Did not download twice.\n")

