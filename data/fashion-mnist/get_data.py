"""
@author: Hayeon Lee
2020/02/19
Script for downloading, and reorganizing fashion-mnist 
for few shot classification
Run this file as follows:
    python get_data.py

"""
import os
import numpy as np
from tqdm import tqdm
import requests
from PIL import Image
import glob
import shutil
import collections
import gzip


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


if not os.path.exists("t10k-images-idx3-ubyte.gz"):
    print("Downloading t10k-images-idx3-ubyte.gz\n")
    download_file(
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        't10k-images-idx3-ubyte.gz')
    print("Downloading done.\n")
else:
    print("t10k-images-idx3-ubyte.gz has already been downloaded. Did not download twice.\n")


if not os.path.exists("t10k-labels-idx1-ubyte.gz"):
    print("Downloading t10k-labels-idx1-ubyte.gz\n")
    download_file(
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        't10k-labels-idx1-ubyte.gz')
    print("Downloading done.\n")
else:
    print("t10k-labels-idx1-ubyte.gz has already been downloaded. Did not download twice.\n")


with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
        offset=8)
with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
        offset=16).reshape(len(labels), 784)

print("Generate preprocessed Fashion-MNIST test.npy data")
data = collections.defaultdict(list)
for img, label in tqdm(zip(images, labels)):
    img = Image.fromarray(img.reshape((28, 28)))
    img = np.array(img.convert('RGB').resize((32, 32)))
    data[label].append(img/ 255.0)
data = [np.array(_) for _ in data.values()]
np.save('test.npy', np.array(data))
print('Done')

print("Removing original t10k-labels-idx1-ubyte.gz")
os.remove('t10k-labels-idx1-ubyte.gz')
print("Removing original t10k-images-idx3-ubyte.gz")
os.remove('t10k-images-idx3-ubyte.gz')
