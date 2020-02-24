import pickle
import os
import numpy as np
from tqdm import tqdm
import requests
from PIL import Image
import shutil
import scipy.io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ver', type=int, default=1,
  help='1: valid set our model used, 2: valid set randomly generated valid set')
args = parser.parse_args()

np.random.seed(222)

def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. 
    Returns a pointer to `filename`.
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

if not os.path.exists("test_32x32.mat"):
    print("Downloading test dataset of SVHN\n")
    download_file('http://ufldl.stanford.edu/housenumbers/test_32x32.mat','test_32x32.mat')
    print("\nDownloading done.\n")
else:
    print("test_32x32.mat is already downloaded. Did not download twice.\n")

if not os.path.exists("train_32x32.mat"):
    print("Downloading training dataset of SVHN\n")
    download_file('http://ufldl.stanford.edu/housenumbers/train_32x32.mat','train_32x32.mat')
    print("\nDownloading done.\n")
else:
    print("train_32x32.mat is already downloaded. Did not download twice.\n")

print("Generate preprocessed valid.npy data")
data = scipy.io.loadmat('train_32x32.mat')
x_lst = data['X'].transpose(3,0,1,2)
y_lst = data['y']
x_per_cls = [[] for _ in range(10)]
for i in range(len(y_lst)):
    x = x_lst[i]
    y = y_lst[i][0] - 1
    x_per_cls[y].append(x)

val_idx = np.load('val_idx.npy')
meta_val = []
for c in range(10):
    if args.ver == 1:
        meta_val.append(np.array([x_per_cls[c][idx] / 255.0 for idx in val_idx[c]]))
    elif args.ver == 2:
        idx = np.arange(len(x_per_cls[c]))
        np.random.shuffle(idx)
        meta_val.append(np.array(x_per_cls[c])[idx[:600]] / 255.0)
    else:
        raise ValueError

np.save('valid.npy', np.array(meta_val))
print("Done")

print("Generate preprocessed test.npy data")
data = scipy.io.loadmat('test_32x32.mat')
x_lst = data['X'].transpose(3,0,1,2)
y_lst = data['y']
x_per_cls = [[] for _ in range(10)]
for i in range(len(y_lst)):
    x = x_lst[i]
    y = y_lst[i][0] - 1
    x_per_cls[y].append(x / 255.0)
np.save('test.npy', np.array([np.array(_) for _ in x_per_cls]))
print("Done")

print('Removing redudant files')
os.remove('train_32x32.mat')
os.remove('test_32x32.mat')
