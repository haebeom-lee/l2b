"""
We modify original code (https://github.com/ArnoutDevos/maml-cifar-fs/get_cifarfs.py)
@author: Arnout Devos
2018/12/06
MIT License
Script for downloading, and reorganizing CIFAR few shot from CIFAR-100 according
to split specifications in Luca et al. '18.
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
import sys

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

if not os.path.exists("cifar-100-python.tar.gz"):
    print("Downloading cifar-100-python.tar.gz\n")
    download_file('http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz','cifar-100-python.tar.gz')
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")

if not os.path.exists("cifar-100-python"):
    tarname = "cifar-100-python.tar.gz"
    print("Untarring: {}".format(tarname))
    tar = tarfile.open(tarname)
    tar.extractall()
    tar.close()
else:
    print("cifar-100-python folder already exists. Did not untarring twice\n")

datapath = os.path.join("cifar-100-python")
print("Extracting jpg images and classes from pickle files")

# in CIFAR 100, the files are given in a train and test format
for batch in ['test','train']:

    print("Handling pickle file: {}".format(batch))

    # Create variable which is the exact path to the file
    fpath = os.path.join(datapath, batch)

    # Unpickle the file, and its metadata (classnames)
    f = open(fpath, 'rb')
    labels = pickle.load(open(os.path.join(datapath, 'meta'), 'rb'))
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='bytes')
    # decode utf8 encoded keys, and copy files into new dictionary d_decoded
    d_decoded = {}
    for k, v in d.items():
        d_decoded[k.decode('utf8')] = v

    d = d_decoded
    f.close()

    i=0
    for filename in tqdm(d['filenames']):
        folder = os.path.join('images',
                              labels['fine_label_names'][d['fine_labels'][i]]
        )

        png_path = os.path.join(folder, filename.decode())
        jpg_path = os.path.splitext(png_path)[0]+".jpg"

        if os.path.exists(jpg_path):
            continue
        else:
            if not os.path.exists(folder):
                os.makedirs(folder)
            q = d['data'][i]
            with open(jpg_path, 'wb') as outfile:
                img = Image.fromarray(q.reshape((32, 32, 3), order='F').swapaxes(0,1), 'RGB')
                img.save(outfile)

        i+=1

'''
These files define a few shot learning data set from CIFAR 100 are based on Bertinetto, Luca, et al.
"Meta-learning with differentiable closed-form solvers." arXiv preprint arXiv:1805.08136 (2018).
'''
print("Depending on the split files, organize train, val and test sets")
cls_name = {}
cls_name['train'] = [
    'train', 'skyscraper', 'turtle', 'raccoon', 'spider', 'orange', 'castle', 'keyboard',
    'clock', 'pear', 'girl', 'seal', 'elephant', 'apple', 'aquarium_fish', 'bus',
    'mushroom', 'possum', 'squirrel', 'chair', 'tank', 'plate', 'wolf', 'road', 'mouse',
    'boy', 'shrew', 'couch', 'sunflower', 'tiger', 'caterpillar', 'lion', 'streetcar',
    'lawn_mower', 'tulip', 'forest', 'dolphin', 'cockroach', 'bear', 'porcupine', 'bee',
    'hamster', 'lobster', 'bowl', 'can', 'bottle', 'trout', 'snake', 'bridge',
    'pine_tree', 'skunk', 'lizard', 'cup', 'kangaroo', 'oak_tree', 'dinosaur', 'rabbit',
    'orchid', 'willow_tree', 'ray', 'palm_tree', 'mountain', 'house', 'cloud'
    ]

cls_name['valid'] = [
    'otter', 'motorcycle', 'television', 'lamp', 'crocodile', 'shark', 'butterfly', 'sea',
    'beaver', 'beetle', 'tractor', 'flatfish', 'maple_tree', 'camel', 'crab', 'cattle'
    ]

cls_name['test'] = [
    'baby', 'bed', 'bicycle', 'chimpanzee', 'fox', 'leopard', 'man', 'pickup_truck',
    'plain', 'poppy', 'rocket', 'rose', 'snail', 'sweet_pepper', 'table', 'telephone',
    'wardrobe', 'whale', 'woman', 'worm'
    ]

for datatype in ['train', 'valid', 'test']:
    data_lst = []
    print("Generate preprocessed {} data".format(datatype))
    for img_cls in tqdm(cls_name[datatype]):
        img_path_lst = glob.glob(os.path.join('images', img_cls, '*'))
        img_lst = []
        for img_path in img_path_lst:
            img = Image.open(img_path)
            img_lst.append(np.array(img) / 255.0)
        data_lst.append(np.array(img_lst))
    np.save('{}.npy'.format(datatype), data_lst)

print("Removing redundant files")
shutil.rmtree('images', ignore_errors=True)
os.remove('cifar-100-python.tar.gz')
shutil.rmtree('cifar-100-python', ignore_errors=True)
