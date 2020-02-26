"""
@author: Hayeon Lee
2020/02/19
Script for downloading, and reorganizing traffic 
for few shot classification
Run this file as follows:
    python get_data.py

"""
import pickle
import os
import numpy as np
from tqdm import tqdm
import requests
import zipfile
from PIL import Image
import glob
import shutil
import pickle
import collections
from scipy.io import loadmat


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


if not os.path.exists("GTSRB_Final_Training_Images.zip"):
    print("Downloading GTSRB_Final_Training_Images.zip\n")
    download_file(
        'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip',
        'GTSRB_Final_Training_Images.zip')
    print("\nDownloading done.\n")
else:
    print("GTSRB_Final_Training_Images.zip has already been downloaded. Did not download twice.\n")

zipname = "GTSRB_Final_Training_Images.zip"
print("Unzipping: {}".format(zipname))
zip = zipfile.ZipFile(zipname)
zip.extractall()
zip.close()

print("Generate preprocessed traffic_sign test.npy data")
data = []
cls_path_lst = glob.glob(os.path.join('GTSRB', 'Final_Training', 'Images', '*'))
for cls_path in tqdm(cls_path_lst):
    cls_data = []
    file_path_lst = glob.glob(os.path.join(cls_path, '*'))
    for file_path in file_path_lst:
        if not 'csv' in file_path:              
            img = Image.open(os.path.join(file_path)).convert('RGB')
            img = np.asarray(img.resize((32, 32))) / 255.0
            cls_data.append(img)
    data.append(np.array(cls_data))
np.save('test.npy', np.array(data))

print('Done')

print("Removing original GTSRB_Final_Training_Images.zip")
os.remove('GTSRB_Final_Training_Images.zip')


