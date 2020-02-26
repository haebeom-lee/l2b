"""
@author: Hayeon Lee
2020/02/19
Script for downloading, and reorganizing aircraft 
for few shot classification
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
import collections


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

if not os.path.exists("fgvc-aircraft-2013b.tar.gz"):
    print("Downloading fgvc-aircraft-2013b.tar.gz\n")
    download_file(
        'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz',
        'fgvc-aircraft-2013b.tar.gz')
    print("\nDownloading done.\n")
else:
    print("fgvc-aircraft-2013b.tar.gz has already been downloaded. Did not download twice.\n")

if not os.path.exists("fgvc-aircraft-2013b"):
    tarname = "fgvc-aircraft-2013b.tar.gz"
    print("Untarring: {}".format(tarname))
    tar = tarfile.open(tarname)
    tar.extractall()
    tar.close()
else:
    print("fgvc-aircraft-2013b folder already exists. Did not untarring twice\n")

splits = {}
splits['train'] = [
    "A340-300", "A318", "Falcon 2000", "F-16A/B", "F/A-18", "C-130", 
    "MD-80", "BAE 146-200", "777-200", "747-400", "Cessna 172", "An-12", 
    "A330-300", "A321", "Fokker 100", "Fokker 50", "DHC-1", "Fokker 70", 
    "A340-200", "DC-6", "747-200", "Il-76", "747-300", "Model B200", 
    "Saab 340", "Cessna 560", "Dornier 328", "E-195", "ERJ 135", "747-100", 
    "737-600", "C-47", "DR-400", "ATR-72", "A330-200", "727-200", "737-700", 
    "PA-28", "ERJ 145", "737-300", "767-300", "737-500", "737-200", "DHC-6", 
    "Falcon 900", "DC-3", "Eurofighter Typhoon", "Challenger 600", "Hawk T1", 
    "A380", "777-300", "E-190", "DHC-8-100", "Cessna 525", "Metroliner", 
    "EMB-120", "Tu-134", "Embraer Legacy 600", "Gulfstream IV", "Tu-154", 
    "MD-87", "A300B4", "A340-600", "A340-500", "MD-11", "707-320", 
    "Cessna 208", "Global Express", "A319", "DH-82"
    ]
splits['test'] = [
    "737-400", "737-800", "757-200", "767-400", "ATR-42", "BAE-125", 
    "Beechcraft 1900", "Boeing 717", "CRJ-200", "CRJ-700", "E-170", 
    "L-1011", "MD-90", "Saab 2000", "Spitfire"
    ]
splits['valid'] = [
    "737-900", "757-300", "767-200", "A310", "A320", "BAE 146-300", 
    "CRJ-900", "DC-10", "DC-8", "DC-9-30", "DHC-8-300", "Gulfstream V", 
    "SR-20", "Tornado", "Yak-42"
    ]


# Cropping images with bounding box same as meta-dataset.
bboxes_path = os.path.join('./fgvc-aircraft-2013b', 'data', 'images_box.txt')
with open(bboxes_path, 'r') as f:
  names_to_bboxes = [
      line.split('\n')[0].split(' ') for line in f.readlines()]
  names_to_bboxes = dict(
      (name, map(int, (xmin, ymin, xmax, ymax)))
      for name, xmin, ymin, xmax, ymax in names_to_bboxes)
# Retrieve mapping from filename to cls
cls_trainval_path = os.path.join('./fgvc-aircraft-2013b', 'data', 'images_variant_trainval.txt')
with open(cls_trainval_path, 'r') as f:
  filenames_to_clsnames = [
      line.split('\n')[0].split(' ', 1) for line in f.readlines()]

cls_test_path = os.path.join('./fgvc-aircraft-2013b', 'data', 'images_variant_test.txt')
with open(cls_test_path, 'r') as f:
  filenames_to_clsnames += [
      line.split('\n')[0].split(' ', 1) for line in f.readlines()]
filenames_to_clsnames = dict(filenames_to_clsnames)
clss_to_names = collections.defaultdict(list)
for filename, cls in filenames_to_clsnames.items():
  clss_to_names[cls].append(filename)

for name in ['valid', 'test', 'train']:
    print("Generate preprocessed {}.npy data".format(name))
    data = []
    for class_id, cls_name in enumerate(tqdm(splits[name])):
        cls_data = []
        for filename in sorted(clss_to_names[cls_name]):
            file_path = os.path.join(
                            './fgvc-aircraft-2013b', 
                            'data',
                            'images', 
                                    '{}.jpg'.format(filename))
            img = Image.open(file_path)
            bbox = names_to_bboxes[filename]
            img = np.asarray(img.crop(bbox).resize((32, 32)))
            cls_data.append(img / 255.0)
        data.append(np.array(cls_data))
    fname = '%s.npy'%name
    data = np.array(data)
    np.save(fname, data)
    print('Done')

print("Removing original fgvc-aircraft-2013b.tar.gz")
os.remove('fgvc-aircraft-2013b.tar.gz')
shutil.rmtree('fgvc-aircraft-2013b', ignore_errors=True)
