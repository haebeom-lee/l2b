"""
@author: Hayeon Lee
2020/02/19
Script for downloading, and reorganizing vgg_flower 
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


if not os.path.exists("102flowers.tgz"):
    print("Downloading 102flowers.tgz\n")
    download_file(
        'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
        '102flowers.tgz')
    print("\nDownloading done.\n")
else:
    print("102flowers.tgz has already been downloaded. Did not download twice.\n")

if not os.path.exists("imagelabels.mat"):
    print("Downloading imagelabels.mat\n")
    download_file(
        'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
        'imagelabels.mat')
    print("\nDownloading done.\n")
else:
    print("imagelabels.mat has already been downloaded. Did not download twice.\n")

if not os.path.exists("jpg"):
    tarname = "102flowers.tgz"
    print("Untarring: {}".format(tarname))
    tar = tarfile.open(tarname)
    tar.extractall()
    tar.close()
else:
    print("jpg folder already exists. Did not untarring twice\n")

splits = {}
splits['train'] = [
    "090.canna lily", "038.great masterwort", "080.anthurium", "030.sweet william",
    "029.artichoke", "012.colt's foot", "043.sword lily", "027.prince of wales feathers",
    "004.sweet pea", "064.silverbush", "031.carnation", "099.bromelia", 
    "008.bird of paradise", "067.spring crocus", "095.bougainvillea", "077.passion flower",
    "078.lotus", "061.cautleya spicata", "088.cyclamen", "074.rose", "055.pelargonium",
    "032.garden phlox", "021.fire lily", "013.king protea", "079.toad lily", "070.tree poppy",
    "051.petunia", "069.windflower", "014.spear thistle", "060.pink-yellow dahlia?",
    "011.snapdragon", "039.siam tulip", "063.black-eyed susan", "037.cape flower",
    "036.ruby-lipped cattleya", "028.stemless gentian", "048.buttercup", "007.moon orchid",
    "093.ball moss", "002.hard-leaved pocket orchid", "018.peruvian lily", "024.red ginger",
    "006.tiger lily", "003.canterbury bells", "044.poinsettia", "076.morning glory",
    "075.thorn apple", "072.azalea", "052.wild pansy", "084.columbine", "073.water lily",
    "034.mexican aster", "054.sunflower", "066.osteospermum", "059.orange dahlia",
    "050.common dandelion", "091.hippeastrum", "068.bearded iris", "100.blanket flower",
    "071.gazania", "081.frangipani", "101.trumpet creeper", "092.bee balm", 
    "022.pincushion flower", "033.love in the mist", "087.magnolia", "001.pink primrose",
    "049.oxeye daisy", "020.giant white arum lily", "025.grape hyacinth", "058.geranium"
    ]
splits['valid'] = [
    "010.globe thistle", "016.globe-flower", "017.purple coneflower", "023.fritillary",
    "026.corn poppy", "047.marigold", "053.primula", "056.bishop of llandaff", 
    "057.gaura", "062.japanese anemone", "082.clematis", "083.hibiscus", 
    "086.tree mallow", "097.mallow", "102.blackberry lily"
    ]
splits['test'] = [
    "005.english marigold", "009.monkshood", "015.yellow iris", "019.balloon flower",
    "035.alpine sea holly", "040.lenten rose", "041.barbeton daisy", "042.daffodil",
    "045.bolero deep blue", "046.wallflower", "065.californian poppy", "085.desert-rose",
    "089.watercress", "094.foxglove", "096.camellia", "098.mexican petunia"
    ]


imagelabels_path = './imagelabels.mat'
with open(imagelabels_path, 'rb') as f:
  labels = loadmat(f)['labels'][0]
filepaths = collections.defaultdict(list)
for i, label in enumerate(labels):
  filepaths[label].append(
      os.path.join('./jpg', 'image_{:05d}.jpg'.format(i + 1)))

for name in ['valid', 'test', 'train']:
    print("Generate preprocessed {} data".format(name))
    data = []
    for cls_name in tqdm(splits[name]):
        cls_id = int(cls_name[:3])
        cls_data = []
        for file_path in filepaths[cls_id]:
            img = Image.open(file_path)
            img = np.asarray(img.resize((32, 32)))
            cls_data.append(img)
        data.append(np.array(cls_data))
    fname = '%s.npy'%name
    data = np.array(data)
    np.save(fname, data)
    print('Done')

print("Removing original 102flowers.tgz")
os.remove('102flowers.tgz')
shutil.rmtree('jpg', ignore_errors=True)
os.remove('imagelabels.mat')

