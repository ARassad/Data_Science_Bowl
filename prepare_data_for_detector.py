

import os
import numpy as np

from tqdm import tqdm


from skimage.io import imread, imsave
from skimage.transform import resize
from data_preparation import get_source_data


TRAIN_PATH = '../data/stage1_train/'
TEST_PATH = '../data/stage1_test/'
SAVE_PATH = '../data/input_data/'
TRAIN_SAVE_PATH = SAVE_PATH + 'train/'
TEST_SAVE_PATH = SAVE_PATH + 'test/'
NAME_SAVED_IMAGE = 'image'
NAME_SAVED_MASK = 'mask'
IMG_FORMAT = '.png'


def get_train_data(out_size=None):
    if not os.path.isdir(TRAIN_PATH):
        raise OSError

    ids = next(os.walk(TRAIN_PATH))[1]

    images = []
    masks = []
    image_id = []

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        if out_size is not None and out_size < n:
            break

        path = TRAIN_PATH + id_ + '/' 

        if not os.path.isdir(path):
            raise OSError
        image_id.append(id_)

        images.append(imread(path + "images/" + id_ + IMG_FORMAT).astype(np.uint8))

        mask = np.zeros(images[-1].size)
        items = next(os.walk(path + "masks/" ))[2]
        for name_mask in items:
            name_mask = name_mask
            if os.path.isfile(path + "masks" + name_mask):
                mask = np.maximum(mask, item)

        

    return np.array(images), np.array(masks).astype(np.bool), np.array(image_id)


training_images_, training_masks_, train_ids_ = get_train_data()

sizes = (20, 20)

goodImages = []
coords = []

for image, mask, i, id in training_images_, training_masks_, enumerate(train_ids_):
    goodImages.append([])
    x, y = size[0], size[1]
    imsizes = image.sizes()
    while x < imsizes[0]:
        while y < imsizes[1]:
            nextHorizontal = [im[x-ix][y+1] for ix in range(sizes[0])]
            nextVertical = [im[x+1][y-iy] for iy in rannge(sizes[1])]
            if not any(nextHorizontal):
                if not any(nextVertical):
                    goodImages.append(image[y+1 - size[1] : y+1+1][x+1-size[0]:x+1+1])
                    
                else:
                   x = x + size[0]-1
            else:
                y = y + size[1]-1
            y = y + 1
        x = x + 1


for i, id in enumerate(train_ids_):
    for j, pi in enumerate(goodImages):
        imsave("../data/not_nuclei" + "/" + id + "/" + j, pi)
