import os
import warnings

import numpy as np

from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize
import warnings
import numpy as np
import pandas as pd
from skimage.transform import resize
from keras.models import load_model
from data_preparation import get_train_data, get_test_data
from function import mean_iou, prob_to_rles

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'
SAVE_PATH = 'input_data/'
TRAIN_SAVE_PATH = SAVE_PATH + 'train/'
TEST_SAVE_PATH = SAVE_PATH + 'test/'
NAME_SAVED_IMAGE = 'image.png'
NAME_SAVED_MASK = 'mask.png'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []

print('Getting and resizing images... ')
for PATH, imgs, ids in [(TRAIN_PATH, X_train, train_ids), (TEST_PATH, X_test, test_ids)]:
    for n, id_ in enumerate(ids):
        path = PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        imgs[n] = img

print('Getting and resizing masks... ')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_ + '/masks/'
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    ad = next(os.walk(path))[2]
    for mask_file in next(os.walk(path))[2]:
        if not mask_file.endswith('.png'):
            continue
        mask_ = imread(path + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH),
                                      mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask


# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)

