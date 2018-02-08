import os
import warnings
import numpy as np

from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'data/stage1_train/'
TEST_PATH = 'data/stage1_test/'
SAVE_PATH = 'data/input_data/'
TRAIN_SAVE_PATH = SAVE_PATH + 'train/'
TEST_SAVE_PATH = SAVE_PATH + 'test/'
NAME_SAVED_IMAGE = 'image'
NAME_SAVED_MASK = 'mask'
IMG_FORMAT = '.png'


def get_train_data():
    if not os.path.isdir(TRAIN_SAVE_PATH):
        raise OSError

    ids = next(os.walk(TRAIN_SAVE_PATH))[1]

    images = []
    masks = []
    image_id = []

    for n, id_ in enumerate(ids):
        path = TRAIN_SAVE_PATH + id_ + '/'

        if not os.path.isdir(path):
            raise OSError

        items = next(os.walk(path))[2]
        for item in items:
            if item.startswith(NAME_SAVED_IMAGE) and item.endswith(IMG_FORMAT):
                name_mask = NAME_SAVED_MASK + item[len(NAME_SAVED_IMAGE):]
                if os.path.isfile(path + name_mask):
                    images.append(imread(path + item)[:, :, :IMG_CHANNELS])
                    masks.append(imread(path + name_mask).reshape((128, 128, 1)))
                    image_id.append(id_)

    return np.array(images), np.array(masks).astype(np.bool), np.array(image_id)


def get_test_data():

    if not os.path.isdir(TEST_SAVE_PATH):
        raise OSError

    # Get IDs
    ids_test = next(os.walk(TEST_SAVE_PATH))[1]

    # Get and resize images and masks
    images_test = np.zeros((len(ids_test), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    # Sizes primary imgs
    size_test = []

    for n, id_ in enumerate(ids_test):
        path = TEST_SAVE_PATH + id_ + '/'

        if not os.path.isdir(path):
            raise OSError

        items = next(os.walk(path))[2]
        for item in items:
            if item.startswith(NAME_SAVED_IMAGE) and item.endswith(IMG_FORMAT):
                images_test[n] = imread(path + item)[:, :, :IMG_CHANNELS]

        size_test.append(np.load(path + 'sizes_test.npy'))

    return images_test, ids_test, size_test


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    # Get train and test IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    # Get and resize train images and masks
    training_images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    training_masks = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    testing_images = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []

    print('Getting and resizing images... ')
    for PATH, imgs, ids, sizes in [(TRAIN_PATH, training_images, train_ids, None),
                                   (TEST_PATH, testing_images, test_ids, sizes_test)]:
        for n, id_ in enumerate(ids):
            path = PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
            if sizes is not None:
                sizes.append([img.shape[0], img.shape[1]])
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
        training_masks[n] = mask

    print("Saving images and masks")
    for PATH, images, masks, ids, sizes in [(TRAIN_SAVE_PATH, training_images, training_masks, train_ids, None),
                                            (TEST_SAVE_PATH, testing_images, None, test_ids, sizes_test)]:
        for n, id_ in enumerate(ids):
            path = PATH + id_

            if not os.path.isdir(path):
                os.makedirs(path)

            if images is not None:
                imsave(path + "/" + NAME_SAVED_IMAGE + IMG_FORMAT, images[n])

            if masks is not None:
                imsave(path + "/" + NAME_SAVED_MASK + IMG_FORMAT, masks[n].reshape((IMG_WIDTH, IMG_HEIGHT)))

            if sizes is not None:
                np.save(path + '/sizes_test', sizes[n])
