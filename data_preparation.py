import os
import warnings
import numpy as np

from tqdm import tqdm

from skimage.io import imread, imsave
from skimage.transform import resize

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
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

    print("get data...")
    for n, id_ in tqdm(enumerate(ids), total = len(ids)):
        path = TRAIN_SAVE_PATH + id_ + '/'

        if not os.path.isdir(path):
            raise OSError

        items = next(os.walk(path))[2]
        for item in items:
            if item.startswith(NAME_SAVED_IMAGE) and item.endswith(IMG_FORMAT):
                name_mask = NAME_SAVED_MASK + item[len(NAME_SAVED_IMAGE):]
                if os.path.isfile(path + name_mask):
                    images.append(imread(path + item).reshape(IMG_WIDTH, IMG_HEIGHT, 1))
                    masks.append(imread(path + name_mask).reshape(IMG_WIDTH, IMG_HEIGHT, 1))
                    image_id.append(id_)

    return np.array(images), np.array(masks).astype(np.bool), np.array(image_id)


def get_test_data():

    if not os.path.isdir(TEST_SAVE_PATH):
        raise OSError

    # Get IDs
    ids_test = next(os.walk(TEST_SAVE_PATH))[1]

    # Get and resize images and masks
    images_test = [[]] * len(ids_test) # = np.zeros((len(ids_test), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    # Sizes primary imgs
    size_test = []

    for n, id_ in enumerate(ids_test):
        path = TEST_SAVE_PATH + id_ + '/'

        if not os.path.isdir(path):
            raise OSError

        items = next(os.walk(path))[2]
        for item in items:
            if item.startswith(NAME_SAVED_IMAGE) and item.endswith(IMG_FORMAT):
                images_test[n] = imread(path + item)

        #size_test.append(np.load(path + 'sizes_test.npy'))

    return images_test, ids_test#, size_test


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    # Get train and test IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]
    img_names = [[[]] * len(train_ids), [[]] * len(test_ids)]
    imgs = [[[]] * len(train_ids), [[]] * len(test_ids)]
    masks = [[]] * len(train_ids)
    masks_names = [[]] * len(train_ids)
    # Get and resize train images and masks
    #training_images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    #training_masks = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    #testing_images = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    #sizes_test = []

    print('Getting and slicing images... ')
 
    imgs_group = 0
    for PATH, ids in [(TRAIN_PATH, train_ids),
                                  (TEST_PATH, test_ids)]:
        
        for n, id_ in tqdm(enumerate(ids), total = len(ids)):
            path = PATH + id_
            img = imread(path + '/images/' + id_ + '.png', as_grey = True)
            if imgs_group == 0:
                imwidth = img.shape[0]
                imheight = img.shape[1]
                slicedImgs = []
                slicedImgsNames = []
                for ix, x in enumerate(range(0, imwidth, IMG_WIDTH)):
                    for iy, y in  enumerate(range(0, imheight, IMG_HEIGHT)):
                        imPartWidth = min(IMG_WIDTH, imwidth-x)
                        imPartHeight = min(IMG_HEIGHT, imheight-y)
                        slicedImg = np.zeros((IMG_WIDTH, IMG_HEIGHT))
                    
                        # работает
                        slicedImg[0:imPartWidth, 0:imPartHeight] = img[x:x+imPartWidth,y:y+imPartHeight]
                        slicedImgs.append(slicedImg)
                        slicedImgsNames.append(NAME_SAVED_IMAGE + '_' + str(ix) + '_' + str(iy))
                        imgs[imgs_group][n] = slicedImgs
                        img_names[imgs_group][n] = slicedImgsNames
            else:
                imgs[imgs_group][n] = img
                img_names[imgs_group][n] = NAME_SAVED_IMAGE

        imgs_group += 1
      
            
            
            #if sizes is not None:
            #    sizes.append([img.shape[0], img.shape[1]])
            #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            

    print('Getting and slicing masks... ')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_ + '/masks/'
        
        
        mask = imread(path + next(os.walk(path))[2][0], as_grey = True)
        
        #ad = next(os.walk(path))[2]
        for mask_file in next(os.walk(path))[2]:
            if not mask_file.endswith('.png'):
                continue
            mask_ = imread(path + mask_file,as_grey = True)

            mask = np.maximum(mask, mask_)

        mask = mask / 255

        maskwidth = mask.shape[0]
        maskheight = mask.shape[1]
        slicedMasks = []
        slicedMasksNames = []
        for ix, x in enumerate(range(0, maskwidth, IMG_WIDTH)):
            for iy, y in  enumerate(range(0, maskheight, IMG_HEIGHT)):
                maskPartWidth = min(IMG_WIDTH, maskwidth-x)
                maskPartHeight = min(IMG_HEIGHT, maskheight-y)
                slicedMask= np.zeros((IMG_WIDTH, IMG_HEIGHT))
                # работает
                slicedMask[ 0:maskPartWidth, 0:maskPartHeight ] = mask[ x:x+maskPartWidth, y:y+maskPartHeight]
                slicedMasks.append(slicedMask)
                slicedMasksNames.append(NAME_SAVED_MASK + '_' + str(ix) + '_' + str(iy))
        masks[n] = slicedMasks
        masks_names[n] = slicedMasksNames
            
        
    print("Saving images and masks")
    for PATH, ids, images, masks, i_names, m_names in [(TRAIN_SAVE_PATH, train_ids, imgs[0], masks, img_names[0], masks_names),
                                            (TEST_SAVE_PATH, test_ids, imgs[1], None, img_names[1], None)]:
        for n, id_ in enumerate(ids):
            path = PATH + id_

            if not os.path.isdir(path):
                os.makedirs(path)

            if images is not None:
                for i, img_ in enumerate(images[n]):
                    imsave(path + "/" + i_names[n][i] + IMG_FORMAT, img_)

            if masks is not None:
                for i, mask_ in enumerate(masks[n]):
                    imsave(path + "/" + m_names[n][i] + IMG_FORMAT, mask_)

