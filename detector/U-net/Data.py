import data_augmentation as da
import detector.detector_data_prep as ddp
import os
from tqdm import tqdm
import numpy as np
import data_preparation as dp
from skimage.io import imread
from skimage.transform import resize


def get_data(dir_ids, size=(22, 22, 1), length=None):

    X_train = []
    Y_train = []

    ids = next(os.walk(dir_ids))[1]
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        if length is not None and length < n:
            break

        path_images = dir_ids + id_ + "/images/"
        path_masks = dir_ids + id_ + "/masks/"

        items = next(os.walk(path_images))[2]
        for item in items:
            if os.path.isfile(path_images + item) and item.endswith(dp.IMG_FORMAT) and os.path.isfile(path_masks + item):
                X_train.append(resize(imread(path_images + item, as_grey=True), (size[0], size[1]), mode='constant', preserve_range=True).reshape(size))
                Y_train.append(resize(imread(path_masks + item, as_grey=True).astype(np.bool), (size[0], size[1]), mode='constant', preserve_range=True).reshape(size))

    return np.array(X_train, dtype=np.float64), np.array(Y_train, dtype=np.bool)


if __name__ == "__main__":

    dir_ = "../../../data/detector/"
    ids = next(os.walk(dir_))[1]
    for id_ in tqdm(ids, total=len(ids)):
        for p in ["images/", "masks/"]:
            path = dir_ + id_ + "/" + p
            da.rotate_images_in_directory(path)
