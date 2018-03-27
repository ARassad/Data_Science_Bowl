import warnings
import numpy as np
import pandas as pd
import os
import pickle

from skimage.transform import resize
from keras.models import load_model
import data_preparation as dp
from function import mean_iou, prob_to_rles
import data_augmentation as da
from skimage.io import imread, imsave
from tqdm import tqdm
from math import sqrt


def predict_Unet(data, ids, log=True):
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    model = load_model('model-dsbowl2018-1(0.304).h5', custom_objects={'mean_iou': mean_iou})

    preds_test_upsampled = []

    print("Predict masks")
    for i, d in tqdm(enumerate(data), total=len(data)):
        img = d
        h_img = img.shape[0]
        w_img = img.shape[1]

        curimg = np.array(
            [pic[:dp.IMG_HEIGHT, :dp.IMG_WIDTH].reshape(128, 128, 1) for pic in da.cut_image(img)])
        curimg = curimg * 255
        curimg = curimg.astype(np.uint8)
        f = model.predict(curimg, verbose=1)
        pred_img = da.glue_image(f, h_img, w_img, func_merg='max').reshape(h_img, w_img)
        if log:
            imsave("../data/OUTP/true_" + str(i) + ".png", pred_img)
        preds_test_upsampled.append(pred_img)

    new_test_ids = []
    rles = []
    res_encod = []
    for n, id_ in enumerate(ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        res_encod.append(rle)
        new_test_ids.extend([id_] * len(rle))

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    if log:
        sub.to_csv('sub-dsbowl2018-1.csv', index=False)

    return sub, new_test_ids, rles, res_encod


def compute_median_size_nucleas(data, ids):
    _, _, _, encod_px = predict_Unet(data, ids, log=True)

    res_size = {}

    for k in range(len(encod_px)):

        sizes = []

        for i in range(len(encod_px[k])):

            x_min, y_min = 10e10, 10e10
            x_max, y_max = -1, -1

            x_shape = data[k].shape[1]
            y_shape = data[k].shape[0]

            y_min = int(encod_px[k][i][0] / y_shape)
            y_max = int((encod_px[k][i][-2] + encod_px[k][i][-1]) / y_shape)

            for j in range(0, len(encod_px[k][i]), 2):
                x_tmp = encod_px[k][i][j] % x_shape

                x_min = min(x_min, x_tmp)
                x_max = max(x_max, x_tmp)

                if x_tmp + encod_px[k][i][j+1] > x_shape:
                    x_max = x_shape
                    x_min = 0
                else:
                    x_max = max(x_max, x_tmp + encod_px[k][i][j+1])

            diag = sqrt(((x_max-x_min)**2) + ((y_max-y_min)**2))
            sizes.append(diag)

        sizes.sort()
        length = len(sizes)
        try:
            median = sizes[int(length / 2)] if length % 2 else (sizes[int(length / 2) - 1] + sizes[int(length / 2)]) / 2
        except:
            print("len : {} ids : {} image : {}".format(len(sizes), ids[k], encod_px[k]))
        res_size[ids[k]] = median

    return res_size


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":

    dir_ = "../data/stage1_test/"
    #dir_ = "../data/1/"
    # Считывание данных
    test_data = []
    print("Read data")
    ids = next(os.walk(dir_))[1]
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        path = dir_ + id_ + "/images/"
        test_data.append(imread(path + id_ + ".png", as_grey=True))

    res_dict = compute_median_size_nucleas(test_data, next(os.walk(dir_))[1])

    save_obj(res_dict, "sizes_nuclears")
