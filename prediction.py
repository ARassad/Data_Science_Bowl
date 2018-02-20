import warnings
import numpy as np
import pandas as pd
import os

from skimage.transform import resize
from keras.models import load_model
import data_preparation as dp
from function import mean_iou, prob_to_rles
import data_augmentation as da
from skimage.io import imread, imsave

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})

preds_test_upsampled = []
ids_test = next(os.walk(dp.TEST_PATH))[1]
for i,id_ in enumerate(ids_test):
    path = dp.TEST_PATH + id_ + '/' + 'images/'
    items = next(os.walk(path))[2]
    for item in items:
        if item.endswith(dp.IMG_FORMAT):
            img = imread(path + item, as_grey=True)
            h_img = img.shape[0]
            w_img = img.shape[1]
            
            curimg = da.cut_image(img)
            f = model.predict(curimg, verbose=1)
            #imsave("data/OUTP/true_" + str(i) + ".png", da.glue_image(f, h_img, w_img))
            preds_test_upsampled[i] = da.glue_image(f, h_img, w_img)


# Create list of upsampled test masks

#for i in range(len(preds_test)):
#preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1]),
#mode='constant', preserve_range=True))

new_test_ids = []
rles = []
for n, id_ in enumerate(ids_test):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)
