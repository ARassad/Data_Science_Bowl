import warnings
import numpy as np
import pandas as pd

from skimage.transform import resize
from keras.models import load_model
from data_preparation import get_test_data
from function import mean_iou, prob_to_rles

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

X_test, test_ids, sizes_test = get_test_data()
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})



# Predict
preds_test = model.predict(X_test, verbose=1)



# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1]),
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
