import warnings
import numpy as np
import pandas as pd

from skimage.transform import resize
from skimage.io import imsave
from keras.models import load_model
from data_preparation import get_test_data
from function import mean_iou, prob_to_rles
import matplotlib.pyplot as plt
import os
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from data_preparation import IMG_WIDTH, IMG_HEIGHT

X_test, test_ids = get_test_data()
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})

def onePredict( oneImg):
    imwidth = oneImg.shape[0]
    imheight = oneImg.shape[1]
    predictMatr = []
    for ix, x in enumerate(range(0, imwidth, IMG_WIDTH)):
        linePredicts = []
        lineSlices = []
        for iy, y in  enumerate(range(0, imheight, IMG_HEIGHT)):
            imPartWidth = min(IMG_WIDTH, imwidth-x)
            imPartHeight = min(IMG_HEIGHT, imheight-y)
            slicedImg = np.zeros((IMG_WIDTH, IMG_HEIGHT))
           # работает
            slicedImg[0:imPartWidth, 0:imPartHeight] = oneImg[x:x+imPartWidth,y:y+imPartHeight]
            
            lineSlices.append(slicedImg.reshape(IMG_WIDTH, IMG_HEIGHT,1))

            #oneSlicePredict = model.predict([slicedImg], verbose = 1)
            
        linePredicts = model.predict(np.array( lineSlices))
        predictMatr.append(linePredicts)

    result = np.ndarray(oneImg.shape)#.reshape(imwidth, imheight, 1)
    for ix, x in enumerate(range(0, imwidth, IMG_WIDTH)):
        for iy, y in  enumerate(range(0, imheight, IMG_HEIGHT)):
            imPartWidth = min(IMG_WIDTH, imwidth-x)
            imPartHeight = min(IMG_HEIGHT, imheight-y)

            result[x:x+imPartWidth,y:y+imPartHeight] = predictMatr[ix][iy:iy+1,0:imPartWidth, 0:imPartHeight].reshape(imPartWidth, imPartHeight)
   
    #plt.imshow(result) 

    return result;

def predictAll(imgList):
    predicts = []
    #os.makedirs("predicts")
    for i,img in enumerate( imgList):
        predict = onePredict(img);
        
        predicts.append(predict)
    return predicts

# Predict
#preds_test = model.predict(X_test, verbose=1)
preds_test = predictAll(X_test)
# Create list of upsampled test masks
#preds_test_upsampled = []
#for i in range(len(preds_test)):
#    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1]),
#                                mode='constant', preserve_range=True))

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    #rle = list(prob_to_rles(preds_test_upsampled[n]))
    pr_to_rle = prob_to_rles(preds_test[n])
    #imsave("predicts/" + str(n) + ".png",pr_to_rle)
    rle = list(pr_to_rle)
    
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))


# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)
