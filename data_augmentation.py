import os
import warnings
import tqdm

from skimage.transform import rotate
from skimage.io import imread, imsave
import data_preparation as dp
import numpy as np

def data_rotate():

	if not os.path.isdir(dp.TRAIN_SAVE_PATH):
		raise OSError

	ids = next(os.walk(dp.TRAIN_SAVE_PATH))[1]

	print('Rotating images')
	for id_ in tqdm.tqdm(ids, total=len(ids)):
		path = dp.TRAIN_SAVE_PATH + id_ + '/'
		if not os.path.isdir(path):
			raise OSError
			
		items = next(os.walk(path))[2]
		for item in items:
			if item == dp.NAME_SAVED_IMAGE + dp.IMG_FORMAT:
				if os.path.isfile(path + dp.NAME_SAVED_MASK + dp.IMG_FORMAT):
					for nameSaved in [dp.NAME_SAVED_IMAGE, dp.NAME_SAVED_MASK]:
						pic = imread(path + nameSaved + dp.IMG_FORMAT)
						for angle in [90, 180, 270]:
							array_cut_pic = cut_image(rotate(pic,angle))
							for numpic,finalpic in enumerate(array_cut_pic):
								imsave(path + nameSaved + '_' + str(angle)+ '_'+ str(numpic) + dp.IMG_FORMAT, finalpic)

def cut_image(nparr, w_cut = dp.IMG_WIDTH, h_cut = dp.IMG_HEIGHT):
	h_img = nparr.shape[0]
	w_img = nparr.shape[1]
	for i in range(h_cut, h_img + h_cut//2, h_cut//2):
		lower_bound = min(i,h_img)
		for j in range(w_cut, w_img + w_cut//2, w_cut//2):
			right_bound = min(int(j),w_img)
			yield nparr[lower_bound-h_cut : lower_bound][right_bound-w_cut: right_bound]
							
							
if __name__ == "__main__":

	warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
	
	data_rotate()
