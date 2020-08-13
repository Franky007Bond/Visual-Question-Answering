import csv
import os
import pandas as pd
import json
from tqdm import tqdm
from app.static.models import VQA_model

def vqa_prepro():
	'''
	This procedure writes the features of all training images to a json file that is read-in before training
	This approach is computationally more efficient than retrieve the image features directly during training
	INPUT
	None
	OUTPUT
	None
	'''
	# retrieve list of all available images
	images = os.listdir('images/train2014')

	# load pretrained VGG model
	mod_vgg_pretrained = VQA_model.build_VGG()

	image_features = []

	for im in tqdm(images):
		features = VQA_model.get_image_features('images/train2014/'+im, mod_vgg_pretrained)
		image_features.append({'image':im, 'features':features})

	df_image_features = pd.DataFrame(image_features)

	df_image_features.to_json('app/static/models/im_features_train.json', orient='records')

if __name__ == '__main__':
    vqa_prepro()