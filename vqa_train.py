import pandas as pd
import numpy as np
import json
import en_core_web_lg
from tqdm import tqdm
from app.static.models.VQA_model import get_question_features, build_VQA_model
from EDA import read_pathDict, load_questions, load_annotations

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def vqa_train(num_epochs = 5):
	'''
	Trains a VQA network
	INPUT
	num_epochs		number of epochs for training
	OUTPUT
	vqa_model       trained model
	'''
	
	annFile, quesFile, imgDir = read_pathDict()

	# load questions
	print('Loading questions from training set...')
	df_questions = load_questions(quesFile)
	print("Number of open ended questions loaded :  ", df_questions.shape[0])

	# load images features
	print('Loading image features...')
	im_features = json.load(open('app/static/models/im_features_train.json', 'r'))
	df_im_features =pd.DataFrame.from_dict(im_features)

	# load annotions
	print('Loading annotations from training set...')
	df_annotations = load_annotations(annFile, df_questions)
	print("Number of annotations loaded :  ", df_annotations.shape[0])

	# filter questions
	
	im_list = list(df_im_features['image'])
	im_list_cut = []
	for im in im_list:
		id = im[-12:]
		id = int(id[:8])
		im_list_cut.append(id)

	df_questions = df_questions[df_questions['image_id'].isin(im_list_cut)]
	

	# load label encoder
	label_encoder_file_name  = 'app/static/models/label_encoder.csv'

	word_embeddings = en_core_web_lg.load()
	label_encoder = pd.read_csv(label_encoder_file_name)

	# prepare model
	vqa_model = build_VQA_model()
	vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	
	## training
	print('Training started...')

	for k in range(num_epochs):
		print('Epoch {}/{}'.format(k+1,num_epochs))

		for row in tqdm(range(df_questions.shape[0])):
			x_q = get_question_features(df_questions.iloc[k].question, word_embeddings)
			x_i = np.asarray(list(df_im_features[df_im_features['image']=='COCO_train2014_'+str(df_questions.iloc[k].image_id).zfill(12)+'.jpg'].features))
			y = 1 * label_encoder['token'].isin(df_annotations[df_annotations['question_id'] == df_questions.iloc[k].question_id].answers_extracted)
			y = np.expand_dims(y, axis=0)
			loss = vqa_model.train_on_batch([x_i, x_q], y)

	return vqa_model

if __name__ == '__main__':
    vqa_train()