import cv2
import argparse
import numpy as np
import tensorflow
from tensorflow.keras.layers import LSTM, Reshape, Dense, Dropout, Activation
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import concatenate, Concatenate
from tensorflow.keras.layers import Input


def build_VGG():
    '''
    builds a keras VGG-16 model trained on imagenet data without the classification (softmax) layer 
    as this layer is not required for the VQA model
    '''

    from tensorflow.keras.applications.vgg16 import VGG16

    tensorflow.keras.backend.set_image_data_format('channels_last')
    input_tensor = Input(shape=(224, 224, 3))
    model = VGG16(weights='imagenet', input_shape=(224, 224, 3))
    model_vgg = Sequential()

    # cut the last layer (softmax classification)
    for layer in model.layers[:-1]:
        model_vgg.add(layer)

    # Freeze the layers 
    for layer in model_vgg.layers:
        layer.trainable = False

    return model_vgg


def get_image_features(image_file_name, CNN_model, input_size=(224,224)):
    ''' 
    Return the features for a given image when applying the CNN_model
    INPUT
    image_file_name		string containing full path to image file
    CNN_model 			Deep Convolutional Neural network to be used
    input_size			input size of the CNN (default: VGG16 input size)
    OUTPUT
    image_features		features for the given image
    					in case of VGG 16 model it returns a 1, 4096 dimension vector 
    '''

    # resize image to CNN input size
    # e.g. since VGG was trained on images of 224x224 px, the image
    # needs to be transformed accordingly	
    im = cv2.resize(cv2.imread(image_file_name), input_size)

    # this axis dimension is required because VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size)
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0) 

    # forward propagate the image through the CNN
    image_features = CNN_model.predict(im)[0]

    return image_features


def get_question_features(question, word_embeddings, tensor_dims=(1, 30, 300)):
    ''' 
    For a given question in form of a unicode string, returns the time series vector
    with each word (token) transformed using word_embeddings
    INPUT
    question 			string containing question
    word_embeddings		word_embeddings to be used, e.g. Glove
    tensor_dims			dimension of question tensor (default: Glove dimensions)
    OUTPUT
    question_tensor		time series vector of embedded tokens
    '''
   
    tokens = word_embeddings(question)
    question_tensor = np.zeros((tensor_dims))
   
    for j in range(len(tokens)):
        question_tensor[0,j,:] = tokens[j].vector

    return question_tensor


def build_VQA_model():
    '''
    builds a keras VQA model:
    - Question token GloVe vectores first pass a Long-Short-Term-Memory network to derive features
    - Question and image features are simply concatenated
    - concatenated features are passed through additonal fully-connected layers
    - final step is a softmax classification based on the most frequent answers
    '''

    img_dim                 =     4096
    glove_dim               =      300
    num_hidden_nodes_mlp    =     1024
    num_hidden_nodes_lstm   =      512
    num_layers_lstm         =        3
    dropout                 =      0.5
    activation_mlp          =    'tanh'
    upper_lim               =     1000
    embedding_vector_length =       30

    mod_lstm = Sequential()

    mod_lstm.add(LSTM(units=num_hidden_nodes_lstm,  
                        return_sequences=True, input_shape=(embedding_vector_length, glove_dim)))

    for i in range(num_layers_lstm-2):
        mod_lstm.add(LSTM(units=num_hidden_nodes_lstm, return_sequences=True))
        mod_lstm.add(LSTM(units=num_hidden_nodes_lstm,  return_sequences=False))

    mod_img = Sequential()
    mod_img.add(Reshape(input_shape = (img_dim,), target_shape=(img_dim,)))

    concatted = concatenate([mod_lstm.output, mod_img.output])

    model = Dense(num_hidden_nodes_mlp,  activation = activation_mlp)(concatted)
    model = Dropout(dropout)(model)

    model = Dense(num_hidden_nodes_mlp,  activation = activation_mlp)(model)
    model = Dropout(dropout)(model)

    model = Dense(num_hidden_nodes_mlp, activation = activation_mlp)(model)
    model = Dropout(dropout)(model)

    model = Dense(upper_lim)(model)
    model = Activation("softmax")(model)

    model = Model(inputs=[mod_img.input, mod_lstm.input], outputs=model)

    return model


def predict_VQA_by_image_features(image_features, question, vqa_model, label_encoder, word_embeddings, ques_tensor_dims=(1, 30, 300), top_answers=3):
    '''
    predicts top_answers to a given image-question pair 
    INPUT
    image_features      extracted features of the image
    question            string containing question
    vqa_model           Deep neural VQA network to be used for prediction 
    label_encoder       label_encoder to be used to map NN network to answers
    word_embeddings     word_embeddings to be used, e.g. Glove
    ques_tensor_dims    dimension of question tensor (default: Glove dimensions)
    im_input_size       input size of the image CNN (default: VGG16 input size)
    top_answers         number of top answers to be returned
    OUTPUT
    answers             list of dicts containing top answers and corresponding probabilities
    '''

    # adjust dimensions of image feature vestor
    image_features = np.expand_dims(image_features, axis=0)

    # get time series vector of embedded tokens of question
    question_features = get_question_features(question, word_embeddings, ques_tensor_dims)
  
    # get model predictions
    y_output = vqa_model.predict([image_features, question_features])

    answers = []

    for label in reversed(np.argsort(y_output)[0,-top_answers:]):
        answers.append({'answer': label_encoder.iloc[label,1], 'probability': "{:.1%}".format(y_output[0,label])})

    return answers


def predict_VQA_by_image(image_file_name, question, vqa_model, label_encoder, word_embeddings, CNN_model, ques_tensor_dims=(1, 30, 300), im_input_size=(224,224), top_answers=3):
    '''
    predicts top_answers to a given image-question pair
    INPUT
    image_file_name     string containing full path to image file
    question            string containing question
    vqa_model           Deep neural VQA network to be usedtensor_dims=(1, 30, 300), 
    label_encoder       label_encoder to be used to map NN network to answers
    word_embeddings     word_embeddings to be used, e.g. Glove
    CNN_model           Deep Convolutional Neural network to be used for image processing
    ques_tensor_dims         dimension of question tensor (default: Glove dimensions)
    im_input_size       input size of the image CNN (default: VGG16 input size)
    top_answers         number of top answers to be returned
    OUTPUT
    answers             list of dicts containing top answers and corresponding probabilities
    '''

    # forward propagate image through CNN to extract image features
    image_features = get_image_features(image_file_name, CNN_model, im_input_size)

    # resuse above defiend function
    answers = predict_VQA_by_image_features(image_features, question, vqa_model, label_encoder, word_embeddings, ques_tensor_dims, top_answers)

    return answers