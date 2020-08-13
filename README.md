

# Visual Question Answering

## Table of Contents

- [Introduction & Motivation](#introduction)
- [Running the web app](#web_app)
- [Model training](#model_training)
- [Structure of repository](#structure_rep)
- [Contributions](#contributions)

## Introduction & Motivation<a name="introduction"></a>
This project has been carried out as the Capstone project of my [udacity](https://www.udacity.com/) Data Scienctist nanodegree program.

Visual Question Answering (VQA) is a fascinating research field at the intersection of computer vision and language understanding. The objective is to solve the following tasks: Given an image and a natural language question about the image, the VQA model needs to provide an accurate natural language answer.

An article going in more detail can be found here on [Medium](https://medium.com/@frank.merwerth/a-visual-turing-test-for-modern-ai-systems-de7530416e57)

## Running the web app<a name="web_app"></a>
The repository contains all necessary data to run the web app demontrator without training. 
The web app is programmed in Python 3 and requires the following libraries:

* os
* pandas
* numpy
* json
* plotly
* flask
* tensorflow
* keras
* spacy - en_core_web_lg

To start the app, follow these steps:

>1. Change into the directory the / app

>2. Run the following command in the app's directory to run your web app.
>
    ```python run.py```

>3. Open the following web address in a browser:  http://0.0.0.0:3001/, http://localhost:3001/ respectively

## Model training<a name="model_training"></a>
Training the model requires additional packages:

* tqdm

Before training the model, you first have to download the training data from [visualqa.org](https://visualqa.org/download.html). These dataset are not included in the github repo due to their size.

Following steps have to be taken:

* Download and unzip [Training annotations 2017 v2.0](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip) to the folder **data/Annotations**
* Download and unzip [Training questions 2017 v2.0](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip) to the folder **data/Questions**
* Download and unzip the [Training images](http://images.cocodataset.org/zips/train2014.zip) to the folder **data/images/train2014**

Before starting the training some preprocessing is required. Therefore run the following command in the main directory

    ```python vqa_prepro.py```

Start the training by running the following command in the main directory:

    ```python vqa_train.py```


>Due to the size of the training data the training is extremely computational greedy. Be prepared that it can easily run for several hours.

##Structure of repository<a name="structure_rep"></a>

```
- app
| - template
| |- header.html       # header layout that all the other pages inherit from 
| |- index.html        # main page to introduce the VQA topic
| |- EDA.html          # Exploratory data analysis on the training data 
| |- demo.html         # intercative platform to try out the algorithm
| |- performance.html  # illustration of the evaluation performance
| - static
| | - images
| | | - train          # folder containing excerpt of training images
| | | - val            # folder containing excerpt of validation images
| | - models
| | | - VQA_MODEL_WEIGHTS.hdf5    # trained weights of VQA model
| | | - VQA_model.py              # library containing several function regarding the VQA model including its architecture
| | | - label_encoder.csv         # label encoder of 1000 most frequent answers
| | | - v2_OpenEnded_mscoco_***_final_accuracy.json  # evaluation results for training / validation set
| | - scripts
| | | - main.js        # javascript-file for interactive elements of web app
| | - styles
| | | -main.css        # css-file on web app styles
|- run.py              # Flask file that runs app

- data
|- VQA_EDA.db          # database containing results of Exploratory data analysis (EDA.py)

- EDA.py                      # Exploratory data analysis functions - results stored in data/VQA_EDA.db
- VQA_EDA.ipynb               # corresponding jupyter-notebook
- ElementCountVectorizer.py   # Class to count vectorize the annotaions efficiently
- path_dict.json              # central storage-file for file-/folder-pathes
- vqa_prepro                  # image preprocessing steps before training
- vqa_train                   # training of VQA model

- README.md
```

## Contributions<a name="contributions"></a>

* The VQA approach has mainly derived from material available on [visualqa.org](https://visualqa.org)
* models has been validated using the tools available on https://github.com/tejaskhot/VQA-Evaluation
* ElementCountVectorizer is inspired by the MyCountVectorizer class posted by SongDark on https://github.com/SongDark/MyCountVectorizeranalysis on the 