import os
import json
import plotly
import pandas as pd
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify, flash, redirect, url_for
from plotly.graph_objs import Bar, Sunburst, Scatter
from sqlalchemy import create_engine
from static.models.VQA_model import build_VQA_model, predict_VQA_by_image_features
import en_core_web_lg

# ensure tensorflow proper gpu / memory management
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app = Flask(__name__)
app.secret_key = 'random string'

# define filepaths
image_faetures_file_name = 'static/models/im_features_val.json'
VQA_weights_file_name    = 'static/models/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name  = 'static/models/label_encoder.csv'

# load question & annotations data 
engine = create_engine('sqlite:///../data/VQA_EDA.db')
df_question_type        = pd.read_sql_table('tbl_question_type', engine)
df_annotations_stats    = pd.read_sql_table('tbl_annotations_stats', engine)
df_question_stats       = pd.read_sql_table('tbl_question_stats', engine)
df_question_coverage    = pd.read_sql_table('tbl_question_coverage', engine)
df_qa_samples           = pd.read_sql_table('tbl_qa_samples', engine)

question_type_list = df_qa_samples['question_type'].unique().tolist()
question_type_list.sort()

# load image features 
im_features = json.load(open(image_faetures_file_name, 'r'))
df_im_features = pd.DataFrame.from_dict(im_features)

# build VQA model incl. label encoder
vqa_model = build_VQA_model()
vqa_model.load_weights(VQA_weights_file_name)
word_embeddings = en_core_web_lg.load()
label_encoder = pd.read_csv(label_encoder_file_name)


# index webpage ilustrates VQA approach
@app.route('/')
@app.route('/index')
def index():
    
    return render_template('index.html')


@app.route('/demo', methods = ['POST', 'GET'])
def demo():
    images = os.listdir('static/images/val')

    answers = []
    current_image = images[0]
    question = ""

    if request.method == "POST":
        # user has selected image and posted question
        question        = request.form["question"]
        current_image   = request.form['current-image-src']
        current_image   = current_image.replace('http://localhost:3001/','')
        current_image   = current_image.replace('static/images/val/','')
        current_image   = current_image.replace('/','')

        if question != "":
            # predict answer
            image_features = np.asarray(df_im_features[df_im_features['image']==current_image].iloc[0].features)
            answers = predict_VQA_by_image_features(image_features, question, vqa_model, label_encoder, word_embeddings)

    return render_template('demo.html', images=images, current_image=current_image, question=question, answers=answers)


@app.route('/EDA', methods = ['POST', 'GET'])
def EDA():
    
    # retrieve question type selected from dropdown 
    if request.method == "POST":
        question_type = request.form["comp_select"]
    else:
        question_type = question_type_list[0]

    # extract data needed for website visualizations
    qa_set = df_qa_samples[df_qa_samples['question_type'] ==question_type].to_dict(orient='records')

    df_most_frequent_answers = df_annotations_stats[df_annotations_stats['question_type']==question_type]

    x_data_abs = df_annotations_stats[df_annotations_stats['question_type']==question_type]['count_abs']
    y_data = df_annotations_stats[df_annotations_stats['question_type']==question_type]['answer']
    labels = ['{:.1%}'.format(x) for x in df_annotations_stats[df_annotations_stats['question_type']==question_type]['count_rel']]

    website_data=dict(question_type_list=question_type_list, choice=question_type, qa_set=qa_set,
                nr_datasets= f"{df_question_type[df_question_type['question_length']==1]['count'].sum():,}")

    # create charts
    graphs = [
        {
            'data': [
                Sunburst(
                    ids=df_question_type['question_type'],
                    labels=df_question_type['label'],
                    parents=df_question_type['parent'],
                    values=df_question_type['count'],
                    maxdepth=5,
                    insidetextorientation='radial',
                    branchvalues='total'
                )
            ],
            'layout': {
                'title': 'Distribution of questions <br> by Question Type',
                'margin': dict(t=50, l=0, r=0, b=10),
            }
        }, 
        {
            'data': [
                Bar(
                   x=df_question_stats['q_length'],
                   y=df_question_stats['q_count'],           
                )
            ],
            'layout': {
                'title': 'Histogram on length of question',
                'yaxis': {
                    'title': "Count",
                    # 'autorange': "reversed"
                },
                'xaxis': {
                    'title': "Length of question in words"
                },
                # 'textposition':'auto',
                # 'textfont': {'color': '#ffffff'}
            }
        },
        {
            'data': [
                Scatter(
                    x=df_question_coverage['num'],
                    y=df_question_coverage['count_rel'],           
                )
            ],
            'layout': {
                'title': 'Coverage of questions<br>by number of answers*',
                'yaxis': {
                    'title': "Coverage",
                    'tickformat':"%",
                    'range':[0, 1],
                },
                'xaxis': {
                    'title': "Number of answers",
                    'linecolor':'white'
                },
            }
        },
        {
            'data': [
                Bar(
                    y=df_most_frequent_answers['count_abs'],
                    x=df_most_frequent_answers['answer'],
                    text = ['{:.1%}'.format(x) for x in df_most_frequent_answers['count_rel']]                
                )
            ],
            'layout': {
                'title': 'Most frequent answers',
                'yaxis': {
                    'title': "Number of occurences",
                    # 'autorange': "reversed"
                },
                'xaxis': {
                    'title': "Answers"
                },
                'textposition':'auto',
                'textfont': {'color':'#ffffff'}
            }
        } 
    
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs and additional data
    return render_template('EDA.html', ids=ids, graphJSON=graphJSON,  website_data=website_data)
    

@app.route('/performance')
def performance():

    return render_template('performance.html')


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()