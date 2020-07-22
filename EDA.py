import json
import pandas as pd
import shutil, os
from itertools import chain 
from ElementCountVectorizer import ElementCountVectorizer
from tqdm import tqdm
from sqlalchemy import create_engine


def read_pathDict():
	'''
	build the file path to the training data
	INPUT
	None
	OUTPUT
	annFile, quesFile, imgDir   File path to question file, annotation file and the image directory

	'''
	path_dict 		= json.load(open('path_dict.json', 'r'))

	dataDir 		= path_dict['dataDir']
	annotationDir 	= path_dict['annotationDir']
	questionDir 	= path_dict["questionDir"]
	imageDir    	= path_dict["imageDir"]
	versionType 	= path_dict['versionType']
	taskType    	= path_dict['taskType']
	dataType    	= path_dict['dataType']
	dataSubType 	= path_dict['dataSubType']
	annFile     	= '%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType[0])
	quesFile    	= '%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType[0])
	imgDir      	= '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType[0])

	return annFile, quesFile, imgDir


def load_questions(quesFile):
	'''
	loads the question file and builds the corresponding DataFrame
	INPUT
	quesfile       File path to question file
	OUTPUT
	df_questions   Pandas Dataframe containing Questions incl. additional information

	'''

	ques_open = json.load(open(quesFile, 'r'))
	questions_open = ques_open['questions']
	df_questions = pd.DataFrame.from_dict(questions_open)

	return df_questions


def extract_answers(list):
    '''
    Extracts the 10 answers included in the answers list in a list
    INPUT:
    list:        answers list out of annotations dictionary
    OUTPUT:
    answer_list: list of extracted 10 answers
    '''
    answer_list = []
    for i in range(len(list)):
        answer_list.append(list[i]['answer'])
    return answer_list


def load_annotations(annFile, df_questions):
	'''
	loads the annotation file and builds the corresponding DataFrame
	INPUT
	quesfile       File path to annotation file
	OUTPUT
	df_questions   Pandas Dataframe containing Annotations incl. additional information

	'''

	anno = json.load(open(annFile, 'r'))
	annotations = anno['annotations']
	df_annotations = pd.DataFrame.from_dict(annotations)

	# generate new column in dataframe df_annotations containing 10 extracted answers
	df_annotations['answers_extracted'] = df_annotations['answers'].apply(lambda x: extract_answers(x))
	df_annotations = df_annotations.merge(df_questions[['question_id', 'question']], on='question_id')
	df_annotations['q_length'] = df_annotations['question'].apply(lambda x: len(x.split(' ')))

	return df_annotations


def extract_qa_samples(df, imgDir, top=6):
    '''
    extracts top datasets per question type out dataframe df as samples for illustration
    tries to copy images contained in qa samples to corresponding web app folder
    INPUT
    df               pandas dataframe containing question_type, question, image_id, answers_extracted
    OUTPUT
    df_qa_samples    pandas dataframe containing question, image-filename and answers for top samples
    '''
    
    df_qa_samples = df.groupby('question_type').head(top).reset_index(drop=True)
    df_qa_samples['image_filename'] = df_qa_samples['image_id'].apply(lambda x: 'COCO_train2014_'+str(x).zfill(12)+'.jpg')
    df_qa_samples['answers'] = df_qa_samples['answers_extracted'].apply(lambda x: ', '.join(x))

    # copy images

    dest_folder = 'app/static/images/train' 

    for f in df_qa_samples['image_filename']:
    	try:
    		shutil.copy(imgDir+f, dest_folder)
    	except:
    		pass
    
    return df_qa_samples[['question_type', 'question', 'answers', 'image_filename']]


def Question_type_sunburst_dataframe(df_annotations):
    '''
    Prepares all data to illustrate the distribution of questions by question type in a sunburst diagram
    INPUT
    df_annotations     pandas dataframe containing VQA annotations
    OUTPUT
    df_question_type   pandas dataframe containing sunburst diagram data
    '''
    
    # group df_annotations by question types and count questions by question types
    df_question_type = df_annotations.groupby('question_type').count().reset_index()[['question_type', 'answers']]
    df_question_type.columns = ['question_type', 'count']
    df_question_type['question_length'] = df_question_type['question_type'].apply(lambda x: len(x.split(' ')))
    df_question_type.sort_values(by=['question_length', 'count'], inplace=True, ascending=[True, False])
    
    # to create hierarchical structure we add a column to the dataframe that refernces to the 
    # 'parent' element, i.e. the element one level above in the hierarchy
    # level is hereby equal to the number of words in the question string
    df_question_type['parent'] = df_question_type.apply(lambda x: ' ' if x['question_length'] == 1 \
                                                        else x['question_type'].rsplit(' ', 1)[0], axis=1)
    
    # as the procedure above does not ensure the consistency of the hierarchy some corrections are required
    # We need to add additional rows to the dataframe in case parent references to a non-exisiting element    
    for i in range(df_question_type['question_length'].max(),1,-1):
        df_subset = df_question_type[df_question_type['question_length']==i]
        for j in range(df_subset.shape[0]):
            if df_subset['parent'].iloc[j] not in list(df_question_type['question_type']):
                if i > 2:
                    # haven't reached the hierarchy root
                    parent = df_subset['parent'].iloc[j].rsplit(' ', 1)[0]
                else:
                    parent = ' '
                df_question_type = df_question_type.append({'question_type': df_subset['parent'].iloc[j], 'count':0,                     
                                                            'question_length': i-1, 
                                                            'parent':parent}, ignore_index=True)
                
    # we need to cummulate the figures throughout the hierarchy
    for i in range(df_question_type['question_length'].max(),1,-1):
        df_subset = df_question_type[df_question_type['question_length']==i]
        for j in range(df_subset.shape[0]):
            index = df_question_type[df_question_type['question_type'] == 
                                     df_subset['parent'].iloc[j]].index[0]
            df_question_type['count'].iloc[index] += df_subset['count'].iloc[j]
            
    # Finally we sort the dataframe and add an additional column for the labels in the sunburst diagram 
    df_question_type.sort_values(by=['question_length', 'count'], inplace=True, ascending=[True, False])
    df_question_type['label'] = df_question_type.apply(lambda x: x['question_type'] if x['question_length']==1 else 
                                                       x['question_type'].rsplit(' ', 1)[1], axis=1 )
    
    return df_question_type


def most_frequent_answers(df_element_cv, top=10):
    '''
    creates dataframe containing most frequent answers by question_type
    INPUT
    df_element_cv      panadas Dataframe containing element vectorizer
    top                number of top answers to retrieve
    OUTPUT
    df_annotations_stats  pandas dataframe containing top answre both with absolute and relative count
    
    '''
    df_element_cv_perc = df_element_cv.div(df_element_cv.sum(axis=1), axis=0)
    rows_list = []

    for row in df_element_cv.index:
        df_abs = df_element_cv[df_element_cv.index == row].sort_values(by=row, axis=1, ascending=False)
        df_rel = df_element_cv_perc[df_element_cv_perc.index == row].sort_values(by=row, axis=1, ascending=False)
            
        for col in range(top):
            dict1 = {
                'question_type': row,
                'answer':  df_abs.columns[col],
                'count_abs': df_abs.iloc[0, col],
                'count_rel':  df_rel.iloc[0, col]}
            rows_list.append(dict1)
    return pd.DataFrame(rows_list)


def Annotations_Stats(df_annotations):
	'''
	Create a DataFrame that contains the quantity for every annotation by question type
	INPUT
	df_annotations      pandas DataFrame containing annotations for every question
	OUTPUT
	df_element          pandas Dataframe containing annotation count sorted by relevance
	'''
	df_anno_pivot = df_annotations.groupby(['question_type'])['answers_extracted'].agg(list).reset_index()
	df_anno_pivot['answers_extracted'] = df_anno_pivot['answers_extracted'].apply(lambda x: list(chain.from_iterable(x)))
	cv = ElementCountVectorizer(pass_stop=False)
	cv.fit(df_anno_pivot['answers_extracted'])
	element_cv = cv.transform(df_anno_pivot['answers_extracted']).toarray()
	df_element_cv = pd.DataFrame(element_cv, columns=cv.elements_)
	df_element_cv.set_index(df_anno_pivot['question_type'], inplace=True)

	# sort answers by number of occurences
	df_element_cv.loc['Totals'] = df_element_cv.sum()
	df_element_cv.sort_values(by='Totals', axis=1, ascending=False, inplace=True)
	df_element_cv.drop('Totals', inplace=True)

	return df_element_cv 


def Question_Coverage(df_annotations, annotations_sorted, num_annotations=5000):
	'''
	Calculates a series that contains the share of questions that can be correctly answered with 
	the first n annotations when ordered by relevance
	INPUT
	df_annotations        pandas DataFrame containing annotations for every question
	annotations_sorted    array of annotations sorted by relevance, i.e. annotation with highest frequency appears at first position
	num_annotations       limit to which the Question_Coverage shall be claculated (step size = 1)
	OUTPUT
	df_question_coverage  pandas DataFrame containing absolute and relevative count for question coverage
	'''

	# we build up a list first as appending dicts to a list is more
	# performant than adding rows to a Dataframe
	result_list = []
	cnt = 0

	df_anno_cmpt = df_annotations.copy(deep=True)

	for index, col in tqdm(enumerate(annotations_sorted[:num_annotations]), total=num_annotations):

		# identify all questions covered by annotation
		df_anno_cmpt['contains'] = df_anno_cmpt['answers_extracted'].apply(lambda x: len(set(x)&set([col]))>0)
		cnt += df_anno_cmpt['contains'].sum()

		# drop all questions that are already covered
		# that should speed up the algorithm the longer it runs
		df_anno_cmpt.drop(df_anno_cmpt[df_anno_cmpt['contains']].index, inplace=True)

		dict1 = {'num': index+1, 'count': cnt }
		result_list.append(dict1)

	df_question_coverage = pd.DataFrame(result_list)
	df_question_coverage['count_rel'] = df_question_coverage['count']/df_annotations.shape[0]

	return df_question_coverage


def save_data(df_list, tablename_list, database_filename):
    """
    Stores the dataframes to an SQL database
    INPUT
    df_list				list of pandas dataframes to be saved
    tablename_list		list of table names to be used 
    database_filename   filepath where to store datanase file      
    OUTPUT
    None
    """
    try:
        engine = create_engine('sqlite:///{}'.format(database_filename))                     
    except:
        sys.exit("Couldn't open databbase: {}".format(database_filename))    

    for index, df in enumerate(df_list):
    	try:
    		df.to_sql(tablename_list[index], engine, index=False, if_exists='replace')
    	except:
    		sys.exit("Couldn't write table {} to : {}".format(tablename_list[index], database_filename))                       
    


def main():
	annFile, quesFile, imgDir = read_pathDict()

	print('Loading questions from training set...')
	df_questions = load_questions(quesFile)
	print("Number of open ended questions loaded :  ", df_questions.shape[0])

	print('Loading annotations from training set...')
	df_annotations = load_annotations(annFile, df_questions)
	print("Number of annotations loaded :  ", df_annotations.shape[0])

	print('Extracting Question-Answer samples ...')
	df_qa_samples = extract_qa_samples(df_annotations, imgDir)

	print('Analyzing structure of questions by question type ...')
	df_question_type = Question_type_sunburst_dataframe(df_annotations)

	print('Calculating statistics on annotation frequency ...')
	df_element_cv = Annotations_Stats(df_annotations)
	print('Number of unique annotations: ', df_element_cv.shape[1])
	df_annotations_stats = most_frequent_answers(df_element_cv)

	print('Calculating statistics on questions length ...')
	df_question_stats =	pd.DataFrame(df_annotations['q_length'].value_counts().rename_axis('q_length').reset_index(name='q_count'))

	print('Calculating question coverage ...')
	print('This may take a while ...')
	df_question_coverage = Question_Coverage(df_annotations, df_element_cv.columns)

	print('Savings data to SQLite database ...')
	save_data([df_question_type , df_annotations_stats, df_question_stats, df_question_coverage, df_qa_samples], 
			   ['tbl_question_type', 'tbl_annotations_stats', 'tbl_question_stats', 'tbl_question_coverage', 'tbl_qa_samples'], 'data/VQA_EDA.db')

if __name__ == '__main__':
    main()