
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import random
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import json
import sklearn
import sys
#
# Read training table
# We use the pickle file provided 
# https://www.kaggle.com/rohanrao/tutorial-on-reading-large-datasets/data?select=riiid_train.pkl.gzip
# Thanks to @Vopani

train = pd.read_pickle("riiid_train.pkl.gzip")

#
# Compute time_lag
#
tmp_df = train[['user_id', 'timestamp', 'task_container_id']]
tmp_df = tmp_df.drop_duplicates(['user_id', 'timestamp', 'task_container_id'])
time_lag = np.array([0]+(tmp_df.timestamp[1:].values - tmp_df.timestamp[:-1].values).tolist(), 'float32')
time_lag[time_lag < 0] = 0
tmp_df['time_lag'] = time_lag
del time_lag
train = train.merge(tmp_df, on=['user_id', 'timestamp', 'task_container_id'], how = 'left')
del tmp_df

#
# Compute question_elapsed_time and question_had_explanation 
# then drop prior_question_elapsed_time, prior_question_had_explanation
#
tmp_df = train[['user_id', 'timestamp', 'task_container_id', 
                             'prior_question_elapsed_time', 'prior_question_had_explanation']][train.content_type_id==False]

tmp_df = tmp_df.drop_duplicates(['user_id', 'timestamp', 'task_container_id'])

tmp_df['question_elapsed_time'] = tmp_df.prior_question_elapsed_time[1:].tolist()+[pd.NA]
tmp_df['question_had_explanation'] = tmp_df.prior_question_had_explanation[1:].tolist()+[pd.NA]
tmp_df = tmp_df[(tmp_df.user_id[1:].values==tmp_df.user_id[:-1].values).tolist()+[False]]
tmp_df = tmp_df[['user_id', 'timestamp', 'task_container_id','question_elapsed_time','question_had_explanation']]
train = train.merge(tmp_df, on=['user_id', 'timestamp', 'task_container_id'], how = 'left')
train = train.drop(columns = ['prior_question_elapsed_time', 'prior_question_had_explanation'])
del tmp_df

#
# Compute stats on questions: question difficulty & question popularity
#
#
question_stats = train[
                       train['content_type_id'] == False
                       ].groupby(
                           ['content_id', 'content_type_id']
                           ).agg({
                               'answered_correctly':'sum',
                               'user_id':'count'
                           }).reset_index()
#Rescaling                          
question_stats['difficulty'] = np.sqrt(1.0 - question_stats['answered_correctly']/question_stats['user_id'])
question_stats['popularity'] = np.power(question_stats['user_id']/question_stats['user_id'].max(), 0.25)
question_stats = question_stats[['content_id', 'content_type_id', 'difficulty', 'popularity']]

#
# Generate ids for question & lecture metadata for embeddings
#
questions = pd.read_csv('questions.csv')
lectures = pd.read_csv('lectures.csv')

encoded_questions = pd.DataFrame(data=questions[['question_id']].values, columns=['content_id'])
encoded_questions['content_type_id'] = False
encoded_questions['encoded_question_id'] = LabelEncoder().fit_transform(questions.question_id)
encoded_questions['bundle_id'] = LabelEncoder().fit_transform(questions.bundle_id)
encoded_questions['correct_answer'] = questions.correct_answer
encoded_questions['part'] = questions.part
tags = questions.tags.fillna('').apply(lambda x: [1+int(t) for t in str(x).split()])
encoded_questions['tags'] = tf.keras.preprocessing.sequence.pad_sequences(tags).tolist()

encoded_questions  = encoded_questions.merge(question_stats, 
                        how = 'left', 
                        on=['content_id', 'content_type_id']).fillna(0)

encoded_lectures = pd.DataFrame(data=lectures[['lecture_id']].values, columns=['content_id'])
encoded_lectures['content_type_id'] = True

encoded_lectures['encoded_lecture_id'] = LabelEncoder().fit_transform(lectures.lecture_id)
encoded_lectures['part'] = lectures.part
encoded_lectures['tag'] = LabelEncoder().fit_transform(lectures.tag)
encoded_lectures['type_of'] = LabelEncoder().fit_transform(lectures.type_of)

# Append questions and lectures table, create new index to have
# a common id (encoded_content_id)
encoded_content = pd.DataFrame.from_dict({
    'content_id':questions.question_id,
    'content_type_id':len(questions)*[False]
  }).append(
      pd.DataFrame.from_dict({
      'content_id':lectures.lecture_id,
      'content_type_id':len(lectures)*[True]
    })
)
encoded_content['encoded_content_id'] = range(len(encoded_content))

#
# Put the generated encoded_content_id in the train table, 
# 

train = train.merge(encoded_content, on=['content_id', 'content_type_id'], how = 'left')
train = train.drop(columns = ['content_id', 'content_type_id', 'task_container_id'])
train = train.astype({'question_had_explanation': 'boolean', 'encoded_content_id':'int32', 'time_lag':'float32'})
train.to_pickle('whole-train.pickle')

#
# Saving question & lecture meta data map (with generated ids)
#
lectures = encoded_lectures.to_dict(orient='list')
questions = encoded_questions.to_dict(orient='list')

encoded_content_map_v2 = {
    'encoded_question_id': questions['encoded_question_id'],
    'question_bundle_id': questions['bundle_id'],
    'question_tags': questions['tags'],
    'question_part': questions['part'],
    'question_difficulty': questions['difficulty'],
    'question_popularity': questions['popularity'],
    'encoded_lecture_id': lectures['encoded_lecture_id'],
    'lecture_part': lectures['part'],
    'lecture_tag': lectures['tag'],
    'lecture_type_of': lectures['type_of']
}
with open('encoded_content_map_v2.json', 'w') as f:
            json.dump(encoded_content_map_v2, f)

encoded_content = encoded_content.to_dict(orient='list')
encoded_content_id_map = {
    'content_id': encoded_content['content_id'],
    'content_type_id': encoded_content['content_type_id'],
    'encoded_content_id': encoded_content['encoded_content_id']
}
with open('encoded_content_id_map.json', 'w') as f:
            json.dump(encoded_content_id_map, f)