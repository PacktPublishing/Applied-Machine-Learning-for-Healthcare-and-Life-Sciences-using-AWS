import csv
import wget
import zipfile
import os
import pandas as pd
import boto3
import time
import json
import argparse
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--bucket', type=str)
parser.add_argument('--region',type=str)
args = parser.parse_args()

bucket=args.bucket
cm = boto3.client('comprehendmedical',region_name=args.region)
s3_client = boto3.client('s3',region_name=args.region)

if os.path.exists('data')==False:
    os.mkdir('data')

file_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip'
dest_file = 'data/drugsCom_raw.zip'

print("Downloading source files...")

wget.download(file_url, dest_file)

with zipfile.ZipFile('data/drugsCom_raw.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

os.remove('data/drugsCom_raw.zip')

orig_list = list()
for filename in os.listdir('data'):
    with open('data/'+filename) as csvfile:
        myreader = csv.reader(csvfile, delimiter='\t')
        for row in myreader:
            if row[0] == '':
                continue
            else:
                orig_list.append({
                    'id': row[0],
                    'drugName': row[1],
                    'condition': row[2],
                    'review': row[3]
                })

    
if os.path.exists('processed_data')==False:
    os.mkdir('processed_data')
    
raw_df=pd.DataFrame.from_records(orig_list)
raw_df.to_csv('processed_data/raw_df.csv', index=False)

print("\nRaw data processed from input files")
print("\nRamdomly sampling 100 rows for topic extraction")

df_sample=raw_df.sample(n=100)
sample_list = list()


for index,row in df_sample.iterrows():
    entities = cm.detect_entities(Text=row['review'])
    topic_list = []
    for entity in entities['Entities']:
        if entity['Category'] == 'MEDICAL_CONDITION':
            topic_list.append(entity['Text'])

    sample_list.append({
            'id': row['id'],
            'drugName': row['drugName'],
            'condition': row['condition'],
            'review': row['review'],
            'topics': topic_list[:5]
        })
        
sample_df=pd.DataFrame.from_records(sample_list)

sample_df.to_csv('processed_data/sample_df.csv', index=False) 


sampled_topics=pd.read_csv('processed_data/sample_df.csv')['topics'].tolist()
print(sampled_topics)
vectorizer = TfidfVectorizer()
vecs = vectorizer.fit_transform(sampled_topics)
normalizer = Normalizer(copy=False)
normalized_data = normalizer.fit_transform(vecs).toarray()
normalized_data.shape
np.savetxt("processed_data/prediction_data.csv", normalized_data, delimiter=",")




s3_client.upload_file('processed_data/sample_df.csv', bucket, 'chapter9/data/sample_df.csv')
s3_client.upload_file('processed_data/raw_df.csv', bucket, 'chapter9/data/raw_df.csv')
s3_client.upload_file('processed_data/prediction_data.csv', bucket, 'chapter9/data/prediction_data.csv')


print("\nprocessed files uploaded to s3")
