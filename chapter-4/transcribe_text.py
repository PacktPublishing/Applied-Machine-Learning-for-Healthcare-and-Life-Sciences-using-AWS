from __future__ import print_function
import time
import boto3
import json
import pandas as pd

transcribe = boto3.client('transcribe')

job_name = "med-transcription-job"
job_uri = "" #enter the S3 URI of the audio file between the double quotes

try:
    transcribe.delete_medical_transcription_job(MedicalTranscriptionJobName=job_name)
    print('creating new transcript job med-transcription-job')
except:
     print('creating new transcript job med-transcription-job')


transcribe.start_medical_transcription_job(
     MedicalTranscriptionJobName = job_name,
     Media = {'MediaFileUri': job_uri},
     LanguageCode = 'en-US',
     Specialty = 'PRIMARYCARE',
     Type = 'DICTATION',
     OutputBucketName = 'output bucket name', #enter the output buket name here
     OutputKey='output/'
  )

while True:
    status = transcribe.get_medical_transcription_job(MedicalTranscriptionJobName=job_name)
    if status['MedicalTranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    print("Not ready yet...")
    time.sleep(5)
print('transcription complete. Transcription Status: ',status['MedicalTranscriptionJob']['TranscriptionJobStatus'])
s3 = boto3.client('s3')
s3.download_file('output bucket name', 'output/medical/med-transcription-job.json', 'transcript.json') #enter the output bucket name here.

json_file = "transcript.json"

with open(json_file, 'r') as j:
     transcript_dict = json.loads(j.read())
        
print('****Transcription Output***')
print(transcript_dict['results']['transcripts'][0]['transcript'])

cm = boto3.client('comprehendmedical')

entities=cm.detect_entities_v2(Text=transcript_dict['results']['transcripts'][0]['transcript'])

data = []

for i in entities['Entities']:   
    data.append([i['Id'],i['Text'],i['Category'],i['Type']])
    
df = pd.DataFrame(data, columns=['Id', 'Text', 'Category','Type'])
df.to_csv('entities.csv', index=False)
print('Transcription analysis complete. Entities saved in entities.csv')



                        