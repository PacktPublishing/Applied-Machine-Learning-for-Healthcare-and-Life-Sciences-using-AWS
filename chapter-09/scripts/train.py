
import argparse
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO

def model_fn(model_dir):
    
    kmeans = joblib.load(os.path.join(model_dir, "kmeansmodel.joblib"))
    return kmeans

def input_fn(input_data, content_type):
    
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--n_clusters', type=int, default=2)
    parser.add_argument('--random_state', type=int, default=0)
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args = parser.parse_args()
    
    input_files = [ os.path.join(args.training, file) for file in os.listdir(args.training) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.training, "train"))
    
    raw_data = [ pd.read_csv(file) for file in input_files ]
    train_data = pd.concat(raw_data)
    print(train_data.shape)
    kmeans = KMeans(n_clusters=2,random_state=0).fit(train_data)
    joblib.dump(kmeans, os.path.join(args.model_dir, "kmeansmodel.joblib"))
