# Portions of this script is borrowed from https://github.com/mims-harvard/TDC/blob/main/tutorials/TDC_104_ML_Model_DeepPurpose.ipynb


from tdc.utils import retrieve_dataset_names
from tdc.single_pred import ADME
from DeepPurpose import utils, CompoundPred
import warnings
import argparse
import os
from shutil import make_archive
import boto3
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--sagemaker_program', type=str, default=os.environ.get('SAGEMAKER_PROGRAM'))
    parser.add_argument('--models_output_bucket', type=str)
    args = parser.parse_args()
    
    # Check whether the specified path exists or not
    isExist = os.path.exists('./models')

    if not isExist:

      # Create a new directory because it does not exist 
      os.makedirs('./models')
      print("The new directory is created!")
    
    
    bucket_name=args.models_output_bucket
    adme_datasets = retrieve_dataset_names('ADME')   
    for dataset_name in adme_datasets:
        X, y = ADME(name = dataset_name).get_data(format = 'DeepPurpose')
        drug_encoding = 'Morgan'
        train, val, test = utils.data_process(X_drug = X, 
                                          y = y, 
                                          drug_encoding = drug_encoding,
                                          random_seed = 'TDC')
        config = utils.generate_config(drug_encoding = drug_encoding, 
                             train_epoch = 5, 
                             LR = 0.001, 
                             batch_size = 128,
                             mpnn_hidden_size = 32,
                             mpnn_depth = 2
                            )
        model = CompoundPred.model_initialize(**config)
        model.train(train, val, test)
        model.save_model('./models/' + dataset_name + '_model')
        res = os.listdir('./models/')
        print(res)

    res1 = os.listdir('.')
    print(res1)
    make_archive('./models', 'zip', root_dir='./models')

    s3 = boto3.client("s3")
    s3.upload_file('models.zip', bucket_name, 'ADME/models/models.zip')
