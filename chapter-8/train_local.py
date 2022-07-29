# Portions of this script is borrowed from https://github.com/mims-harvard/TDC/blob/main/tutorials/TDC_104_ML_Model_DeepPurpose.ipynb

from DeepPurpose import utils, CompoundPred
from tdc.single_pred import ADME
import warnings
import argparse
import os
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--sagemaker_program', type=str, default=os.environ.get('SAGEMAKER_PROGRAM'))
    args = parser.parse_args()

    X, y = ADME(name = 'HIA_Hou').get_data(format = 'DeepPurpose')
    drug_encoding = 'MPNN'
    train, val, test = utils.data_process(X_drug = X, 
                                      y = y, 
                                      drug_encoding = drug_encoding,
                                      random_seed = 'TDC')
    config = utils.generate_config(drug_encoding = drug_encoding, 
                         train_epoch = 1, 
                         LR = 0.001, 
                         batch_size = 128,
                         mpnn_hidden_size = 32,
                         mpnn_depth = 2
                        )
    model = CompoundPred.model_initialize(**config)
    model.train(train, val, test)
    
    model.save_model(args.model_dir)
