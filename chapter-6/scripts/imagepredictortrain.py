import argparse
import os
import autogluon.core as ag
from autogluon.vision import ImageDataset, ImagePredictor
import zipfile
import boto3


def get_env_if_present(name):
    result = None
    if name in os.environ:
        result = os.environ[name]
    return result


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        print(f"WARN: more than one file is found in {channel} directory")
    print(f"Using {file}")
    filename = f"{path}/{file}"
    return filename
    
if __name__ == "__main__":
# Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    # ------------------------------------------------------------ Args parsing
    print("Starting AG")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=get_env_if_present("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model-dir", type=str, default=get_env_if_present("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=str, default=get_env_if_present("SM_NUM_GPUS"))
    parser.add_argument("--training_dir", type=str, default=get_env_if_present("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--test_dir", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TEST")
    )
    parser.add_argument("--ag_config", type=str, default=get_env_if_present("SM_CHANNEL_CONFIG"))

    args, _ = parser.parse_known_args()

    print(f"Args: {args}")
    
     # ---------------------------------------------------------------- Training

    train_file = get_input_path(args.training_dir)
    
    with zipfile.ZipFile(train_file, 'r') as zip_ref:
        zip_ref.extractall('data') 
    train_data,val_data, _= ImageDataset.from_folders('data', train='train', val='val')
    print('train #', len(train_data), 'val #', len(val_data))
    predictor = ImagePredictor()
    predictor.fit(train_data,val_data)
    predictor.save(os.path.join(args.model_dir, "model.ag"))
