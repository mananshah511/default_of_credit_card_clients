import os
from datetime import datetime


ROOT_DIR = os.getcwd()
CONFIF_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIF_DIR,CONFIG_FILE_NAME)

CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


#training pipeline related variables

TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"

#data ingestion related variables

DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_URL_KEY = "dataset_download_url"
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_ZIP_DATA_DIR_KEY = "zip_data_dir"
DATA_INGESTION_INGESTION_DATA_DIR_KEY = "ingested_data_dir"
DATA_INGESTION_INGESTION_TRAIN_DATA_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_INGESTION_TEST_DATA_DIR_KEY = "ingested_test_dir"


