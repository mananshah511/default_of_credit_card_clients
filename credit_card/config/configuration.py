import os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.constant import *
from credit_card.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
from credit_card.util.util import read_yaml

class Configuration:

    def __init__(self,
                 config_file_path:str = CONFIG_FILE_PATH,
                 current_time_stamp:str = CURRENT_TIME_STAMP) -> None:
        try:
            self.config_info = read_yaml(file_path=config_file_path)
            self.time_stamp = current_time_stamp
            self.training_pipeline_config = self.get_training_pipeline_config()
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            logging.info(f"get data ingestion config function started")

            artifact_dir = self.training_pipeline_config.artifact_dir   

            data_ingestion_config = self.config_info[DATA_INGESTION_CONFIG_KEY]

            data_ingestion_artifact_dir = os.path.join(artifact_dir,DATA_INGESTION_DIR,self.time_stamp)

            dataset_download_url = data_ingestion_config[DATA_INGESTION_URL_KEY]

            raw_data_dir = os.path.join(data_ingestion_artifact_dir,data_ingestion_config[DATA_INGESTION_RAW_DATA_DIR_KEY])

            zip_data_dir = os.path.join(data_ingestion_artifact_dir,data_ingestion_config[DATA_INGESTION_ZIP_DATA_DIR_KEY])

            ingested_data_dir = os.path.join(data_ingestion_artifact_dir,data_ingestion_config[DATA_INGESTION_INGESTION_DATA_DIR_KEY])

            ingested_train_dir = os.path.join(ingested_data_dir,data_ingestion_config[DATA_INGESTION_INGESTION_TRAIN_DATA_DIR_KEY])

            ingested_test_dir = os.path.join(ingested_data_dir,data_ingestion_config[DATA_INGESTION_INGESTION_TEST_DATA_DIR_KEY])

            data_ingestion_config = DataIngestionConfig(dataset_download_url=dataset_download_url,
                                                        raw_data_dir=raw_data_dir,
                                                        zip_data_dir=zip_data_dir,
                                                        ingested_train_dir=ingested_train_dir,
                                                        ingested_test_dir=ingested_test_dir,)
            logging.info(f"data ingestion config : {data_ingestion_config}")

            return data_ingestion_config
        
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            logging.info("get training pipeline config function started")
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]

            artifact_dir = os.path.join(ROOT_DIR,training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)

            logging.info(f"training pipeline config : {training_pipeline_config}")
            
            return training_pipeline_config
        except Exception as e:
            raise CreditCardException(e,sys) from e