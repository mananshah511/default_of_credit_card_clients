import os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.constant import *
from credit_card.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformConfig,ModelTrainerConfig,ModelEvulationConfig,ModelPusherConfig
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
        
    def get_data_validation_config(self)->DataValidationConfig:
        try:
            logging.info(f"get data validation config function started")
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_validation_config = self.config_info[DATA_VALIDTION_CONFIG_KEY]

            data_validation_artifact_dir = os.path.join(artifact_dir,DATA_VALIDATION_DIR,self.time_stamp)

            schema_dir = os.path.join(ROOT_DIR,data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY])

            schema_file_dir = os.path.join(schema_dir,data_validation_config[DATA_VALIDATION_SCHEMA_FILE_KEY])

            

            data_validation_config = DataValidationConfig(schema_file_dir=schema_file_dir,
                                                          report_page_file_dir=data_validation_artifact_dir,
                                                          report_name=data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME]
                                                          )
            logging.info(f"data validation config : {data_validation_config}")
            return data_validation_config

        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_data_transform_config(self)->DataTransformConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_transform_config = self.config_info[DATA_TRANSFORM_CONFIG_KEY]

            data_transform_config_dir = os.path.join(artifact_dir,DATA_TRANSFORM_DIR,self.time_stamp)

            data_transform_graph_dir = os.path.join(data_transform_config_dir,data_transform_config[DATA_TRANSFORM_GRAPH_DIR_KEY])

            data_transform_tain_dir = os.path.join(data_transform_config_dir,data_transform_config[DATA_TRANSFORM_TRAIN_DIR_KEY])

            data_transform_test_dir = os.path.join(data_transform_config_dir,data_transform_config[DATA_TRANSFORM_TEST_DIR_KEY])

            data_transform_preprocessed_dir = os.path.join(data_transform_config_dir,data_transform_config[DATA_TRANSFORM_PREPROCESSED_OBJECT_DIR_KEY],
                                                           data_transform_config[DATA_TRANSFORM_PREPROCESSED_OBJECT_FILE_NAME_KEY])
            
            data_transform_cluster_dir = os.path.join(data_transform_config_dir,data_transform_config[DATA_TRANSFORM_CLUSTER_MODEL_DIR_KEY],
                                                      data_transform_config[DATA_TRANSFORM_CLUSTER_MODEL_NAME_KEY])
            
            data_transform_config =  DataTransformConfig(transform_train_dir=data_transform_tain_dir,
                                                         transform_test_dir=data_transform_test_dir,
                                                         graph_save_dir=data_transform_graph_dir,
                                                         preprocessed_file_path=data_transform_preprocessed_dir,
                                                         cluster_model_file_path=data_transform_cluster_dir)
            logging.info(f"data transform config : {data_transform_config}")

            return data_transform_config
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_model_trainer_config(self)->ModelTrainerConfig:
        try:
            logging.info(f"get model trainer config function started")
            artifact_dir = self.training_pipeline_config.artifact_dir

            model_trainer_config = self.config_info[MODEL_TRAINER_CONFIG_KEY]

            model_trainer_artifact_dir = os.path.join(artifact_dir,MODEL_TRAINER_DIR,self.time_stamp)

            base_accuracy = model_trainer_config[MODEL_TRAINER_BASE_ACCURACY_KEY]

            model_config_file_path = os.path.join(ROOT_DIR,model_trainer_config[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],
                                                  model_trainer_config[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY])
            
            model_trained_model_path = os.path.join(model_trainer_artifact_dir,model_trainer_config[MODEL_TRAINER_MODEL_FILE_NAME_KEY])

            model_trainer_config = ModelTrainerConfig(base_accuracy=base_accuracy,
                                                      trained_model_file_path=model_trained_model_path,
                                                      model_config_file_path=model_config_file_path)
            logging.info(f"model trainer config : {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_model_evulation_config(self)->ModelEvulationConfig:
        try:
            logging.info(f"get model evulation function started")

            artifact_dir = self.training_pipeline_config.artifact_dir

            model_evulation_config = self.config_info[MODEL_EVULATION_CONFIG_KEY]

            model_evulation_file_path = os.path.join(artifact_dir,MODEL_EVULATION_DIR,model_evulation_config[MODEL_EVULATION_FILE_NAME_KEY])

            model_evulation_config = ModelEvulationConfig(evulation_file_path=model_evulation_file_path,
                                                          time_stamp=self.time_stamp)
            logging.info(f"model evulation config : {model_evulation_config}")
            return model_evulation_config
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_model_pusher_config(self)->ModelPusherConfig:
        try:
            logging.info(f"get model pusher config function started")
            artifact_dir = self.training_pipeline_config.artifact_dir

            model_pusher_config = self.config_info[MODEL_PUSHER_CONFIG_KEY]

            export_dir_path = os.path.join(artifact_dir,MODEL_PUSHER_DIR,model_pusher_config[MODEL_PUSHER_EXPORT_MODEL_DIR_KEY])

            model_pusher_config = ModelPusherConfig(export_dir_path=export_dir_path)

            logging.info(f"model pusher config : {model_pusher_config}")

            return model_pusher_config
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            logging.info(f"get training pipeline config function started")
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]

            artifact_dir = os.path.join(ROOT_DIR,training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)

            logging.info(f"training pipeline config : {training_pipeline_config}")
            
            return training_pipeline_config
        except Exception as e:
            raise CreditCardException(e,sys) from e