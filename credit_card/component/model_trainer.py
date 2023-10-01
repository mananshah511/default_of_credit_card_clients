import os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.entity.artifact_entity import ModelArtifactConfig,DataTransformArtifact
from credit_card.entity.config_entity import ModelTrainerConfig
from credit_card.entity.model_trainer import ModelFactory,get_evulated_classification_model,GridSearchedBestModel,MetricInfoArtifact
import pandas as pd
import numpy as np
import dill


class ModelTrainer:

    def __init__(self, model_trainer_config:ModelTrainerConfig,
                 data_transform_artifact:DataTransformArtifact):
        try:
            logging.info(f"{'>>'*20}Model Trainer log started.{'<<'*20} ")
            self.model_trainer_config = model_trainer_config
            self.data_transform_aartifact = data_transform_artifact
        except Exception as e:
            raise CreditCardException(e,sys) from e
    
    def intitate_model_trainer(self)->ModelArtifactConfig:
        try:
            logging.info(f"intitate model trainer function started")

            transform_train_file = self.data_transform_aartifact.transform_train_dir
            transform_test_file = self.data_transform_aartifact.transform_test_dir

            logging.info(f"transform train files are at : {transform_train_file}")
            logging.info(f"transform test files are at : {transform_test_file}")

            train_files = list(os.listdir(transform_train_file))
            test_files = list(os.listdir(transform_test_file))

            logging.info(f"training files are : {train_files}")
            logging.info(f"testing files are : {test_files}")

            model_trainer_artifact = None

            train_accuracy = []
            test_accuracy = []
            model_accuracy = []

            for cluster_number in range(len(train_files)):
                logging.info(f"{'>>'*20}cluster : {cluster_number}{'<<'*20}")

                train_file_name = train_files[cluster_number]
                test_file_name = test_files[cluster_number]

                train_file_path = os.path.join(transform_train_file,train_file_name)
                test_file_path = os.path.join(transform_test_file,test_file_name)

                logging.info(f"reading train data from the file : {train_file_path}")    
                train_df = pd.read_csv(train_file_path)
                logging.info(f"train data reading successfull")
                logging.info(f"reading test data from the file : {test_file_path}")    
                test_df = pd.read_csv(test_file_path)
                logging.info(f"test data reading successfull")

                logging.info("splitting data into input and output feature")
                X_train,y_train,X_test,y_test = train_df.iloc[:,:-1],train_df.iloc[:,-1],test_df.iloc[:,:-1],test_df.iloc[:,-1]

                logging.info("exctracting model cofig file path")
                model_config_file_path = self.model_trainer_config.model_config_file_path
                logging.info("intilization of model factory class")
                model_factory = ModelFactory(model_config_file_path)

                base_accuracy = self.model_trainer_config.base_accuracy
                logging.info(f"base accuracy is : {base_accuracy}")
            
                logging.info(f"finding best model for the cluster : {cluster_number}")
                best_model = model_factory.get_best_model(X=np.array(X_train),y=np.array(y_train),base_accuracy=base_accuracy)
                logging.info(f"best model on trained data is : {best_model}")

                grid_searched_best_model_list:list[GridSearchedBestModel] = model_factory.grid_searched_best_model_list
                model_list = [model.best_model for model in grid_searched_best_model_list]
                logging.info(f"individual best model list : {model_list}")

                logging.info(f"finding best model after evulation on train and test data")
                metric_info:MetricInfoArtifact=get_evulated_classification_model(X_train=np.array(X_train),y_train=np.array(y_train),
                                                                        X_test=np.array(X_test),
                                                                            y_test=np.array(y_test),base_accuracy=base_accuracy,model_list=model_list)
                

                
                
                
                model_object = metric_info.model_object
                logging.info(f"----------best model after train and test evulation : {model_object} accuracy : {metric_info.model_accuracy}-----------")
                model_path = self.model_trainer_config.trained_model_file_path
                model_base_name = os.path.basename(model_path)
                logging.info(f"base model name is : {model_base_name}")
                dir_name = os.path.dirname(model_path)
                cluster_dir_name = os.path.join(dir_name,'cluster'+str(cluster_number))
                logging.info(f"model will be saved in {cluster_dir_name}")
                os.makedirs(cluster_dir_name,exist_ok=True)
                model_cluster_path = os.path.join(cluster_dir_name,model_base_name)

                with open(model_cluster_path,'wb') as obj_file:
                    dill.dump(model_object,obj_file)
                logging.info(f"model saved successfully")
                
                train_accuracy.append(metric_info.train_accuracy)
                test_accuracy.append(metric_info.test_accuracy)
                model_accuracy.append(metric_info.model_accuracy)


            model_trainer_artifact = ModelArtifactConfig(is_trained=True,
                                                        message="Model trained successfully",
                                                        trained_model_path=self.model_trainer_config.trained_model_file_path,
                                                        train_accuracy=train_accuracy,
                                                        test_accuracy=test_accuracy,
                                                        model_accuracy=model_accuracy)
            return model_trainer_artifact

        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Model trainer log completed.{'<<'*20} \n\n")