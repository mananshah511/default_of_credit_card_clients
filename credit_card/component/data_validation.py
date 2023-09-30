import os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.entity.config_entity import DataValidationConfig
from credit_card.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import pandas as pd
import numpy as np
from credit_card.util.util import read_yaml
from credit_card.constant import COLUMN_KEY,NUMERIC_COULMN_KEY,CATEGORICAL_COLUMN_KEY,TARGET_COLUMN_KEY
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

class DataValidation:

    def __init__(self,data_vaidation_config : DataValidationConfig,
                 data_ingestion_artifact : DataIngestionArtifact) -> None:
        try:
            logging.info(f"{'>>'*20}Data Validation log started.{'<<'*20} ")
            self.data_validation_config = data_vaidation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_train_test_dataframe(self):
        try:
            logging.info(f"get train test dataframe function started")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"data read successfull")
            return train_df,test_df
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_check_train_test_dir_exist(self)->bool:
        try:
            logging.info(f"get check train test dir exist function started")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_flag = False
            test_flag = False
            if os.path.exists(train_file_path):
                logging.info("train path is availabel")
                train_flag = True

            if os.path.exists(test_file_path):
                logging.info("test path is availabel")
                test_flag = True

            if test_flag==False:
                logging.info(f"test path is not there")
            if train_flag==False:
                logging.info(f"train path is not there")

            return train_flag and test_flag
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_column_count_validation(self)->bool:
        try:
            logging.info(f"get column count validation function started")

            schema_file_path = self.data_validation_config.schema_file_dir
            schema_file = read_yaml(file_path=schema_file_path)

            train_df,test_df = self.get_train_test_dataframe()

            train_count = len(train_df.columns)
            test_count = len(test_df.columns)

            schema_count = len(schema_file[COLUMN_KEY])

            logging.info(f"number of columns in train file is : {train_count}")
            logging.info(f"number of columns in test file is : {test_count}")

            logging.info(f"number of columns in schema file is: {schema_count}")

            train_flag=False
            test_flag=False

            if schema_count==train_count:
                logging.info(f"number of column in train file is ok")
                train_flag=True
            
            if schema_count==test_count:
                logging.info(f"number of column in test file is ok")
                test_flag=True

            if train_flag==False:
                logging.info(f"number of column in train file is not correct")

            if test_flag==False:
                logging.info(f"number of column in test file is not correct")

            return train_flag and test_flag
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_column_name_validation(self)->bool:
        try:
            logging.info(f"get column name validation function started")

            schema_file_path = self.data_validation_config.schema_file_dir
            schema_file = read_yaml(file_path=schema_file_path)

            train_df,test_df = self.get_train_test_dataframe()

            train_column = list(train_df.columns)
            test_column = list(test_df.columns)

            schema_column = list(schema_file[NUMERIC_COULMN_KEY])+list(schema_file[CATEGORICAL_COLUMN_KEY])

            train_column.sort()
            test_column.sort()
            schema_column.sort()

            logging.info(f"number of columns in train file is : {train_column}")
            logging.info(f"number of columns in test file is : {test_column}")

            logging.info(f"number of columns in schema file is: {schema_column}")

            train_flag=False
            test_flag=False

            if schema_column==train_column:
                logging.info(f"name of column in train file is ok")
                train_flag=True
            
            if schema_column==test_column:
                logging.info(f"name of column in test file is ok")
                test_flag=True

            if train_flag==False:
                logging.info(f"name of column in train file is not correct")

            if test_flag==False:
                logging.info(f"name of column in test file is not correct")

            return train_flag and test_flag
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_column_data_type_validation(self)->bool:
        try:
            logging.info(f"get column name validation function started")

            schema_file_path = self.data_validation_config.schema_file_dir
            schema_file = read_yaml(file_path=schema_file_path)

            train_df,test_df = self.get_train_test_dataframe()

            train_column_data = dict(train_df.dtypes)
            test_column_data = dict(test_df.dtypes)

            schema_column_data = schema_file[COLUMN_KEY]

            
            logging.info(f"datatype of columns in train file is : {train_column_data}")
            logging.info(f"datatype of columns in test file is : {test_column_data}")

            logging.info(f"datatype of columns in schema file is: {schema_column_data}")

            train_flag=False
            test_flag=False

            for column_name in schema_column_data.keys():
                if schema_column_data[column_name]!=train_column_data[column_name]:
                    logging.info(f"data type for {column_name} in train file is not correct")
                    return train_flag
                if schema_column_data[column_name]!=test_column_data[column_name]:
                    logging.info(f"data type for {column_name} in test file is not correct")
                    return test_flag
                
            logging.info("data type for train file is correct")
            logging.info("data type for test file is correct")
            train_flag=True
            test_flag=True

            return train_flag and test_flag
                
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_null_value_vaidation(self)->bool:
        try:
            logging.info(f"get null value validation function started")
            train_df,test_df = self.get_train_test_dataframe()

            train_null_count = dict(train_df.isnull().sum())
            test_null_count = dict(test_df.isnull().sum())

            logging.info(f"null value count in train data is : {train_null_count}")
            logging.info(f"null value count in test data is : {test_null_count}")

            for column_name,null_count in train_null_count.items():
                if null_count>0:
                    logging.info(f"null values found in {column_name}  columns of train file")
                    return False
                
            for column_name,null_count in test_null_count.items():
                if null_count>0:
                    logging.info(f"null values found in {column_name}  columns of test file")
                    return False
                
            logging.info(f"no null values found in train file")
            logging.info(f"no null values found in test file")
            return True
                
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_and_save_data_drift_report(self):
        try:
            logging.info(f"get and save data drift function started")
            report_dir = self.data_validation_config.report_page_file_dir
            os.makedirs(report_dir,exist_ok=True)
            report_file_path = os.path.join(report_dir,self.data_validation_config.report_name)

            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df,test_df = self.get_train_test_dataframe()
            dashboard.calculate(train_df,test_df)
            logging.info(f"{report_file_path}")
            dashboard.save(report_file_path)
            logging.info(f"report saved sucessfully")
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def intiate_data_validation(self)->DataValidationArtifact:
        try:
            validation5 = False
            validation1 = self.get_check_train_test_dir_exist
            if validation1:
                validation2 = self.get_column_count_validation()
            if validation2:
                validation3 = self.get_column_name_validation()
            if validation3:
                validation4 = self.get_column_data_type_validation()
            if validation4:
                validation5 = self.get_null_value_vaidation()
            self.get_and_save_data_drift_report()

            data_validation_artifact = DataValidationArtifact(is_validated=validation5,message="Data Vaalidation completed",
                                                              schema_file_path=self.data_validation_config.schema_file_dir,
                                                              reprot_file_path=self.data_validation_config.report_page_file_dir)
            return data_validation_artifact
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Validation log completed.{'<<'*20} \n\n")
    
