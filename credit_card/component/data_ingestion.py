import os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.entity.artifact_entity import DataIngestionArtifact
from credit_card.entity.config_entity import DataIngestionConfig
from six.moves import urllib
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self,data_ingestion_config : DataIngestionConfig) -> None:
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_download_credit_card_data(self)->str:
        try:
            logging.info(f"get download credit card data function started")
            dataset_download_url = self.data_ingestion_config.dataset_download_url
            zip_data_dir = self.data_ingestion_config.zip_data_dir

            os.makedirs(zip_data_dir,exist_ok=True)
            file_name = os.path.basename(dataset_download_url)

            zip_file_path = os.path.join(zip_data_dir,file_name)

            logging.info(f"downloding data from : {dataset_download_url}")
            logging.info(f"saving zip data in folder : {zip_data_dir}")
            logging.info(f"data download started")

            urllib.request.urlretrieve(dataset_download_url,zip_file_path)
            logging.info(f"data download completed")

            return zip_file_path
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_extract_data(self,zip_file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            
            os.makedirs(raw_data_dir,exist_ok=True)
            logging.info(f"exracting zip file into folder : {raw_data_dir}")

            with ZipFile(zip_file_path,'r') as zip:
                zip.extractall(raw_data_dir)
            logging.info(f"data extraction completed")

        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_train_test_split_data(self):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            data_file_name = os.listdir(raw_data_dir)[0]
            logging.info(f"data file name is : {data_file_name}")

            file_path = os.path.join(raw_data_dir,data_file_name)

            logging.info(f"reading file : {file_path}")
            creditcard_df = pd.read_excel(file_path)
            headers = creditcard_df.iloc[0]
            creditcard_df.columns = headers
            logging.info(f"reading data completed")

            logging.info(f"splitting data in train and test")
            X_train,X_test,y_train,y_test = train_test_split(creditcard_df.iloc[:,:-1],creditcard_df.iloc[:,-1],test_size=0.2, random_state=42)
            logging.info(f"splitting of data completed")

            data_file_name = data_file_name.replace('xls','csv')

            train_df = None
            test_df = None

            logging.info(f"combining input feature and target columns")
            train_df = pd.concat([X_train,y_train],axis=1)
            test_df = pd.concat([X_test,y_test],axis=1)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,data_file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,data_file_name)
            logging.info(f"train file path is :{train_file_path}")
            logging.info(f"test file path is :{test_file_path}")

            if train_df is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"moving train data as csv")
                train_df.to_csv(train_file_path,index=False)

            if train_df is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir,exist_ok=True)
                logging.info(f"moving test data as csv")
                train_df.to_csv(test_file_path,index=False)

            data_ingestion_artifact = DataIngestionArtifact(is_ingested=True,message="Data Ingestion completed successfully",
                                                            train_file_path=train_file_path,test_file_path=test_file_path)
            
            return data_ingestion_artifact
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def intitate_data_ingestion(self)->DataIngestionArtifact:
        try:
            zip_file_path = self.get_download_credit_card_data()
            raw_data = self.get_extract_data(zip_file_path=zip_file_path)
            return self.get_train_test_split_data()
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")