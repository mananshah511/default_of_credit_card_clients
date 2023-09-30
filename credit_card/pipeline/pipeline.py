import os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.entity.artifact_entity import DataIngestionArtifact
from credit_card.entity.config_entity import DataIngestionConfig
from credit_card.config.configuration import Configuration
from credit_card.component.data_ingestion import DataIngestion

class Pipeline:
    def __init__(self,config:Configuration=Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.intitate_data_ingestion()
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise CreditCardException(e,sys) from e