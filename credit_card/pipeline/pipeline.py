import os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformArtifact
from credit_card.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformConfig
from credit_card.config.configuration import Configuration
from credit_card.component.data_ingestion import DataIngestion
from credit_card.component.data_validation import DataValidation
from credit_card.component.data_transformation import DataTransformation

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
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_vaidation_config=self.config.get_data_validation_config())
            return data_validation.intiate_data_validation()
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def start_data_transform(self,data_ingestion_artifact:DataIngestionArtifact,
                             data_validation_artifact:DataValidationArtifact)->DataTransformArtifact:
        try:
            data_transform = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                data_validation_artifact=data_validation_artifact,
                                                data_transform_config=self.config.get_data_transform_config())
            return data_transform.intiate_data_transform()
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transform_artifact = self.start_data_transform(data_ingestion_artifact=data_ingestion_artifact,
                                                                data_validation_artifact=data_validation_artifact)
        except Exception as e:
            raise CreditCardException(e,sys) from e