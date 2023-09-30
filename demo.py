from credit_card.config.configuration import Configuration
from credit_card.pipeline.pipeline import Pipeline

config = Configuration()
config.get_data_validation_config()

#pip = Pipeline()
#pip.start_data_ingestion()