import yaml,os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException


def read_yaml(file_path:str):
    try:
        with open(file_path,"rb") as yaml_file:
            logging.info(f"reading yaml file from : {file_path}")
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CreditCardException(e,sys) from e
