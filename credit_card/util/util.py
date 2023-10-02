import yaml,os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
import dill


def read_yaml(file_path:str):
    try:
        with open(file_path,"rb") as yaml_file:
            logging.info(f"reading yaml file from : {file_path}")
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CreditCardException(e,sys) from e

def write_yaml_file(file_path:str, data:dict=None):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        
        with open(file_path,"w") as yaml_file:
            yaml.dump(data, yaml_file)
    except Exception as e:
        raise CreditCardException(e,sys) from e  
    
def load_object(file_path:str):
    try:
        with open(file_path,"rb") as object_file:
            return dill.load(object_file)
    except Exception as e:
        raise CreditCardException(e,sys) from e