import os,sys,shutil
from credit_card.exception import CreditCardException
from credit_card.logger import logging
from credit_card.entity.artifact_entity import ModelEvulationArtifact,ModelPusherArtifact
from credit_card.entity.config_entity import ModelPusherConfig

class ModelPusher:


    def __init__(self,model_pusher_config : ModelPusherConfig,
                 model_evulation_artifact : ModelEvulationArtifact) -> None:
        try:
            logging.info(f"{'>>'*20}Model Pusher log started.{'<<'*20} \n\n")
            self.model_pusher_config = model_pusher_config
            self.model_evulation_artifact = model_evulation_artifact
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def export_model_dir(self)->ModelPusherArtifact:
        try:
            logging.info(f"export model dir function started")
            trained_models_path = self.model_evulation_artifact.evulation_model_file_path
            logging.info(f"trained models are at : {trained_models_path}")
            export_dir = self.model_pusher_config.export_dir_path
            
            export_dir_path_list = []
            for cluster_number in range(len(trained_models_path)):
                train_file_name = os.path.basename(trained_models_path[cluster_number])
                export_dir_path = os.path.join(export_dir,'cluster'+str(cluster_number))
                os.makedirs(export_dir_path,exist_ok=True)
                shutil.copy(src= trained_models_path[cluster_number],dst= export_dir_path)
                export_dir_path_list.append(export_dir_path)
            logging.info(f"all model are copied to : {export_dir_path_list}" )
            model_pusher_artifact = ModelPusherArtifact(export_dir_path=export_dir_path_list)
            return model_pusher_artifact
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def intiate_model_pusher(self)->ModelPusherArtifact:
        try:
            logging.info(f"intiate model pusher function started")
            return self.export_model_dir()

        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Model Pusher log completed.{'<<'*20} \n\n")