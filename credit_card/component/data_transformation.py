import os,sys,csv
from credit_card.logger import logging
from credit_card.exception import CreditCardException
from credit_card.entity.config_entity import DataTransformConfig
from credit_card.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformArtifact
import pandas as pd
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline
import dill
from sklearn.base import BaseEstimator,TransformerMixin
from credit_card.util.util import read_yaml
from credit_card.constant import NO_CLUSTER,TARGET_COLUMN_KEY
import matplotlib.pyplot as plt
from pathlib import Path


class trans(BaseException,TransformerMixin):
  def __init(self):
    pass

  def fit(self,X,y=None):
    return self

  def transform(self,X,y=None):
    X=pd.get_dummies(X,columns = ['SEX','MARRIAGE'],drop_first=True,dtype='int64')
    global column_trans
    column_trans = X.columns
    logging.info(f"columns after transformation are :{column_trans}")
    return X

class DataTransformation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact,
                 data_transform_config:DataTransformConfig,
                 ) -> None:
        try:
            logging.info(f"{'>>'*20}Data Transformation log started.{'<<'*20} ")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transform_config = data_transform_config
            self.target_column = read_yaml(file_path=self.data_validation_artifact.schema_file_path)[TARGET_COLUMN_KEY]
        except Exception as e:
            raise CreditCardException(e,sys) from e
    
    @staticmethod
    def log_transform(X):
        try:
            logging.info(f"log transform function started")
            return np.log(X+20)
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_pre_processing_object(self)->Pipeline:
        try:
            logging.info("get preprocessing object function started")

            logging.info("making transformer object for log transform")
            transformer = FunctionTransformer(DataTransformation.log_transform)

            logging.info(f"pre processing pipeline esamble started")
            pipeline = Pipeline(steps=[('transform',trans()),
                                      ('scaler',StandardScaler())])
            logging.info(f"pre processing pipeline esamble completed : {pipeline}")

            return pipeline
            
        except Exception as e:
            raise CreditCardException(e,sys) from e

    def perform_pre_processing(self,pre_processing_object:Pipeline, is_test_data:bool=False)->pd.DataFrame:
        try:
            logging.info(f"perform pre processing function started")
            target_column = self.target_column
            logging.info(f"target column is : {target_column}")

            if is_test_data==False:
                train_file_path = self.data_ingestion_artifact.train_file_path
                logging.info(f"train file path is : {train_file_path}")

                train_df = pd.read_csv(train_file_path)
                logging.info(f"data read sucessfull")

                target_df = train_df.iloc[:,-1]
                logging.info(f"dropping target column from train data")
                train_df.drop(target_column,axis=1,inplace=True)
                train_df.drop('ID',axis=1,inplace=True)
                columns = train_df.columns
                logging.info(f"column name after dropping target columns : {columns}")
                
                train_df = pre_processing_object.fit_transform(train_df)
                
                logging.info(f"preprocessing on train data is performed")

                train_df = pd.DataFrame(train_df,columns=column_trans)
                train_df = pd.concat([train_df,target_df],axis=1)
                
                logging.info(f"combining train and target dataframe after preprocessing")
                return train_df
            else:
                test_file_path = self.data_ingestion_artifact.test_file_path
                logging.info(f"test file path is : {test_file_path}")

                test_df = pd.read_csv(test_file_path)
                logging.info(f"data read sucessfull")

                target_df = test_df.iloc[:,-1]
                logging.info(f"dropping target column from test data")
                test_df.drop(target_column,axis=1,inplace=True)
                
                columns = test_df.columns
                logging.info(f"column name after dropping target columns : {columns}")
                test_df.drop('ID',axis=1,inplace=True)
                test_df = pre_processing_object.transform(test_df)
                logging.info(f"preprocessing on test data is performed")

                test_df = pd.DataFrame(test_df,columns=column_trans)
                test_df = pd.concat([test_df,target_df],axis=1)
                logging.info(f"combining test and target dataframe after preprocessing")
                return test_df
        except Exception as e:
            raise CreditCardException(e,sys) from e
            
    def get_and_save_graph_cluster(self,train_df:pd.DataFrame):
        try:
            logging.info(f"get and save graph cluster function started")

            logging.info(f"making k-means object")
            kmeans = KMeans(init='k-means++',random_state=42)
            logging.info(f"making visualizer object and fitting input train data")
            visulizer = KElbowVisualizer(kmeans,k=(2,11))
            visulizer.fit((train_df.drop(self.target_column,axis=1)))

            graph_path = self.data_transform_config.graph_save_dir
            logging.info(f"saving graph at: {graph_path}")
            os.makedirs(graph_path,exist_ok=True)
            file_save_path = os.path.join(graph_path,'graph_cluster.png')
            visulizer.show(file_save_path)

            logging.info(f"graph saving completed")
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def get_and_save_silhouette_score_graph(self,train_df:pd.DataFrame):
        try:
            logging.info(f"get and save silhouette score graph function started")
            fig, ax = plt.subplots(2, 2, figsize=(15,8))
            for no_of_clusters in [2,3,4,5]:
                logging.info(f"finding and saving graph for silhouette score for {no_of_clusters} clusters")
                kmeans = KMeans(n_clusters=no_of_clusters, init='k-means++',random_state=42)
                q, mod = divmod(no_of_clusters, 2)
                visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick',ax=ax[q-1][mod])
                visualizer.fit((train_df.drop(self.target_column,axis=1)))
                score_graph_path = self.data_transform_config.graph_save_dir
                os.makedirs(score_graph_path,exist_ok=True)
                file_save_path = os.path.join(score_graph_path,"cluster_"+str(no_of_clusters)+"_silihoetter_score.png") 
                
                visualizer.show(file_save_path)

        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def save_data_based_on_cluster(self,n_clusters,train_df,test_df):
        try:
            logging.info(f"save data based on cluster function started")

            logging.info(f"making cluster object and fitting data")
            kmeans = KMeans(n_clusters=n_clusters,init='k-means++',random_state=42)
            kmeans.fit((train_df.drop(self.target_column,axis=1)))

            logging.info(f"prediting train data's cluster")
            train_predict = kmeans.predict((train_df.drop(self.target_column,axis=1)))

            transform_train_folder = self.data_transform_config.transform_train_dir
            logging.info(f"train cluster files will be saved: {transform_train_folder}")
            os.makedirs(transform_train_folder,exist_ok=True)
            
            column_names = list(train_df.columns)
            logging.info(f"training columns are : {column_names}")

            cluster_numbers = list(np.unique(np.array(train_predict)))
            logging.info(f"cluster numbers are : {cluster_numbers}")

            logging.info(f"making csv files for training data cluster wise")

            for cluster_number in cluster_numbers:
                file_path = os.path.join(transform_train_folder,'train_cluster'+str(cluster_number)+'.csv')
                with Path(file_path).open('w',newline='') as csvfiles:
                    csvwriter = csv.writer(csvfiles)

                    csvwriter.writerow(column_names)
                    for index in range(len(train_predict)):
                        if train_predict[index]==cluster_number:
                            csvwriter.writerow(train_df.iloc[index])
            logging.info(f"csv files write for train data is completed")

            logging.info(f"prediting test data's cluster")
            test_predict = kmeans.predict((test_df.drop(self.target_column,axis=1)))

            transform_test_folder = self.data_transform_config.transform_test_dir
            logging.info(f"test cluster files will be saved: {transform_test_folder}")
            os.makedirs(transform_test_folder,exist_ok=True)
            
            column_names = list(test_df.columns)
            logging.info(f"testing columns are : {column_names}")

            cluster_numbers = list(np.unique(np.array(test_predict)))
            logging.info(f"cluster numbers are : {cluster_numbers}")

            logging.info(f"making csv files for testing data cluster wise")

            for cluster_number in cluster_numbers:
                file_path = os.path.join(transform_test_folder,'test_cluster'+str(cluster_number)+'.csv')
                with Path(file_path).open('w',newline='') as csvfiles:
                    csvwriter = csv.writer(csvfiles)

                    csvwriter.writerow(column_names)
                    for index in range(len(test_predict)):
                        if test_predict[index]==cluster_number:
                            csvwriter.writerow(test_df.iloc[index])
            logging.info(f"csv files write for test data is completed")

            return kmeans
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def intiate_data_transform(self)->DataTransformArtifact:
        try:
            preprocessed_obj = self.get_pre_processing_object()
            preprocessed_dir = os.path.dirname(self.data_transform_config.preprocessed_file_path)
            os.makedirs(preprocessed_dir,exist_ok=True)
            with open(self.data_transform_config.preprocessed_file_path,'wb') as objfile:
                dill.dump(preprocessed_obj,objfile)
            logging.info(f"pre processing object saved")

            train_df = self.perform_pre_processing(pre_processing_object=preprocessed_obj)

            test_df = self.perform_pre_processing(pre_processing_object=preprocessed_obj,is_test_data=True)

            self.get_and_save_graph_cluster(train_df=train_df)
            self.get_and_save_silhouette_score_graph(train_df=train_df)

            k_means_object = self.save_data_based_on_cluster(n_clusters=NO_CLUSTER,train_df=train_df,test_df=test_df)
            cluster_model_path = os.path.dirname(self.data_transform_config.cluster_model_file_path)    
            logging.info(f"saving cluster model object at : {cluster_model_path}")
            os.makedirs(cluster_model_path,exist_ok=True)
            with open(self.data_transform_config.cluster_model_file_path,'wb') as obj_file:
                dill.dump(k_means_object,obj_file)

            data_transform_artifact = DataTransformArtifact(transform_train_dir=self.data_transform_config.transform_train_dir,
                                                            transform_test_dir=self.data_transform_config.transform_test_dir,
                                                            preprocessed_dir=self.data_transform_config.preprocessed_file_path,
                                                            cluster_model_dir=self.data_transform_config.cluster_model_file_path,
                                                            is_transform=True,
                                                            message="Data transform sucessfull")
            return data_transform_artifact
            
        except Exception as e:
            raise CreditCardException(e,sys) from e
        

    def __del__(self):
        logging.info(f"{'>>'*20}Data Transformation log completed.{'<<'*20} \n\n")
    

    