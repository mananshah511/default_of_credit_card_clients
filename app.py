from flask import Flask,request,jsonify,render_template
from flask_cors import CORS,cross_origin
from credit_card.entity.artifact_entity import DataTransformArtifact,ModelArtifactConfig,ModelEvulationArtifact
import pickle
from credit_card.util.util import read_yaml
from credit_card.constant import BEST_MODEL_KEY,MODEL_PATH_KEY
from credit_card.logger import logging
import pandas as pd
import numpy as np
from credit_card.pipeline.pipeline import Pipeline

columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
class CONFIG:
    def __init__(self,data_transform_artifact:DataTransformArtifact,
                 model_trainer_artifact:ModelArtifactConfig,
                 model_evulation_artifact:ModelEvulationArtifact) -> None:
        self.data_transform_artifact = data_transform_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evulation_artifact = model_evulation_artifact

app = Flask(__name__)

@app.route('/',methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    #config = CONFIG()
    cluster_model_path = r"credit_card/artifact/data_transform/2023-10-02-16-09-17/cluster_model/cluster_model.pkl"
    cluster_model = pickle.load(open(cluster_model_path,'rb'))

    preprocessed_object_path = r"D:\projects\default_of_credit_card_clients\credit_card\artifact\data_transform\2023-10-02-16-09-17\preprocessed\preprocessed.pkl"
    preprocessed_model = pickle.load(open(preprocessed_object_path,'rb'))

    data = [float(x) for x in request.form.values()]
    data = np.array(data)
    
    logging.info(f"{data}")
    data = pd.DataFrame(data.reshape(-1,len(data)),columns=columns)
    data['SEX']=data['SEX'].astype('int')
    data['MARRIAGE']=data['MARRIAGE'].astype('int')
    data['EDUCATION']=data['EDUCATION'].astype('int')
    train_df = pd.read_csv(r'D:\projects\default_of_credit_card_clients\credit_card\artifact\data_ingestion\2023-10-02-16-09-17\ingested_data\train\default of credit card clients.csv')
    train_df.drop(['ID','default payment next month'],axis=1,inplace=True)
    logging.info(f"{data}")
    data_preprocessed = preprocessed_model.transform(pd.concat([data,train_df]))
    logging.info(f"{data_preprocessed[0]}")
    cluster_number = cluster_model.predict(data_preprocessed[0].reshape(1,30))[0]

    model_file =  read_yaml(r"D:\projects\default_of_credit_card_clients\credit_card\artifact\model_evulation\model_evulation.yaml")
    model_path = model_file['cluster'+str(cluster_number)][BEST_MODEL_KEY][MODEL_PATH_KEY]

    best_model = pickle.load(open(model_path,'rb'))
    outcome = best_model.predict(np.array(data_preprocessed[0].reshape(1,30)))

    if int(outcome[0])==0:
        return render_template('index.html', prediction_text = "This person will not be defaulter next month")
    else:
        return render_template('index.html', prediction_text = "This person will be defaulter next month")


@app.route('/train',methods=['GET'])
@cross_origin()
def train():
    pip = Pipeline()
    pip.run_pipeline()
    return render_template('index.html', prediction_text = "Training completed successfully")


if __name__ == "__main__":
    app.run()
