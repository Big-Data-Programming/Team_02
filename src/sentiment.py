from urllib.parse import urlparse
import pandas as pd
import nltk

import re
import string

import spacy
import transformers
import torch 
from transformers import DistilBertModel 
import numpy as np
import mlflow
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences

def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts





if __name__ =="__main__":

  
  with mlflow.start_run() as run: 
    mlflow.sklearn.autolog()
    clf = RandomForestClassifier(max_depth=2, random_state=0)


    clf.fit(train_X, train_y)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme  

    if tracking_url_type_store != "file":

      #Register the model
      mlflow.sklearn.log_model(clf, "model", registered_model_name= "Random_Forest")
    else:
      mlflow.sklearn.log_model(clf,"model")


    params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)


    pprint(metrics)

  
