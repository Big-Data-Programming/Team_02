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


nlp = spacy.load("en_core_web_sm")
ps = nltk.PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')

def cmb_fun(text):
  for t in nlp(text):
    if t.like_url:
      pass
    else:
      text = "".join([t for t in text if t not in string.punctuation])
      toke = re.split("\W+", text)
      final = [word for word in toke if word not in stopwords]
      stem_text = [ps.stem(word) for word in final]

  
  
  return stem_text

def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

mlflow.set_registry_uri("http://127.0.0.1:5000")
registry_uri = 'sqlite:///mlflow.db'
mlflow.tracking.set_registry_uri(registry_uri)


nltk.download('wordnet')

nltk.download('stopwords')


if __name__ =="__main__":


  combined_df = pd.read_csv( "tweeets_1000.csv",delimiter= ',' )


  combined_df.head()
  combined_df.sentiment.value_counts()
  combined_df['text_clean'] = combined_df['tweet'].apply(lambda x: cmb_fun(x))
  combined_df.text_clean



  tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
  model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased') 

  text_df = combined_df['text_clean']
  target_df = combined_df['sentiment']
  tokenized_data = text_df.apply(lambda x : tokenizer.encode(x, max_length= 30))

  pd.set_option('display.width', 100)
  tokenized_data



  input_text = pad_sequences(tokenized_data, maxlen=30, padding="post")

  attention_mask = np.where(input_text != 0 ,1,0)
  attention_mask.shape, input_text.shape

  input_ = torch.tensor(input_text)
  atten = torch.tensor(attention_mask)

  print(input_.shape)
  print(atten.shape)

  with torch.no_grad():
    op = model(input_, attention_mask = atten)
  op[0].shape

  dt = op.last_hidden_state[:,0,:].numpy()
  target_df_1 = target_df.to_numpy()

  train_X, valid_X, train_y, valid_y = train_test_split(dt, target_df_1, test_size = 0.2, random_state=15)

  train_X.shape
  train_y.shape








  

  
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

  
