import re
import string
import json

import pandas as pd
import numpy as np
import spacy
import transformers
import torch 
import mlflow
import nltk
import tweepy
import hashlib
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pymongo import MongoClient

class TweetAnalytics:

  def __init__(self):
    client = MongoClient(host='localhost', port=27019, username='root', password='pass')
    db = client.commodity
    self.data = db.data
    with open('./config.json') as config_file:
      config = json.load(config_file)
    mlflow.set_tracking_uri(config['tracking_uri'])
    registry_uri = config['tracking_database_uri']
    mlflow.tracking.set_tracking_uri(registry_uri)
    try:
      nltk.download('wordnet')
      nltk.download('stopwords')
      self.nlp = spacy.load("en_core_web_sm")
    except IOError:
      print('Error fetching spacy module or nltk')
    self.ps = nltk.PorterStemmer()
    self.stopwords = nltk.corpus.stopwords.words('english')
    self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    self.model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
    auth = tweepy.OAuth1UserHandler(config['api_key'], config['api_secret'], config['access_token'], config['access_token_secret'])
    self.api = tweepy.API(auth)

  def _clean_tweet(self, tweet):
    stem_text = ''
    for t in self.nlp(tweet):
      if t.like_url:
        pass
      else:
        text = "".join([t for t in tweet if t not in string.punctuation])
        token = re.split("\W+", text)
        final = [word for word in token if word not in self.stopwords]
        stem_text = [self.ps.stem(word) for word in final]
    return stem_text

  def read_and_fetch_data(self, path_to_csv):
    data = pd.read_csv(path_to_csv,delimiter= ',', encoding='latin-1', header= None)
    data.columns = ['sentiment', 'id', 'date','from','user','tweet']
    data = data.drop(['id','date','from','user'] ,axis =1)
    self.total = 2000
    new_data_po = data[data['sentiment']==4].sample(int(self.total / 2))
    new_data_ne = data[data['sentiment']==0].sample(int(self.total / 2))
    data = pd.concat([new_data_ne,new_data_po])
    self.filename = hashlib.md5(data.to_csv().encode('utf-8')).hexdigest() + '.csv'
    self.filepath = './data/' + self.filename
    data.to_csv(self.filepath)
    self.text_df = data['tweet']
    self.target_df = data['sentiment']

  def setup_data_for_ml(self, train = False):
    self.text_df = self.text_df.apply(lambda x: self._clean_tweet(x))
    tokenized_data = self.text_df.apply(lambda tweet : self.tokenizer.encode(tweet, max_length= 30))
    input_text = pad_sequences(tokenized_data, maxlen=30, padding="post")
    attention_mask = np.where(input_text != 0 ,1,0)
    input_data_tensor = torch.tensor(input_text)
    input_mask_tensor = torch.tensor(attention_mask) 
    with torch.no_grad():
      embedded_matrix = self.model(input_data_tensor, attention_mask = input_mask_tensor)
    self.train_df = embedded_matrix.last_hidden_state[:,0,:].numpy()
    if train:
      self.target_df = self.target_df.to_numpy()
      self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(self.train_df, self.target_df, test_size = 0.2, random_state=15)
   
  def train_with_random_forest_classifier(self):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(self.train_X, self.train_y)
    return clf

  def train_with_logistic_regression(self):
    clf = LogisticRegression()
    clf.fit(self.train_X, self.train_y)
    return clf  

  def train_with_svc(self):
    clf = SVC()
    clf.fit(self.train_X, self.train_y)
    return clf

  def train_with_knc(self):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(self.train_X, self.train_y)
    return clf

  def train_with_nbc(self):
    clf = GaussianNB()
    clf.fit(self.train_X, self.train_y)
    return clf

  def train_with_mlflow(self, model_function, model_name):
    with mlflow.start_run() as run: 
      mlflow.sklearn.autolog()
      model = model_function()
      predicted = model.predict(self.valid_X)
      cm = confusion_matrix(predicted, self.valid_y)
      tn, fp, fn, tp = cm.ravel()
      mlflow.sklearn.log_model(model,"model", registered_model_name=model_name)
      mlflow.log_metric("tn", tn)
      mlflow.log_metric('fp', fp)
      mlflow.log_metric('fn', fn)
      mlflow.log_metric('tp', tp)
      mlflow.log_metric('validation precision score', tp / (tp + fp))
      mlflow.log_metric('validation recall score', tp / (tp + fn))
      mlflow.log_metric('data size', self.total)
      mlflow.log_param('filepath', self.filepath)
  
  def fetch_twitter_data(self, keyword):
    tweets = []
    for tweet in self.api.search_tweets(keyword):
      tweets.append(tweet.text)
    self.text_df = pd.DataFrame(tweets, columns=['tweet'])
    self.text_df = self.text_df['tweet']

  def test_with_models(self, model_names, model_stage):
    models = []
    for model_name in model_names:
      model = mlflow.pyfunc.load_model(
          model_uri =f'models:/{model_name}/{model_stage}'
      )
      models.append({
        model_name: model_name,
        model: model 
      })
    data = {}
    for index in range(0, len(self.train_df)):
      tweet = self.text_df[index]
      for model in models:      
        result = model.model.predict(self.train_df[index])
        name = model.model_name
        data[name] = result
      data['tweet'] = tweet
      self.data.insert_one(data)
    # self.data


if __name__ =="__main__":
  tweet_analytics = TweetAnalytics()
  model_stage = 'None'
  tweet_analytics.read_and_fetch_data('./tweets_1mil.csv')
  tweet_analytics.setup_data_for_ml(train=True)
  rfc_model_name = 'Random_Forest_Classifier'
  tweet_analytics.train_with_mlflow(
    tweet_analytics.train_with_random_forest_classifier,
    rfc_model_name
  )
  lr_model_name = 'Logistic_Regression'
  tweet_analytics.train_with_mlflow(
    tweet_analytics.train_with_logistic_regression,
    lr_model_name
  )
  svc_model_name = 'Support_Vector_Classifier'
  tweet_analytics.train_with_mlflow(
    tweet_analytics.train_with_svc,
    svc_model_name
  )
  knc_model_name = 'K_Neighbours_Classifier'
  tweet_analytics.train_with_mlflow(
    tweet_analytics.train_with_knc,
    knc_model_name
  )
  nbc_model_name = 'Naive_Bayes_Classifier'
  tweet_analytics.train_with_mlflow(
    tweet_analytics.train_with_nbc,
    nbc_model_name
  )
  tweet_analytics.fetch_twitter_data('oil')
  tweet_analytics.setup_data_for_ml()
  tweet_analytics.test_with_models([
    rfc_model_name,
    lr_model_name,
    svc_model_name,
    knc_model_name,
    nbc_model_name
  ], model_stage)