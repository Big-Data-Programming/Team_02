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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TweetAnalytics:

  def __init__(self):
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
    combined_df = pd.read_csv(path_to_csv,delimiter= ',')
    combined_df['tweet_orig'] = combined_df['tweet']
    self.text_df = combined_df['tweet']
    self.target_df = combined_df['sentiment']

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

  def train_with_mlflow(self, model_function, model_name):
    with mlflow.start_run() as run: 
      mlflow.sklearn.autolog()
      model = model_function()
      mlflow.sklearn.log_model(model,"model", registered_model_name=model_name)
  
  def fetch_twitter_data(self, keyword):
    tweets = []
    for tweet in self.api.search_tweets(keyword):
      tweets.append(tweet.text)
    self.text_df = pd.DataFrame(tweets, columns=['tweet'])
    self.text_df = self.text_df['tweet']

  def test_with_model(self, model_name, model_stage):
    model = mlflow.pyfunc.load_model(
        model_uri =f'models:/{model_name}/{model_stage}'
    )
    print(model.predict(self.train_df))


if __name__ =="__main__":
  tweet_analytics = TweetAnalytics()
  model_name = 'Random Forest Classifier'
  model_stage = 'None'
  tweet_analytics.read_and_fetch_data('./data/tweets_1000.csv')
  tweet_analytics.setup_data_for_ml(train=True)
  tweet_analytics.train_with_mlflow(
    tweet_analytics.train_with_random_forest_classifier,
    model_name
  )
  tweet_analytics.fetch_twitter_data('oil')
  tweet_analytics.setup_data_for_ml()
  tweet_analytics.test_with_model(model_name, model_stage)