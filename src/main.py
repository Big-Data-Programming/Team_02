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
    self.combined_df = pd.read_csv(path_to_csv,delimiter= ',')
    self.combined_df['tweet_orig'] = self.combined_df['tweet']
    self.combined_df['tweet'] = self.combined_df['tweet'].apply(lambda x: self._clean_tweet(x))

  def setup_data_for_ml(self):
    self.text_df = self.combined_df['tweet']
    target_df = self.combined_df['sentiment']
    tokenized_data = self.text_df.apply(lambda tweet : self.tokenizer.encode(tweet, max_length= 30))
    input_text = pad_sequences(tokenized_data, maxlen=30, padding="post")
    attention_mask = np.where(input_text != 0 ,1,0)
    input_data_tensor = torch.tensor(input_text)
    input_mask_tensor = torch.tensor(attention_mask) 
    with torch.no_grad():
      embedded_matrix = self.model(input_data_tensor, attention_mask = input_mask_tensor)
    train_df = embedded_matrix.last_hidden_state[:,0,:].numpy()
    target_df = target_df.to_numpy()
    self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(train_df, target_df, test_size = 0.2, random_state=15)

  def train_with_random_forest_classifier(self):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(self.train_X, self.train_y)
    return clf

  def train_with_mlflow(self, model_function, model_name):
    with mlflow.start_run() as run: 
      mlflow.sklearn.autolog()
      model = model_function()
      mlflow.sklearn.log_model(model,"model", registered_model_name=model_name)
    

if __name__ =="__main__":
  tweet_analytics = TweetAnalytics()
  tweet_analytics.read_and_fetch_data('./data/tweets_1000.csv')
  tweet_analytics.setup_data_for_ml()
  tweet_analytics.train_with_mlflow(
    tweet_analytics.train_with_random_forest_classifier,
    "Random Forest Classifier"
  )