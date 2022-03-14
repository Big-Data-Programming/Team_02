import mlflow 
import pandas as pd
import tweepy
import nltk
import spacy
import transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import re
import string



 
auth = tweepy.OAuthHandler(
   API_key, API_Secert)

api = tweepy.API(auth)

list_tweet = []

for tweet in tweepy.Cursor(api.search, q= 'Oil').items(200):
  list_tweet.append(tweet) 

tweets = []
for i in range(0,len(list_tweet)):
 
  tweets.append(list_tweet[i].text)

  data_api = pd.DataFrame({'text':tweets}, index = range(0,len(tweets)))





def get_inputs(text:string):
  #Tokenizer, # cleaning , 

  nlp = spacy.load('en')
  ps = nltk.PorterStemmer()
  nltk.download('wordnet')
  stopwords = nltk.corpus.stopwords.words('english')
  
  for t in nlp(text):
    if t.like_url:
      pass
    else:
      text = "".join([t for t in text if t not in string.punctuation])
      toke = re.split("\W+", text)
      final = [word for word in toke if word not in stopwords]
      stem_text = [ps.stem(word) for word in final]

  tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
  model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
  tokenized = tokenizer.encode(stem_text)
  input_text = pad_sequences([tokenized], maxlen=30, padding="pre")
  attention_mask = np.where(input_text != 0 ,1,0)
  input = torch.tensor(input_text)
  atten = torch.tensor(attention_mask)
  op = model(input, attention_mask = atten)
  dt = op.last_hidden_state[:,0,:].detach().numpy()
  

  return dt

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    modelName = 'Random_Forest'
    stage = 'Staging'
    # registry_uri = 'sqlite:///mlflow.db'
    # mlflow.tracking.set_tracking_uri(registry_uri)
    model = mlflow.pyfunc.load_model(
        modelUri =f'model:/{modelName}/{stage}'
    )


    input_file = get_inputs(data_api[0])
    result = model.predict(input_file)
    print(result)
