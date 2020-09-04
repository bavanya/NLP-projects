import GetOldTweets3 as got
import re 
import nltk
from nltk.tokenize import word_tokenize
import heapq
import preprocessor as p
nltk.download('punkt')

def get_tweets() :

  #getting tweets
  tweetCriteria = got.manager.TweetCriteria().setQuerySearch('covid')\
                                           .setSince("2019-08-01")\
                                           .setUntil("2020-09-02")\
                                           .setMaxTweets(5)
  tweets = got.manager.TweetManager.getTweets(tweetCriteria)
  text_tweets = [[tweet.text] for tweet in tweets]

  #combining the tweets in a string
  text = ""
  length = len(text_tweets)
  for i in range(0, length):
    text = text_tweets[i][0] + " " + text

  #data preprocessing
  text_cleaned = re.sub(r'\[[0-9]*\]', ' ', text)
  text_cleaned = re.sub(r'\s+', ' ', text_cleaned)
  text_cleaned = p.clean(text_cleaned)

  #tokenizing the string
  l = word_tokenize(text_cleaned)
  
  #finding weighted frequency of occurrence
  #not filtering stop words
  word_frequencies = {}
  for word in l:
    if word not in word_frequencies.keys():
      word_frequencies[word] = 1
    else:
      word_frequencies[word] += 1

  max_frequency = max(word_frequencies.values())

  for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/max_frequency)

  #finding sentence scores
  sentence_scores={}
  sentence_tokenize = nltk.sent_tokenize(text)
  for sentence in sentence_tokenize:
    if len(sentence.split(' ')) > 30:
      continue
    for word in nltk.word_tokenize(sentence.lower()):
      if word in word_frequencies.keys():
        if sentence not in sentence_scores.keys():
          sentence_scores[sentence] = word_frequencies[word]
        else:
          sentence_scores[sentence] += word_frequencies[word]


    #obtaining text summary
    summary = heapq.nlargest(8,sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary)
    print(summary)

get_tweets()