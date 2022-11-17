import numpy as np
import nltk
import os
from matplotlib import pyplot as plt

TRAINING_FILE = 'noCR_train.txt'
TESTING_FILE = 'noCR_test.txt'
ELECTION_FILE = 'Tweets_election_trial.txt'

LABELS = {
  'positive',
  'negative'
}

SENTIMENT_MAP = {
  '1': 'positive',
  '0': 'negative'
}

TOKENIZATION_POLICIES = {
  'naive': nltk.tokenize.WhitespaceTokenizer(),
  'punctuation': nltk.tokenize.WordPunctTokenizer(),
  'treebank': nltk.tokenize.TreebankWordTokenizer(),
  'tweet': nltk.tokenize.TweetTokenizer(
    preserve_case=('True' == os.environ.get('PRESERVE_CASE','True')),
    strip_handles=('True' == os.environ.get('STRIP_HANDLES','False'))
  )
}

# simple helper
def accumulate_sentiment(sentiment):
  sentiment_hash[sentiment] += 1
  sentiment_hash['total'] += 1

# compute logpriors based on existing data, depends on `sentiment_hash` existing
# having been populated with frequency data already
def compute_logpriors(sentiment_hash):
  return dict((
    (
      'positive',
      np.log(sentiment_hash['positive']/sentiment_hash['total'])
    ),
    (
      'negative',
      np.log(sentiment_hash['negative']/sentiment_hash['total'])
    )
  ))

# add words from tweet lines to
# - vocabulary set
# - conditional frequency dictionaries
def accumulate_words(sentiment, line_as_word_list):
  for word in line_as_word_list:
    vocabulary.add(word)

    if word in word_counts[sentiment]:
      word_counts[sentiment][word] += 1
    else:
      word_counts[sentiment][word] = 1

# apply tokenization policy to line inputs
def process_line(line):
  policy = os.environ.get('TOKENIZATION','naive')
  tokenizer = TOKENIZATION_POLICIES[policy]
  tokenized_line_as_list = tokenizer.tokenize(line)
  return tokenized_line_as_list

# each entry in sub-dictionaries:
#   {word: frequency of word in tweet corpus given sentiment}
#     e.g. {'falafel', 6}
word_counts = {
  'positive': {},
  'negative': {}
}

# unique words, agnostic of class
vocabulary = set()

# each entry in sub-dictionaries:
#   {word: loglikelihood of observing word given sentiment}
#     e.g. {'falafel', -12.71383}
word_loglikelihoods = {
  'positive': {},
  'negative': {}
}

# class frequency, used for computing P(c), the
# prior probabilities of a tweet's sentiment
sentiment_hash = {
  'positive': 0,
  'negative': 0,
  'total': 0
}

# compute logpriors, word distributions for both classes
with open(TRAINING_FILE) as f:
  for raw_line in f:
    sentiment_str, tweet = raw_line.split('\t')
    sentiment = SENTIMENT_MAP[sentiment_str]
    if tweet: # if there is a tweet associated with the sentiment
      accumulate_sentiment(sentiment)
      accumulate_words(sentiment, process_line(tweet))

log_priors = compute_logpriors(sentiment_hash)
words_by_class = dict([(k,len(v)) for k,v in word_counts.items()])
word_counts_by_class = dict([(k,sum(v.values())) for k,v in word_counts.items()])

# compute loglikelihoods for each word based on class
for label in LABELS:
  total_word_count = word_counts_by_class[label]
  for word in vocabulary:
    if word in word_counts[label]:
      count = word_counts[label][word]
    else:
      count = 0
    total = 0
    alpha = int(os.environ.get('ALPHA','1'))
    word_loglikelihoods[label][word] = np.log(
      (count + alpha) / (total_word_count + len(vocabulary) + alpha)
    )

training_accuracy = 0.0
testing_accuracy = 0.0

# get training data accuracy
with open(TRAINING_FILE) as f:
  num_training_examples = 0
  num_training_correct = 0
  for raw_line in f:
    num_training_examples += 1
    sums = {
      'positive': 0,
      'negative': 0
    }
    prediction = -1
    sentiment_str, tweet = raw_line.split('\t')

    for label in LABELS:
      sums[label] = log_priors[label]
      for word in process_line(tweet):
        if word in vocabulary:
          sums[label] += word_loglikelihoods[label][word]
    
    # evaluating loglikelihoods
    diff = sums['positive'] - sums['negative']
    if diff > 0.0:
      prediction = 1
    else:
      prediction = 0

    if prediction == int(sentiment_str):
      num_training_correct += 1

  training_accuracy = num_training_correct/num_training_examples

# repeat for testing data
with open(TESTING_FILE) as f:
  num_testing_examples = 0
  num_testing_correct = 0
  for raw_line in f:
    num_testing_examples += 1
    sums = {
      'positive': 0,
      'negative': 0
    }
    prediction = -1
    sentiment_str, tweet = raw_line.split('\t')

    for label in LABELS:
      sums[label] = log_priors[label]
      for word in process_line(tweet):
        if word in vocabulary:
          sums[label] += word_loglikelihoods[label][word]
    
    # evaluating loglikelihoods
    diff = sums['positive'] - sums['negative']
    if diff > 0.0:
      prediction = 1
    else:
      prediction = 0

    if prediction == int(sentiment_str):
      num_testing_correct += 1
    
  testing_accuracy = num_testing_correct/num_testing_examples

print('training data accuracy: ' + str(training_accuracy))
print('testing data accuracy: ' + str(testing_accuracy))

########## SECTION 2: ELECTION TWEETS ##########
analyze_election_tweets = 'True' == os.environ.get('ELECTION','False')

if analyze_election_tweets:
  # do election analysis
  num_positive_tweets = 0
  num_total_tweets = 0
  tweet_times = []
  tweet_sentiments = []
  with open(ELECTION_FILE) as f:
    for raw_line in f:
      num_total_tweets += 1
      prediction = -1
      sums = {
        'positive': 0,
        'negative': 0
      }
      username, user_screen_name, time, is_retweet, tweet = raw_line.split('\t')
      for label in LABELS:
        sums[label] = log_priors[label]
        for word in process_line(tweet):
          if word in vocabulary:
            sums[label] += word_loglikelihoods[label][word]
    
      # evaluating loglikelihoods
      diff = sums['positive'] - sums['negative']
      if diff > 0.0:
        prediction = 1
        num_positive_tweets += 1
      else:
        prediction = 0
      
      tweet_times.append(time)
      tweet_sentiments.append(prediction)

  # hash containing day strings and positive, negative tweet counts
  # e.g. {'2020-11-01': 10} (10 negative tweets on Nov 1, 2020)
  tweet_days_positive_counts = {}
  tweet_days_negative_counts = {}

  current_day = '2020-10-27' # start day of election tweets
  num_pos_tweet = 0
  num_neg_tweet = 0
  for tweet_time, tweet_sentiment in zip(tweet_times,tweet_sentiments):
    tweet_day = tweet_time[:10]
    if tweet_day not in tweet_days_positive_counts:
      tweet_days_positive_counts[tweet_day] = 0
      tweet_days_negative_counts[tweet_day] = 0

    if tweet_sentiment == 1:
      tweet_days_positive_counts[tweet_day] += 1
    else:
      tweet_days_negative_counts[tweet_day] += 1 
  
  # sort by day
  tweet_days_positive_counts = sorted(list(tweet_days_positive_counts.items()),key=lambda x: x[0])
  tweet_days_negative_counts = sorted(list(tweet_days_negative_counts.items()),key=lambda x: x[0])

  # get convenient lists for plotting
  positive_counts = [cc[1] for cc in tweet_days_positive_counts]
  negative_counts = [cc[1] for cc in tweet_days_negative_counts]
  days = [dd[0] for dd in tweet_days_positive_counts]

  # plot, some light formatting
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.plot(days, positive_counts,label='Positive Tweets')
  ax.plot(days, negative_counts,label='Negative Tweets')
  fig.legend()
  fig.tight_layout(rect=(0.05,0.1,1,1))
  ax.set_xlabel('Days')
  ax.set_ylabel('Tweet Sentiment Counts')
  plt.xticks(rotation=30)
  plt.savefig('election_sentiment.png')
  plt.close()