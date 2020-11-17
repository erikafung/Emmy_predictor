import pandas as pd
import re
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import csv
import nltk
# download sentiwordnet once at start
nltk.download('sentiwordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from scipy import stats

lemmatizer = WordNetLemmatizer()

# clean tweet of unnecessary data
# function adapted from https://github.com/acgeospatial/EO_tweets/blob/master/pandas_twitter.ipynb
def process_tweet(tweet):
    tweet = tweet.lower()
    # convert url to white space
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    # convert @Username to white space
    tweet = re.sub('@[^\s]+',' ',tweet)
    # convert pictures to white spaces
    tweet = re.sub('((pic.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    # remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub('[\n]+', ' ', tweet)
    # remove not alphanumeric symbols
    tweet = re.sub(r'[^\w]', ' ', tweet)
    # remove hastags from words
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # remove emoticons
    tweet = tweet.replace(':)','')
    tweet = tweet.replace(':(','')
    # remove stopwords using NLTK
    tweet = tweet.strip('\'"')
    stop_words = stopwords.words('english')
    # add words in the titles of shows with sentiment value to stopwords
    new_stop_words = ["schitt", "insecure", "dead", "marvelous", "black"]
    stop_words.extend(new_stop_words)
    tweet_stop = word_tokenize(tweet)
    tweet_filtered = []
    for word in tweet_stop:
        if word not in stop_words:
            tweet_filtered.append(word)
    clean_tweet = (" ").join(tweet_filtered)
    return clean_tweet

# remove non-ascii characters from tweet
def remove_non_ascii(tweet):
    return ''.join(i for i in tweet if ord(i)<128)

# convert sentiment score into sentiment category
# optimized threshold
def get_sentiment(score):
    if score == 0:
        return 'neutral'
    elif score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'

# convert wordnet tag to sentiwordnet tag
# function adapted from https://github.com/SivaAndMe/Coarse-grained-Sentiment-Analysis-on-Swachh-Bharat-using-Tweets/blob/master/swn_sentiment_labeling.py
def convert_tag(tag):
    if tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('R'):
        return wn.ADV
    return None

# perform sentiwordnet analysis on tweets
# function adapted from https://nlpforhackers.io/sentiment-analysis-intro/
def swn_process_tweet(tweet):
    tokens = word_tokenize(tweet)
    # tag words with noun, adj, adv, or verb
    tagged_words = pos_tag(tokens)
    score = 0.0

    for word, tag in tagged_words:
        # convert tag to sentiwordnet
        swn_tag = convert_tag(tag)
        if swn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            return 0.0

        lemma = lemmatizer.lemmatize(word, pos=swn_tag)
        if not lemma:
            return 0.0

        synsets = wn.synsets(lemma, pos=swn_tag)
        if not synsets:
            return 0.0

        # use the first, most common sense
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        # calculate the score
        score += swn_synset.pos_score() - swn_synset.neg_score()

        return score

# get polarity scores and sentiments for tweets based on TextBlob, TextBlob Naive Bayes Analyzer, and sentiwordnet
def process_tweets(tweets):
    tweet_df = pd.read_csv(tweets)
    # make tweets strings
    tweet_df['tweet'] = tweet_df['tweet'].astype(str)
    # clean tweets
    tweet_df['tweet'] = tweet_df['tweet'].apply(remove_non_ascii)
    tweet_df['clean_tweet'] = list(map(process_tweet, tweet_df['tweet']))
    # calculate textblob sentiment scores
    tweet_df['sentiment_score_textblob'] = tweet_df['clean_tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # convert textblob scores to sentiment categories
    tweet_df['sentiment_textblob'] = tweet_df['sentiment_score_textblob'].apply(lambda x: get_sentiment(x))
    # calculate textblob Naive Bayes Analyzer scores
    tweet_df['train_sentiment_score_textblob_NB'] = tweet_df['clean_tweet'].apply(lambda x: tb(x).sentiment.p_pos - tb(x).sentiment.p_neg)
    # convert textblob NB scores to sentiment categories
    tweet_df['train_sentiment_textblob_NB'] = tweet_df['train_sentiment_score_textblob_NB'].apply(lambda x: get_sentiment(x))
    # calculate sentiwordnet sentiment scores
    tweet_df['sentiment_score_swn'] = list(map(swn_process_tweet, tweet_df['clean_tweet']))
    # convert sentiwordnet scores to sentiment categories
    tweet_df['sentiment_swn']=tweet_df['sentiment_score_swn'].apply(lambda x: get_sentiment(x))
    return tweet_df

# perform sentiment analysis on training data set using TextBlob, TextBlob NaiveBayesAnalyzer, and sentiwordnet
def train_tweets(tweets):
    tweet_df = pd.read_csv(tweets)
    # make tweets strings
    tweet_df['clean_tweet'] = tweet_df['clean_tweet'].astype(str)
    # calculate textblob sentiment scores
    tweet_df['train_sentiment_score_textblob'] = tweet_df['clean_tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # convert textblob scores to sentiment categories
    tweet_df['train_sentiment_textblob'] = tweet_df['train_sentiment_score_textblob'].apply(lambda x: get_sentiment(x))
    # calculate textblob Naive Bayes Analyzer scores
    tweet_df['train_sentiment_score_textblob_NB'] = tweet_df['clean_tweet'].apply(lambda x: tb(x).sentiment.p_pos - tb(x).sentiment.p_neg)
    # convert textblob NB scores to sentiment categories
    tweet_df['train_sentiment_textblob_NB'] = tweet_df['train_sentiment_score_textblob_NB'].apply(lambda x: get_sentiment(x))
    # calculate sentiwordnet sentiment scores
    tweet_df['train_sentiment_score_swn'] = list(map(swn_process_tweet, tweet_df['clean_tweet']))
    # convert sentiwordnet scores to sentiment categories
    tweet_df['train_sentiment_swn']=tweet_df['train_sentiment_score_swn'].apply(lambda x: get_sentiment(x))
    # get sentiments
    sentiment = tweet_df.sentiment
    train_sentiment_textblob = tweet_df.train_sentiment_textblob
    train_sentiment_textblob_NB = tweet_df.train_sentiment_textblob_NB
    train_sentiment_swn = tweet_df.train_sentiment_swn
    # calculate accuracies for each sentiment analyzer
    textblob_acc = metrics.accuracy_score(sentiment, train_sentiment_textblob)
    textblob_NB_acc = metrics.accuracy_score(sentiment, train_sentiment_textblob_NB)
    swn_acc = metrics.accuracy_score(sentiment, train_sentiment_swn)
    # calculate confusion matrices
    textblob_confusion = metrics.multilabel_confusion_matrix(sentiment, train_sentiment_textblob, labels=["positive", "neutral", "negative"])
    textblob_NB_confusion = metrics.multilabel_confusion_matrix(sentiment, train_sentiment_textblob_NB, labels=["positive", "neutral", "negative"])
    swn_confusion = metrics.multilabel_confusion_matrix(sentiment, train_sentiment_swn, labels=["positive", "neutral", "negative"])
    # get classification report with precision, recall, f-measure
    textblob_performance = metrics.classification_report(sentiment, train_sentiment_textblob)
    textblob_NB_performance = metrics.classification_report(sentiment, train_sentiment_textblob_NB)
    swn_performance = metrics.classification_report(sentiment, train_sentiment_swn)
    return tweet_df, textblob_acc, textblob_NB_acc, swn_acc, textblob_confusion, textblob_NB_confusion, swn_confusion, textblob_performance, textblob_NB_performance, swn_performance

# get labelled training set tweets about nominees for lead actress in a comedy series
tweets = 'comedy_actress_weka.csv'
tb = Blobber(analyzer=NaiveBayesAnalyzer())
# perform textblob and swn analysis on training data sets
tweet_data_train, textblob_acc, textblob_NB_acc, swn_acc, textblob_confusion, textblob_NB_confusion, swn_confusion, textblob_performance, textblob_NB_performance, swn_performance = train_tweets(tweets)
tweet_data_train.to_csv("textblob_swn_train.csv")
# print performance metrics on training data
print("Textblob accuracy: " + str(textblob_acc))
print("Textblob NB accuracy: " + str(textblob_NB_acc))
print("SWN accuracy: " + str(swn_acc))
print("Textblob confusion matrix: ")
print(textblob_confusion)
print("Textblob NB confusion matrix: ")
print(textblob_NB_confusion)
print("SWN confusion matrix: ")
print(swn_confusion)
print("Textblob performance: ")
print(textblob_performance)
print("Textblob NB performance: ")
print(textblob_NB_performance)
print("SWN performance: ")
print(swn_performance)

# calculate p-values for accuracy on training data set
# textblob and textblob naive bayes
tb_tbnb = [[textblob_acc, textblob_NB_acc], [(1-textblob_acc), (1-textblob_NB_acc)]]
chi2_tbtbnb, pval_tbtbnb, dof_tbtbnb, expected_tbtbnb = stats.chi2_contingency(tb_tbnb)
print("TextBlob and TextBlob Naive Bayes accuracy p-value: ")
print(pval_tbtbnb)
# textblob and sentiwordnet
tb_swn = [[textblob_acc, swn_acc], [(1-textblob_acc), (1-swn_acc)]]
chi2_tbswn, pval_tbswn, dof_tbswn, expected_tbswn = stats.chi2_contingency(tb_swn)
print("TextBlob and SentiWordNet accuracy p-value: ")
print(pval_tbswn)
# textblob naive bayes and sentiwordnet
swn_tbnb = [[textblob_NB_acc, swn_acc], [(1-textblob_NB_acc), (1-swn_acc)]]
chi2_swntbnb, pval_swntbnb, dof_swntbnb, expected_swntbnb = stats.chi2_contingency(swn_tbnb)
print("SentiWordNet and TextBlob Naive Bayes accuracy p-value: ")
print(pval_swntbnb)
# collective
collective = [[textblob_acc, textblob_NB_acc, swn_acc], [(1-textblob_acc), (1-textblob_NB_acc), (1-swn_acc)]]
chi2_collective, pval_collective, dof_collective, expected_collective = stats.chi2_contingency(collective)
print("Collective accuracy p-value: ")
print(pval_collective)
# calculate p-values for classifications on training data set
# number of instances classified as positive, negative, and neutral by models are hard-coded from csv results
# textblob and textblob naive bayes
#tb_tbnb = [[323, 132, 45],[366, 1, 133]]
tb_tbnb = [[323, 366],[132, 1], [45, 133]]
chi2_tbtbnb, pval_tbtbnb, dof_tbtbnb, expected_tbtbnb = stats.chi2_contingency(tb_tbnb)
print("TextBlob and TextBlob Naive Bayes p-value: ")
print(pval_tbtbnb)
# textblob and sentiwordnet
#tb_swn = [[323, 132, 45],[65, 415, 20]]
tb_swn = [[323, 65],[132,415],[45,20]]
chi2_tbswn, pval_tbswn, dof_tbswn, expected_tbswn = stats.chi2_contingency(tb_swn)
print("TextBlob and SentiWordNet p-value: ")
print(pval_tbswn)
# textblob naive bayes and sentiwordnet
#swn_tbnb = [[65, 415, 20],[366, 1, 133]]
swn_tbnb = [[65, 366], [415, 1], [20, 133]]
chi2_swntbnb, pval_swntbnb, dof_swntbnb, expected_swntbnb = stats.chi2_contingency(swn_tbnb)
print("SentiWordNet and TextBlob Naive Bayes p-value: ")
print(pval_swntbnb)
# collective
# collective = [[65, 415, 20],[366, 1, 133],[323, 132, 45]]
collective = [[323, 366, 65],[132,1,415],[45,133,20]]
chi2_collective, pval_collective, dof_collective, expected_collective = stats.chi2_contingency(collective)
print("Collective p-value: ")
print(pval_collective)

# perform textblob, textblob naive bayes, and swn analysis on comedy actress data sets
tweets = 'Catherine_consolidated_tweets.csv'
Catherine_tweet_data = process_tweets(tweets)
Catherine_positive = Catherine_tweet_data.loc[Catherine_tweet_data["sentiment_textblob"] == "positive"].shape[0]
Catherine_total = Catherine_tweet_data.shape[0]
Catherine_tweet_data.to_csv("Catherine_out.csv")

tweets = 'Christina_consolidated_tweets.csv'
Christina_tweet_data = process_tweets(tweets)
Christina_positive = Christina_tweet_data.loc[Christina_tweet_data["sentiment_textblob"] == "positive"].shape[0]
Christina_total = Christina_tweet_data.shape[0]
Christina_tweet_data.to_csv("Christina_out.csv")

tweets = 'Issa_consolidated_tweets.csv'
Issa_tweet_data = process_tweets(tweets)
Issa_positive = Issa_tweet_data.loc[Issa_tweet_data["sentiment_textblob"] == "positive"].shape[0]
Issa_total = Issa_tweet_data.shape[0]
Issa_tweet_data.to_csv("Issa_out.csv")

tweets = 'Linda_consolidated_tweets.csv'
Linda_tweet_data = process_tweets(tweets)
Linda_positive = Linda_tweet_data.loc[Linda_tweet_data["sentiment_textblob"] == "positive"].shape[0]
Linda_total = Linda_tweet_data.shape[0]
Linda_tweet_data.to_csv("Linda_out.csv")

tweets = 'Rachel_consolidated_tweets.csv'
Rachel_tweet_data = process_tweets(tweets)
Rachel_positive = Rachel_tweet_data.loc[Rachel_tweet_data["sentiment_textblob"] == "positive"].shape[0]
Rachel_total = Rachel_tweet_data.shape[0]
Rachel_tweet_data.to_csv("Rachel_out.csv")

tweets = 'Tracee_consolidated_tweets.csv'
Tracee_tweet_data = process_tweets(tweets)
Tracee_positive = Tracee_tweet_data.loc[Tracee_tweet_data["sentiment_textblob"] == "positive"].shape[0]
Tracee_total = Tracee_tweet_data.shape[0]
Tracee_tweet_data.to_csv("Tracee_out.csv")

# calculate rankings from 4 indicators
total_positive = Catherine_positive + Christina_positive + Issa_positive + Linda_positive + Rachel_positive + Tracee_positive
total_tweets = Catherine_total + Christina_total + Issa_total + Linda_total + Rachel_total + Tracee_total
Ind1 = []
Ind2 = []
Ind3 = []
Ind4 = []

Catherine_ind1 = Catherine_positive / total_tweets
Ind1.append(("Catherine", Catherine_ind1))
Catherine_ind2 = Catherine_positive / total_positive
Ind2.append(("Catherine", Catherine_ind2))
Catherine_ind3 = Catherine_total / total_tweets
Ind3.append(("Catherine", Catherine_ind3))
Catherine_ind4 = Catherine_positive / Catherine_total
Ind4.append(("Catherine", Catherine_ind4))

Christina_ind1 = Christina_positive / total_tweets
Ind1.append(("Christina", Christina_ind1))
Christina_ind2 = Christina_positive / total_positive
Ind2.append(("Christina", Christina_ind2))
Christina_ind3 = Christina_total / total_tweets
Ind3.append(("Christina", Christina_ind3))
Christina_ind4 = Christina_positive / Christina_total
Ind4.append(("Christina", Christina_ind4))

Issa_ind1 = Issa_positive / total_tweets
Ind1.append(("Issa", Issa_ind1))
Issa_ind2 = Issa_positive / total_positive
Ind2.append(("Issa", Issa_ind2))
Issa_ind3 = Issa_total / total_tweets
Ind3.append(("Issa", Issa_ind3))
Issa_ind4 = Issa_positive / Issa_total
Ind4.append(("Issa", Issa_ind4))

Linda_ind1 = Linda_positive / total_tweets
Ind1.append(("Linda", Linda_ind1))
Linda_ind2 = Linda_positive / total_positive
Ind2.append(("Linda", Linda_ind2))
Linda_ind3 = Linda_total / total_tweets
Ind3.append(("Linda", Linda_ind3))
Linda_ind4 = Linda_positive / Linda_total
Ind4.append(("Linda", Linda_ind4))

Rachel_ind1 = Rachel_positive / total_tweets
Ind1.append(("Rachel", Rachel_ind1))
Rachel_ind2 = Rachel_positive / total_positive
Ind2.append(("Rachel", Rachel_ind2))
Rachel_ind3 = Rachel_total / total_tweets
Ind3.append(("Rachel", Rachel_ind3))
Rachel_ind4 = Rachel_positive / Rachel_total
Ind4.append(("Rachel", Rachel_ind4))


Tracee_ind1 = Tracee_positive / total_tweets
Ind1.append(("Tracee", Tracee_ind1))
Tracee_ind2 = Tracee_positive / total_positive
Ind2.append(("Tracee", Tracee_ind2))
Tracee_ind3 = Tracee_total / total_tweets
Ind3.append(("Tracee", Tracee_ind3))
Tracee_ind4 = Tracee_positive / Tracee_total
Ind4.append(("Tracee", Tracee_ind4))

# sort indicators from highest to lowest
Ind1.sort(key=lambda x: x[1], reverse = True)
print("Indicator 1")
print(Ind1)
Ind2.sort(key=lambda x: x[1], reverse = True)
print("Indicator 2")
print(Ind2)
Ind3.sort(key=lambda x: x[1], reverse = True)
print("Indicator 3")
print(Ind3)
Ind4.sort(key=lambda x: x[1], reverse = True)
print("Indicator 4")
print(Ind4)

# calculate spearman ranking correlation coefficient
# compare indicators 1 - 4 with IndieWire ranking
# 1 = Catherine, 2 = Christina, 3 = Issa, 4 = Linda, 5 = Rachel, 6 = Tracee
# IndieWire ranking
indiewire = [1,3,5,2,6,4]
# rankings hard coded from indicator outputs
spear_ind1, p_ind1 = stats.spearmanr([3,1,4,6,2,5], indiewire)
print("Spearman Indicator 1")
print(spear_ind1, p_ind1)
spear_ind2, p_ind2 = stats.spearmanr([3,1,4,6,2,5], indiewire)
print("Spearman Indicator 2")
print(spear_ind2, p_ind2)
spear_ind3, p_ind3 = stats.spearmanr([3,1,4,2,6,5], indiewire)
print("Spearman Indicator 3")
print(spear_ind3, p_ind3)
spear_ind4, p_ind4 = stats.spearmanr([1,6,2,3,4,5], indiewire)
print("Spearman Indicator 4")
print(spear_ind4, p_ind4)
