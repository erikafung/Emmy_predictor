# Emmy_predictor

# Description

This project was created for CS 686 Introduction to AI. It retrieves tweets about 2020 Emmy nominees for outstanding lead actress in a comedy series and performs sentiment analysis using the TextBlob, TextBlob Naive Bayes Analyzer, and SentiWordNet analyzers. Performance metrics for these classifiers are generated for comparison on a labelled training data set. The classifications and accuracy values on the training set are compared using a Chi-square test of independence. The TextBlob analyzer is used to rank the nominees based on four criteria. These four rankings are compared to an industry ranking using the Spearman ranking correlation coefficient. 

# Features

This program utilizes the Twint API to retrieve Twitter data. It also employs the TextBlob, TextBlob Naive Bayes Analyzer, and NLTK SentiWordNet sentiment analyzers. Sklearn metrics and SciPy stats are used to evaluate the performance of the sentiment classifiers.

# Required Libraries

Use the package manager [pip] to install twint, pandas, textblob, nltk, sklean, scipy

```bash
pip install twint
pip install pandas
pip install textblob
pip install nltk
pip install sklearn
pip install scipy
```
# Commands

Use the following command to generate a CSV with tweets for a nominee. Specify the name of the nominee in twint_tweet_retriever.py

```python
python twint_tweet_retriever.py
```
Use the following command to perform sentiment analysis on a labelled training data set of tweets and to generate rankings of the nominees based on four criteria. The name of the training data set is hard-coded as 'comedy_actress_weka.csv'. This program returns evaluation metrics and statistical tests for the sentiment analyzers on the training data set and rankings of the nominees based on four criteria. Spearman ranking correlation coefficient scores are also returned for each generated ranking, based on comparison with IndieWire's ranking for nominees in the outstanding lead actress in a comedy series category. The hard-coded IndieWire ranking can be replaced with another industry ranking when calculating the Spearman ranking correlation coefficient scores. 

```python
python textblob_swn_analysis.py
```

# Example Ouput

```python
python textblob_swn_analysis.py

Textblob accuracy: 0.616
Textblob NB accuracy: 0.492
SWN accuracy: 0.378
Textblob confusion matrix:
[[[ 98  89]
  [ 79 234]]

 [[277  66]
  [ 91  66]]

 [[433  37]
  [ 22   8]]]
Textblob NB confusion matrix:
[[[ 57 130]
  [ 77 236]]

 [[342   1]
  [157   0]]

 [[347 123]
  [ 20  10]]]
SWN confusion matrix:
[[[169  18]
  [266  47]]

 [[ 66 277]
  [ 19 138]]

 [[454  16]
  [ 26   4]]]
Textblob performance:
              precision    recall  f1-score   support

    negative       0.18      0.27      0.21        30
     neutral       0.50      0.42      0.46       157
    positive       0.72      0.75      0.74       313

    accuracy                           0.62       500
   macro avg       0.47      0.48      0.47       500
weighted avg       0.62      0.62      0.62       500

Textblob NB performance:
              precision    recall  f1-score   support

    negative       0.08      0.33      0.12        30
     neutral       0.00      0.00      0.00       157
    positive       0.64      0.75      0.70       313

    accuracy                           0.49       500
   macro avg       0.24      0.36      0.27       500
weighted avg       0.41      0.49      0.44       500

SWN performance:
              precision    recall  f1-score   support

    negative       0.20      0.13      0.16        30
     neutral       0.33      0.88      0.48       157
    positive       0.72      0.15      0.25       313

    accuracy                           0.38       500
   macro avg       0.42      0.39      0.30       500
weighted avg       0.57      0.38      0.32       500

TextBlob and TextBlob Naive Bayes accuracy p-value:
0.21271304421764312
TextBlob and SentiWordNet accuracy p-value:
0.2811899206424281
SentiWordNet and TextBlob Naive Bayes accuracy p-value:
0.20633157533818944
Collective accuracy p-value:
0.944894241990287
TextBlob and TextBlob Naive Bayes p-value:
8.945694313996274e-39
TextBlob and SentiWordNet p-value:
7.33449413132621e-72
SentiWordNet and TextBlob Naive Bayes p-value:
5.806453743639465e-154
Collective p-value:
1.5072107277374165e-173
Indicator 1
[('Issa', 0.19897032101756512), ('Catherine', 0.09630526953361598), ('Linda', 0.05784373107207753), ('Tracee', 0.043609933373712904), ('Christina', 0.04239854633555421), ('Rachel', 0.015748031496062992)]
Indicator 2
[('Issa', 0.4374167776298269), ('Catherine', 0.21171770972037285), ('Linda', 0.12716378162450068), ('Tracee', 0.09587217043941411), ('Christina', 0.09320905459387484), ('Rachel', 0.03462050599201065)]
Indicator 3
[('Issa', 0.4600242277407632), ('Catherine', 0.1599030890369473), ('Linda', 0.13476680799515445), ('Christina', 0.09721380981223501), ('Tracee', 0.09630526953361598), ('Rachel', 0.05178679588128407)]
Indicator 4
[('Catherine', 0.6022727272727273), ('Tracee', 0.4528301886792453), ('Christina', 0.43613707165109034), ('Issa', 0.4325213956550362), ('Linda', 0.42921348314606744), ('Rachel', 0.30409356725146197)]
Spearman Indicator 1
-0.2 0.704
Spearman Indicator 2
-0.2 0.704
Spearman Indicator 3
0.7142857142857143 0.1107871720116617
Spearman Indicator 4
0.3142857142857143 0.5440932944606414
```
