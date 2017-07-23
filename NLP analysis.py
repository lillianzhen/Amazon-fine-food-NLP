# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from pandas import Series, DataFrame, datetime
from subprocess import check_output
from datetime import datetime, timedelta, time

%matplotlib inline

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

# Any results you write to the current directory are saved as output.
data = pd.read_csv('Reviews.csv')
data.head()


PU=data.groupby(['ProductId','UserId'])['Id'].count()
PU.sort_values(ascending=False)
##here we can see that a lot of the records are duplicate.try to check on the text.
data[(data['ProductId'] == 'B00008CQVA') & (data['UserId'] == 'A29JUMRL1US6YP')][['Text','Score']]
##there are duplicated records, need to clear it
data_clean=data.drop(['ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator','Summary'], 1)
data_clean.head()
data_clean.groupby(['ProductId','UserId']).head()
len(data_clean)


data_clean = data_clean.drop_duplicates()
len(data_clean)
PU=data_clean.groupby(['ProductId','UserId'])['Id'].count()
PU.sort_values(ascending=False)
PU
data_clean = data_clean[0:35172]
##the data is now cleaned and ready for use


data_clean['datetime'] = pd.to_datetime(data_clean["Time"], unit='s')
df_grp = data_clean.groupby([data_clean.datetime.dt.year, data_clean.datetime.dt.month, data_clean.Score]).count()['ProductId'].unstack().fillna(0)
df_grp.plot(figsize=(20,10), rot=45, colormap='jet')

##add sentiment based on score

def partition(x):
    if x < 3:
        return 'negative'
    elif x == 3:
        return 'neutral'
    return 'positive'

Score = data_clean['Score']
Score = Score.map(partition)

data_clean['Review'] = data_clean['Score'].map(partition)

pos = data_clean.loc[data_clean['Review'] == 'positive']
pos = pos[0:25000]

neu = data_clean.loc[data_clean['Review'] == 'neutral']
neu = pos[0:25000]

neg = data_clean.loc[data_clean['Review'] == 'negative']
neg = neg[0:25000]

df=[]
for i in pos.index:
    df.append(pos.loc[i]['Text'].lower())
pos['Text'] = df

df=[]
for i in neu.index:
    df.append(neu.loc[i]['Text'].lower())
neu['Text'] = df

df=[]
for i in neg.index:
    df.append(neg.loc[i]['Text'].lower())
neg['Text'] = df

df=[]
for i in data_clean.index:
    df.append(data_clean.loc[i]['Text'].lower())
data_clean['Text'] = df

from wordcloud import WordCloud
from  wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
exclude = set(string.punctuation)
stopwords.union(exclude)

def show_wordcloud(data,title=None):
    wordcloud=WordCloud(
               background_color='white',
               stopwords=stopwords,
               max_words=300,
               max_font_size=40
    ).generate(str(data))  
    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()


show_wordcloud(pos['Text'])
show_wordcloud(neu['Text'])
show_wordcloud(neg['Text'])

len(data_clean[data_clean['Review'] == 'neutral'])

data_clean=data_clean[data_clean['Review'] != 'neutral']

Score = data_clean['Score']
Score = Score.map(partition)
Summary = data_clean['Text']
X_train, X_test, y_train, y_test = train_test_split(Summary, Score, test_size=0.1, random_state=42)

stemmer = PorterStemmer()
from nltk.corpus import stopwords
from nltk import *

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text, language='english', preserve_line=False)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stems = stem_tokens(tokens, stemmer)
    return ' '.join(stems)
    
intab = string.punctuation
outtab = "                                "
trantab = str.maketrans(intab, outtab)

corpus = []
for text in X_train:
    text = text.lower()
    text = text.translate(trantab)
    #text = tokenize(text)
    corpus.append(text)
        
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)        
        
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#--- Test set

test_set = []
for text in X_test: 
    text = text.lower()
    text = text.translate(trantab)
    #text = tokenize(text)
    test_set.append(text)

X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

from pandas import *
df = DataFrame({'Before': X_train, 'After': corpus})
print(df.head(20))

prediction = dict()

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, y_train)
prediction['Multinomial'] = model.predict(X_test_tfidf)

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train_tfidf, y_train)
prediction['Bernoulli'] = model.predict(X_test_tfidf)

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_tfidf, y_train)
prediction['Logistic'] = logreg.predict(X_test_tfidf)

def formatt(x):
    if x == 'negative':
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction['Logistic'], target_names = ["positive",  "negative"]))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(Score)))
    plt.xticks(tick_marks, set(Score), rotation=45)
    plt.yticks(tick_marks, set(Score))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
cm = confusion_matrix(y_test, prediction['Logistic'])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm)    

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
