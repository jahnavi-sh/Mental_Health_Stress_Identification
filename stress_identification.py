#Import libraries 

import numpy as np 
import pandas as pd 

#data visualisation 
import matplotlib.pyplot as plt
import seaborn as sns

#nlp 
import nltk
import re 
from urllib.parse import urlparse
from spacy import load
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import collections

#model training 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

#Model validation 
from sklearn.metrics import confusion_matrix,classification_report

#load dataset 
df = pd.read_csv(r'/kaggle/input/human-stress-prediction/Stress.csv')

#view the first 5 rows of the dataset 
df.head()

#view the total number of rows and columns 
df.shape 
#there are 2838 rows and 7 columns 

#statistical measures of the data 
df.info()
#there are no null values in the dataset 

#cross checking for null values 
df.isnull().sum()
#there are no null values in the data 

#most used words 
words = []
for text in df['clean_text']:
    words.extend(text.split())
word_count = collections.Counter(words)
top_words = dict(word_count.most_common(10))

plt.style.use('dark_background')                                           # Dark Background
plt.figure(figsize = (10, 6))                                              # Figure Size
plt.bar(range(len(top_words)), list(top_words.values()), align = 'center') # Create the Barplot
plt.xticks(range(len(top_words)), list(top_words.keys()))                  # Creating a y axis with words
plt.grid(alpha = 0.5)                                                      # Grid Opacity
plt.title('Top 10 most used words', fontsize = 18)                         # Grid Opacity
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

#word cloud 
text = ' '.join(caption for caption in df['clean_text'])
wordcloud = WordCloud(width = 800, height = 500, background_color = 'black', min_font_size = 10).generate(text) 
plt.figure(figsize = (10, 6), facecolor = None) 
plt.imshow(wordcloud)   
plt.axis('off') 
plt.show()

#dropping the columns not required 
not_required_columns = ['subreddit','post_id','sentence_range','confidence','social_timestamp']

#it is always a good practice to create a new pandas dataframe when you are dropping columns or editing the dataframe
#but, here for the sake of convenience, I will simply make edits to the original dataframe

df = df.drop(not_required_columns, axis=1)
df.sample(3)

#Therefore, now we have two columns - text and label 

#Natural language processing 
#Text processing 

#open multilingual wordnet, this is a lexical database 
nltk.download('omw-1.4')
nltk.download('wordnet') 
nltk.download('wordnet2022')
nltk.download('punkt')
nltk.download('stopwords')

#downloading stop words 
lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))
print(stop_words)

#regular expressions 
def textPocess(sent):
    try:
        # brackets replacing by space
        sent = re.sub('[][)(]',' ',sent)

        # url removing
        sent = [word for word in sent.split() if not urlparse(word).scheme]
        sent = ' '.join(sent)

        # removing escap characters
        sent = re.sub(r'\@\w+','',sent)

        # removing html tags 
        sent = re.sub(re.compile("<.*?>"),'',sent)

        # getting only characters and numbers from text
        sent = re.sub("[^A-Za-z0-9]",' ',sent)

        # lower case all words
        sent = sent.lower()
        
        # strip all words from sentences
        sent = [word.strip() for word in sent.split()]
        sent = ' '.join(sent)

        # word tokenization
        tokens = word_tokenize(sent)
        
        # removing words which are in stopwords
        for word in tokens:
            if word in stop_words:
                tokens.remove(word)
        
        # lemmatization
        sent = [lemmatizer.lemmatize(word) for word in tokens]
        sent = ' '.join(sent)
        return sent
    
    except Exception as ex:
        print(sent,"\n")
        print("Error ",ex)

#view processed text 
df['processed_text'] = df['text'].apply(lambda text: textPocess(text))
df.sample(3)

#vectorization using BOW (Bag of Words/TF-IDF vectorizer)

MIN_DF = 1 
cv = CountVectorizer(min_df=MIN_DF)
cv_df = cv.fit_transform(df['processed_text'])
cv_df.toarray()

cv_df = pd.DataFrame(cv_df.toarray(),columns=cv.get_feature_names_out())
cv_df.head()

tf = TfidfVectorizer(min_df=MIN_DF)
tf_df = tf.fit_transform(df['processed_text'])
tf_df.toarray()

tf_df = pd.DataFrame(tf_df.toarray(),columns=tf.get_feature_names_out())
tf_df.head()

tf_df.describe()

cv_df.shape

tf_df.shape

#Train model 

#for BOW - logistic regression, multinomialNB, RandomForestClassifier 
#for TF-IDF - logistic regression, multinomialNB, RandomForestClassifier 

#for BOW 
#train test split 
X_train,X_test,y_train,y_test = train_test_split(cv_df,df['label'],stratify=df['label'])
X_train.shape,y_test.shape

#logistic regression 
model_lr = LogisticRegression().fit(X_train,y_train)
model_lr.score(X_train,y_train),model_lr.score(X_test,y_test)

#naive bayes 
model_nb = MultinomialNB().fit(X_train,y_train)
model_nb.score(X_train,y_train),model_nb.score(X_test,y_test)

#random forest 
model_rf = RandomForestClassifier().fit(X_train,y_train)
model_rf.score(X_train,y_train),model_rf.score(X_test,y_test)

#for TF-IDF
#train test split 
X_train1,X_test1,y_train1,y_test1 = train_test_split(tf_df,df['label'],stratify=df['label'])
X_train1.shape,y_test1.shape

#logistic regression 
model_lr = LogisticRegression().fit(X_train1,y_train1)
model_lr.score(X_train1,y_train1),model_lr.score(X_test1,y_test1)

#naive bayes 
model_nb = MultinomialNB().fit(X_train1,y_train1)
model_nb.score(X_train1,y_train1),model_nb.score(X_test1,y_test1)

#random forest 
model_rf = RandomForestClassifier().fit(X_train1,y_train1)
model_rf.score(X_train1,y_train1),model_rf.score(X_test1,y_test1)

#Here, logistic regression and naive bayes show better results than random forest 

#model validation 
y_pred = model_lr.predict(X_test1)
cm = confusion_matrix(y_pred,y_test1)
cm

sns.heatmap(cm,annot=True,fmt='')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_pred,y_test1))

#Building a prediction system 
model = LogisticRegression().fit(tf_df,df['label'])
model.score(tf_df,df['label'])

def predictor(text):
    processed = textPocess(text)
    embedded_words = tf.transform([text])
    res = model.predict(embedded_words)
    if res[0] == 1:
        res = "this person is stressed"
    else:
        res = "this person is not stressed"
    return res

text1 = """This is the worst thing that happened to me today. I got less marks in my exam, 
            so it is not going to help me in my future."""
text2 = """Hi Shashank sir, I gained a lot of knowledge from you for my future use. 
            This was a very fun journey for me. Thanks for boosting my confidence."""

print(predictor(text1))
print(predictor(text2))