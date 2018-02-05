
# coding: utf-8

# In[5]:


#Importing libraries

import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
from gensim import corpora, models
import string
import re
import spacy
import time
import nltk
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[6]:


#Loading data

data= pd.read_csv("./data/posts.csv")


# In[7]:


data.info()


# Find & count the named entities in the posts

# In[43]:


# Converting the text to unicode

def make_unicode(input):
    if type(input) != unicode:
        input =  input.decode('utf-8')
        return input
    else:
        return input

data['raw_body_text']= data['raw_body_text'].apply(lambda text: make_unicode(text))


# In[44]:


# Removing Emojis and Punctuations (Keeping # and $) 

def remove_emoji(input):
    return re.sub(r'[^A-Za-z0-9$#.]+', ' ', input)

data['raw_body_text']= data['raw_body_text'].apply(lambda text: remove_emoji(text))


# In[45]:


#Loading english vocabulary for finding named entity

nlp= spacy.load('en')
tags={} # <- for storing tags in the form of key value pair


# In[21]:


#Function to find named entities using spaCy package 

def get_entites(text):
    doc= nlp(text)
    for ent in doc.ents:
        tags[ent.text]= ent.label_


# In[22]:


start_time= time.time()
get_ipython().magic(u"time data['raw_body_text'].apply(lambda text: get_entites(text))")
print("--- %s seconds ---" % (time.time() - start_time))


# In[33]:


tags


# In[29]:


df_tags= pd.DataFrame(columns=["name","entity_type"])

start_time= time.time()
for key, value in tags.iteritems():
    df_tags.loc[len(df_tags)]= [key,value]
print("--- %s seconds ---" % (time.time() - start_time))   


# Tried to add the name and the entites directly into the data frame. The operation ran for 6 hours and ultimately the kernel crashed. 

# In[30]:


df_tags.head()


# In[37]:


for entity in df_tags["entity_type"].unique():
    print("------The number of the counts of the entity %s is-------" %entity)
    print(len(df_tags[df_tags["entity_type"]== entity]))


# In[38]:


df_tags[df_tags['entity_type']== "LANGUAGE"]


# There are various packages in the pyhton for named entity recognition. Mainly two packages can be usedwo create tags 1.spaCY 2.NlTK. I my opinion, spaCy is better than the NLTK. I performed the operation using both the packages. First, spacy is faster than the ntlk. Second, NLTK tokenizes the sentence into individual text and thus the word losses its essence. For eg: For the sentecnce "San Fransico is a great city". Due to tokenization, NLTK recognises its has person.

# ## Creating Word Embedding to find any relations between the words

# In[46]:


#Preprocessing the text

from nltk.corpus import stopwords
lemma = nltk.stem.WordNetLemmatizer()

def preprocessing(text):
    #1. Tokenization
    text= nltk.tokenize.casual_tokenize(text)
    
    #2. Convert text to lowercase
    text= [token.lower() for token in text]
    
    #3.Removing Blanks
    text= [token.strip() for token in text]
    
    #4.Removing Stopwords
    text= [token for token in text if token not in stopwords.words('english')]
    
    #5. Removing Punctuations (in this case only .)
    text= [token for token in text if token not in string.punctuation]
    
    return text


# In[47]:


# Implenting preprocessing to the raw_body_text and store it as cleaned text
get_ipython().magic(u"time data['cleaned_text']= data['raw_body_text'].apply(lambda text: preprocessing(text))")


# In[48]:


# Creating corpora 
sentences= []
for text in data['cleaned_text']:
    sentences.append(text)


# In[49]:


sentences


# In[80]:


import multiprocessing


# In[81]:


# Setting hyperparameters for Word2Vec model

min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1


# In[82]:


# Creating Word2Vec model
vec_model = models.word2vec.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[83]:


vec_model.build_vocab(sentences)


# In[38]:


print("Word2Vec vocabulary length:", len(vec_model.wv.vocab))


# In[42]:


#Training sentences on the model build

get_ipython().magic(u'time vec_model.train(sentences, total_examples=vec_model.corpus_count, epochs=vec_model.iter)')


# In[50]:


#  Compressing the vectors into 2D for visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = vec_model.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


# In[52]:


vec_points= pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[vec_model.wv.vocab[word].index])
            for word in vec_model.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


# In[54]:


vec_points.head(15) # <--- To check whther the data is stored in the desired form or not


# In[55]:


vec_points.info()


# In[57]:


sns.set_context("poster")
vec_points.plot.scatter("x", "y", s=10, figsize=(20, 12))


# Word vectors show interesting information about the data that are present in the raw_body_text. Words are forming cluster together. One of the main reasons can be due to the presence of the different languages in the text. Second some words are use for a particular context when compared to the other. For better visualization, so that we can zoom in the particualr section of the graph to see the detailings much better. 

# In[63]:


# Creating scatter plot using iplot

# For Notebooks
init_notebook_mode(connected=True)

# For offline use
cf.go_offline()


# Please uncomment and run the below cell to create the graph. I have commented it out beacuse it was slowing down the notebook.

# In[67]:


# vec_points.iplot(kind='scatter',x='x',y='y',text='word',mode='markers',size=5)


# In[68]:


# Word closest to given word

vec_model.most_similar('jamie')


# In[69]:


vec_model.most_similar('american')


# Above shows the words that have been used in the context with jamie and american. From similarity score, it is evident that word american is always used with express. It also picked up the axp and #axp which represents american express stock name. 

# ## Clustering the data

# In[50]:


#Converting the text into the numbers so that it can be fed into machine laerning algorithms.

from sklearn.feature_extraction.text import TfidfVectorizer


# In[51]:


tfidf_vectorizer= TfidfVectorizer(use_idf=True, ngram_range=(1,2))


# In[52]:


get_ipython().magic(u"time tfidf_matrix= tfidf_vectorizer.fit_transform(data['raw_body_text'])")


# In[53]:


tfidf_matrix.get_shape


# In[54]:


feature_names= tfidf_vectorizer.get_feature_names()


# In[81]:


# Function for creating multiple clusters

from sklearn.cluster import KMeans

def create_clusters(number_of_clusters):
    km= KMeans(number_of_clusters)
    km.fit(tfidf_matrix)
    return km.labels_.tolist()


# Creating three differnet clusters to get the optimum number of cluster size. Elbow Method can be implemented to achieve more accurate number of clusters but it is time consuming.

# In[84]:


get_ipython().magic(u'time cluster_1= create_clusters(5)')


# In[85]:


get_ipython().magic(u'time cluster_2= create_clusters(7)')


# In[86]:


get_ipython().magic(u'time cluster_3= create_clusters(10)')


# In[87]:


data['cluster_id_1']= cluster_1
data['cluster_id_2']= cluster_2
data['cluster_id_3']= cluster_3


# In[89]:


print(data["cluster_id_1"].value_counts())
print(data["cluster_id_2"].value_counts())
print(data["cluster_id_3"].value_counts())


# In[99]:


# Printing top 10 text of the different clusters if num of clusters is 10
print("Top raw data per cluster:")
for i in range(10):
    print("Cluster %d data:" % i)
    print(data['raw_body_text'][data['cluster_id_3']==i].head(10))


# The above cell shows the various textual data that is present in the various clusters. 
# Cluster 0 comprises of the data which is in spanish language.
# Cluster 1 comprises of the data which is talking about some car insuarance.
# Cluster 2 comprises of the data which contains MeganRockefeller #reginageorge #marissacoope.. it is most 
# probably the instagram data of the coresponding hashtags.
# Cluster 3 contains of the data regarding the centurion lounge that is being open by american express at JFK.
# Cluster 4 contains the data regarding the hashtag #BlueMonday
# Cluster 5 comprises of the data which is in german language
# Cluster 7 comprises of the text A Trani si attendeva la sentenz.....
# Cluster 8 comprises of the text which is in italian language
# Cluster 9.. which is largest of all.. is regarding the american express card .. various promotions offered by the 
# amrican express and the feedback regarding the card by various users.

# In[100]:


# Printing top 10 text of the different clusters if num of clusters is 5
print("Top raw data per cluster:")
for i in range(5):
    print("Cluster %d data:" % i)
    print(data['raw_body_text'][data['cluster_id_1']==i].head(10))


# Trying to find some common topics within the the raw_text_data using latent dirichlet allocation so that it can be used majorly for the classification and clustering

# In[102]:


get_ipython().magic(u'time dictionary= corpora.Dictionary(sentences)')


# In[103]:


get_ipython().magic(u'time doc_term_matrix = [dictionary.doc2bow(text) for text in sentences]')


# In[104]:


get_ipython().magic(u'time ldamodel1 = models.ldamodel.LdaModel(doc_term_matrix, num_topics=5, id2word = dictionary, passes=5)')


# In[105]:


get_ipython().magic(u'time ldamodel2 = models.ldamodel.LdaModel(doc_term_matrix, num_topics=10, id2word = dictionary, passes=5)')


# In[111]:


for topic in ldamodel2.show_topics(num_topics=10, formatted=False, num_words=20):
    print("Topic {}: Words: ".format(topic[0]))
    topicwords = [w for (w, val) in topic[1]]
    print(topicwords)


# In[135]:


ldamodel1_transformed= ldamodel1[doc_term_matrix]
print(ldamodel1_transformed)


# In[114]:


# Using sklearn to find the topic words

from sklearn.decomposition import LatentDirichletAllocation
lda_sklearn= LatentDirichletAllocation(n_components=10, max_iter=5, random_state=1)


# In[116]:


get_ipython().magic(u'time lda_sklearn.fit(tfidf_matrix)')


# In[117]:


for topic_idx, topic in enumerate(lda_sklearn.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-10 - 1:-1]]))
        print()


# Constructed two LDA models using gensim and sklearn.. Gensim performed way better than the sklearn. Sklearn picked up the common stopwords like 'from, 'to' ..which is not the case of gensim. And since gensim uses word2vec (the pretained neural network model) to build the model.. it takes half the time than sklearn.

# In[122]:


get_ipython().magic(u'time lda_sklearn_transform= lda_sklearn.transform(tfidf_matrix)')


# In[123]:


def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print documents[doc_index]


# In[127]:


display_topics(lda_sklearn.components_,lda_sklearn_transform,feature_names,data['raw_body_text'], 5, 5)


# Considering the topic words spitted by the LDA model.. It is showing almost the similar information given by the 
# clusters earlier formed. The granuality of the topics are limited due to use of the different languages and the plethora of topics. 

# ##  Additional Insights regarding the data

# In[56]:


data['length_of_text']= data['raw_body_text'].apply(lambda text: len(text))


# In[146]:


data.hist(column= 'length_of_text', by= 'sentiment_category', bins=50,figsize=(12,4), range=(0,10000))


# In[150]:


for category in data['sentiment_category'].unique():
    print("--- Average length of %s  messages  ---" %category)
    print(np.mean(data['length_of_text'][data['sentiment_category']== category]))


# In[57]:


for network in data["network"].unique():
    print("-------Average length of the data for the %s is------"%network)
    print(np.mean(data['length_of_text'][data['network']== network]))


# # Classification of the data

# Trying to classify the "raw_body_text" using sklearn package. Since there is no target variable.. I am trying to classify data on the basis of the network. If the target column changes, same script can be re run with different column . Since the network is classification is sophisticated, higher degree of the accuracy can be expected..  

# In[58]:


# Importing all the neccessary libraries

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, f1_score


# In[63]:


tfidf_matrix


# In[61]:


#Spliting into the test and train ratio

X= tfidf_matrix
y= data['network']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[73]:


def fit_classifier(clf,X_train, y_train):
    clf.fit(X_train, y_train)

def predicted_labels(clf,features):
    return(clf.predict(features))

A = MultinomialNB()
B = DecisionTreeClassifier()
C = AdaBoostClassifier()
D = KNeighborsClassifier()
E = RandomForestClassifier()
F= SVC(kernel='sigmoid')
G= MLPClassifier()

clf= [A,B,C,D,E,F,G]
predicted_values= [0,0,0,0,0,0,0]

for i in range(0,7):
    start_time= time.time()
    fit_classifier(clf[i], X_train, y_train)
    y_predict= predicted_labels(clf[i],X_test)
    predicted_values[i]= f1_score(y_test, y_predict, average= "micro")
    print('f1 score for the classifier %s is ' %clf[i])
    print(predicted_values[i])
    print("--- %s seconds ---" % (time.time() - start_time))


# Based on the F1-score, neural net performed better. It was expected. Genrally, Naive Bayes and Neural Net perform better but neural net takes some time to train. Now the performance of the classifier can be improve using the grid search.

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.001, 0.01, 0.01, 0.1]}   #<--- alter aplha parameter of the classifier
grid = GridSearchCV( MultinomialNB(),param_grid,refit=True,verbose=3)
grid.fit(X_train, y_train)
grid.best_params_

grid_predictions= grid.predict(X_test)
print(classification_report(y_test, grid_predictions))                           


#   Since neural net performs better under these conditions, deep learning can also be implemented. CNN can be used . The word2vec model can be used as the input to the CNN network. Basic CNN can be made and through dropout and adding layers, desired level of accuracy can be achieved. 

# Since 23824 is very number, the input matrix will be large and will require high computational power to execute the code. 
