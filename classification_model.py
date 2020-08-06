##########-------------- Muath Juady - 11440920 --------------##########
import re
import glob
import os
import nltk
import numpy as np
from sklearn.datasets import load_files
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
#nltk.download('stopwords')
#nltk.download('wordnet')


##############################################################################################
# Load all files data with their corresponding class to "data,classes"

tr = load_files(r"./training/")
data,classes = tr.data, tr.target

##############################################################################################
# Normalization

docs = []

for d in range(0, len(data)):

    doc = re.sub(r'\W', ' ', str(data[d]))        # Remove all the special characters
    doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc)     # remove all single characters
    doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc)      # Remove single characters from the start
    doc = re.sub(r'\s+', ' ', doc, flags=re.I)    # Substituting multiple spaces with single space
    doc = re.sub(r'^b\s+', '', doc)               # Removing prefixed 'b'
    doc = doc.lower()                             # Converting to Lowercase
    doc = word_tokenize(doc)                      # Lemmatization
    doc = [WordNetLemmatizer().lemmatize(word) for word in doc]
    doc = ' '.join(doc)

    docs.append(doc)

##############################################################################################
# Generating document Bag of Words + TFIDF for counting occurrences in other documents
# max_features = most occurring words for training classifier
# min_df       = minimum number of documents that contain this word
# max_df       = include only words that occur in a maximum of % of all documents.

tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
data = tfidf.fit_transform(docs)
data.shape

# Divide the data into training and testing sets
data_train, data_test, classes_train, classes_test = train_test_split(data, classes, test_size=0.28, random_state=0)

##############################################################################################
# Training model using Naive Bayes Classifier + Print test data set Predictions and Stats

#classifier = ComplementNB().fit(data_train, classes_train)
#classifier = BernoulliNB().fit(data_train, classes_train)
classifier = MultinomialNB().fit(data_train, classes_train)

Pred = classifier.predict(data_test)

print(classification_report(classes_test,Pred))
print(accuracy_score(classes_test, Pred))
print("---------------------------------------------------\n\n")

##############################################################################################

#--------------------------------------------------------------------------------------------#

##############################################################################################
# Testing New Documents

NewDocs=[]
NewDocs2 = []
DocsNames=[]

for f in glob.glob("./NewDocs/*"):
    with open(f, "r") as infile:
        NewDocs.append(infile.read())
        DocsNames.append(os.path.basename(f))

for d in range(0, len(NewDocs)):
    doc2 = re.sub(r'\W', ' ', str(NewDocs[d]))      # Remove all the special characters
    doc2 = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc2)     # remove all single characters
    doc2 = re.sub(r'\^[a-zA-Z]\s+', ' ', doc2)      # Remove single characters from the start
    doc2 = re.sub(r'\s+', ' ', doc2, flags=re.I)    # Substituting multiple spaces with single space
    doc2 = re.sub(r'^b\s+', '', doc2)               # Removing prefixed 'b'
    doc2 = doc2.lower()                             # Converting to Lowercase
    doc2 = word_tokenize(doc2)                      # Lemmatization
    doc2 = [WordNetLemmatizer().lemmatize(word) for word in doc2]
    doc2 = ' '.join(doc2)
    NewDocs2.append(doc2)

NewTfidf = tfidf.transform(NewDocs2)
NewPred = classifier.predict(NewTfidf)

for DocName, classes in zip(DocsNames, NewPred):
    print('%r \t\t => %s' % (DocName, tr.target_names[classes]))

##############################################################################################
