# NLP Document Classification Model

Build a Naïve Bayes classification model that is able to predict the class (category) of an unseen document correctly out of 91 classes. You are given the following datasets.

•	A training dataset: a collection of 91 folders whose names represent the class of the contained documents, i.e., the class of a specific document is the name of the folder holding this file. That means, all the documents are labeled.

•	A test dataset: this dataset has an organization similar to what is mentioned for the training dataset, i.e., the documents in the test sets are labeled and ready for evaluation purposes.


Follow the following steps that helps building a good classification model:

1.	Word Tokenization: splitting the text into uni-gram tokens.
You can either use you own tokenization methodology or the Stanford tokenizer: http://nlp.stanford.edu/software/tokenizer.html

2.	Remove stopwords: find a list of stop words and use it to exclude stope words from the computation.

3.	Token normalization: use Porters’ Stemmer to return a token back to its base.

4.	Vocabulary set extraction.

5.	Estimation of model parameters. *Estimate the prior distribution of the Naïve Bayes Classifier for each class. Estimate the likelihood of each word in the training dataset for each class. Note: do not forget to handle Zero probabilities using Add-1 Laplacian Smoothing and to deal with unknown words appearing in the test dataset.*

6.	Document classification: predict the class of each document in the test dataset using the model built in step 4.

7.	Model evaluation: Calculate (1) F-score at each class (2) average of all F-scores (3) accuracy

8.	EXTRA: think of new features that can be included into the Naïve Bayes Classifier, which contribute to improve the system performance.
