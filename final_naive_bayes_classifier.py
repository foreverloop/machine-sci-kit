"""
280 is approx 70 percent of the total for this dataset
"""
from __future__ import division
from collections import Counter
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer as CV
import pandas as pd
import numpy as np

df = pd.read_csv('eminspam.csv')
count_vector = CV()

train_words = df['CONTENT'][:280] #data is already randomly sorted thankfully
test_words = df['CONTENT'][280:] #test model later

train_counts = count_vector.fit_transform(train_words)

clf = BernoulliNB()
#training data (counter vector of words in content),target (pre labelled target data)
#fit = training, basically
clf.fit(train_counts,np.asarray(df['CLASS'][:280]))

#read the words and prepare them into a counter vector of words to feed in
test_vector = count_vector.transform(test_words)

#predict if the unlabeled tests are spam, based on training
predicted = clf.predict(test_vector)

#so now, since we actually know what those 168 messages are, we can check
actual_values = np.asarray(df['CLASS'][280:])

results = []
for idx,number in enumerate(predicted):
    if number == 1 and actual_values[idx] == 1:
        results.append(1) #correct positive
    elif number == 1 and actual_values[idx] == 0:
    	results.append(2) #incorrect positive
    elif number == 0 and actual_values[idx] == 1:
    	results.append(3) #incorrect negative
    else: # 0 and 0
        results.append(0) #correct negative

accuracy = Counter(results)
total = (accuracy[0] + accuracy[1] + accuracy[2] + accuracy[3])

print "Overall Accuracy: ", (accuracy[1] + accuracy[0]) / total
print "Correct Positives: ", accuracy[1] / total
print "Correct Negatives: ", accuracy[0] / total
print "Incorrect Positives: ", accuracy[2] / total
print "Incorrect Negatives: ", accuracy[3] / total
