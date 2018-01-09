from matplotlib import pyplot as plt
from sklearn import tree
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import graphviz

df = pd.read_csv('wine.csv')
color_i = df['Color intensity']
alcohol_percent = df['Alcohol']
labels = df['wine type']

#there's a weak correlation, backed by the p-value
plt.scatter(color_i,alcohol_percent)
plt.show()
print pearsonr(color_i,alcohol_percent)

#just try to predict 3 right, using the rest for training
test_idx = [163,104,46]

train_target = np.delete(np.asarray(labels),test_idx)
train_data = np.delete(np.asarray(alcohol_percent),test_idx)
train_data2 = np.delete(np.asarray(color_i),test_idx)

test_target = labels[test_idx]

test_data = alcohol_percent[test_idx]
test_data2 = color_i[test_idx]

#has to be put through as a 2d array
two_feat = np.column_stack((train_data,train_data2))

clf = tree.DecisionTreeClassifier()
clf.fit(two_feat,train_target)

two_feat_test = np.column_stack((test_data,test_data2))

print "Actual types: {0}, predicted types: {1}".format(test_target,clf.predict(two_feat_test))

dot_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph.render("wines")
