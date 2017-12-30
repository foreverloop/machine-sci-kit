"""
given the player's gender, what are the odds a person will steal?

(note that women are slighlty over represented in the contest)

logistic regression in sklearn is used below to predict
based on 39 outcomes
"""
from __future__ import division
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv('golden_balls_final.csv')

prize_values = df['prize_value_gbp']
finalist_1_gender = df['finalist_1_gender']
finalist_2_gender = df['finalist_2_gender']
split_or_steal_1 = df['split_or_steal_finalist_1']
split_or_steal_2 = df['split_or_steal_finalist_2']

finalist_1_gender = finalist_1_gender.values.reshape([-1,1])
finalist_2_gender = finalist_2_gender.values.reshape([-1,1])

lr = LogisticRegression()
lr.fit(finalist_1_gender,split_or_steal_1)

lr2 = LogisticRegression()
lr2.fit(finalist_2_gender,split_or_steal_2)

lr_predict = lr.predict_proba([[0]])
lr2_predict = lr2.predict_proba([[0]])

print lr_predict
print lr2_predict

print "mean chance of stealing: ", (lr_predict[0][0] + lr2_predict[0][0]) / 2
print "mean chance of splitting: ", (lr_predict[0][1] + lr2_predict[0][1]) / 2

#print "coefficient {0}, intercept {1}".format(lr.coef_,lr.intercept_)