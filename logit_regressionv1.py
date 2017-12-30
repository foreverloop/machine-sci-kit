"""
given the prize money, what are the odds a person will steal?

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

split_or_steal_1 = df['split_or_steal_finalist_1']
split_or_steal_2 = df['split_or_steal_finalist_2']

prize_values = prize_values.values.reshape([-1,1])

lr = LogisticRegression()
lr.fit(prize_values,split_or_steal_1)

lr2 = LogisticRegression()
lr2.fit(prize_values,split_or_steal_2)

print "enter amount to predict split or steal: "
predict_split_or_steal_for_amount = float(raw_input())

lr_predict = lr.predict_proba([[predict_split_or_steal_for_amount]])
lr2_predict = lr2.predict_proba([[predict_split_or_steal_for_amount]])

print "predicted steal-split percent for {0}: {1}".format(predict_split_or_steal_for_amount,lr_predict)
print "predicted steal-split percent for {0}: {1}".format(predict_split_or_steal_for_amount,lr2_predict)

print "mean chance of stealing: ", (lr_predict[0][0] + lr2_predict[0][0]) / 2
print "mean chance of splitting: ", (lr_predict[0][1] + lr2_predict[0][1]) / 2

#print "coefficient {0}, intercept {1}".format(lr.coef_,lr.intercept_)