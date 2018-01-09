"""
Shows linear regression calculated by hand, as well as through scikit
as well as some analysis of the SSE, SSR and r2 values
"""

from __future__ import division
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import linear_model

df = pd.read_csv('hourvscore.csv')
scores = df['score']
hours = df['hours']
plt.scatter(scores,hours,color='purple')
plt.xlabel('Hours studied')
plt.ylabel('Test Score')
plt.title('Hours studied Vs Test Score')
plt.show()

#means of hours and scores
mean_score = np.mean(scores)
mean_hours = np.mean(hours)

#score deviations from mean
score_dev = []
for score in scores:
    score_dev.append(score - mean_score)

#hour deviations from mean
hour_dev = []
for hour in hours:
    hour_dev.append(hour - mean_hours)

#product of both sets of deviations
deviation_products = []
for idx,h_dev in enumerate(hour_dev):
    deviation_products.append(h_dev * score_dev[idx])

#hour deviations from mean squared
hour_dev_sq = 0
for hour in hour_dev:
    hour_dev_sq += (hour ** 2)
#rate of change in y for each 1 unit increase in x
slope = sum(deviation_products) / hour_dev_sq
#point in graph which slope intercepts the y axis
#yintercept = (slope * mean_hours) - mean_score
yintercept = mean_score - (slope * mean_hours)


#linear equation to calculate points: y = mx + b
yregress = [(slope * x) + yintercept for x in hours]
mean_only_line = [mean_score for _ in scores]
residual_errors = []

#in order to check if the regression line works better
#as compared to just using the mean value of the dependent variable to predict
def check_residuals(observed,predicted):
    return observed - predicted

#calculate residual errors based on the regression lines
for idx,predicted in enumerate(yregress):
    residual_errors.append(check_residuals(scores[idx],predicted))

#SSE also SST
sse_mean = sum([check_residuals(x,mean_score) ** 2 for x in scores])
#SSE regression used to find SSR
sse_regression = sum([x ** 2 for x in residual_errors])

centroid_y = [mean_score]
centroid_x = [mean_hours]

plt.scatter(centroid_x,centroid_y,color='orange')
plt.plot(hours,yregress,'-',label='regression line')
plt.plot(hours,mean_only_line,'--',label='mean line')
plt.scatter(hours,scores,c='green')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.legend()
plt.title('Linear Regression of Hours Studied/Test Score')
plt.show()

reg_line = linear_model.LinearRegression()

hours = hours.values.reshape([-1,1])
scores = scores.values.reshape([-1,1])

reg_line.fit(hours[:70],scores[:70])
score_predict = reg_line.predict(hours[70:])
plt.scatter(hours[70:],scores[70:],c='purple')
plt.plot(hours[70:],score_predict,c='g')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Linear regression from scikit learn')
plt.show()

#obvious slight difference as the hand calculated ones consider
#the entire data set. sk only considers first 70 pairs
#print "hand calculated regression: ",yregress[70:]
#print "sk regression predict: ", score_predict

print "Each extra hour should result in {0} more score on the test".format(slope)
print "mean squared error", mean_squared_error(scores[70:],score_predict)
print "SSE mean: " , sse_mean #also considered our SST and max sum squares

SSR = sse_mean - sse_regression
print "SSE taking regression into account: ", SSR

#coefficient of determination
# approx 59% of the sum of squares can be explained by the regression model
#the rest, we still attribute to error. this indicates a reasonable fit
print "r2: ",SSR / sse_mean
print "r2 Sk model: ", r2_score(scores[70:],score_predict)