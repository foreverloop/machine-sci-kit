import json,requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#not going to show my API auth key in public!
with open("endpoint.csv","r") as f:
  reader = csv.reader(f,delimiter=',')
  for row in reader:
    endpoint = row[0]

wreports = json.loads(requests.get(endpoint).text)
reg_line = linear_model.LinearRegression()

def check_residuals(observed,predicted):
  return observed - predicted

#5 days regression line
real_temp = []
feel_temp = []

for idx,all_reports in enumerate(wreports["SiteRep"]["DV"]["Location"]["Period"]):

#all_reports is a list of weather reports, so must nest this
  for report in all_reports["Rep"]:
  	real_temp.append(float(report["T"]))
  	feel_temp.append(float(report["F"]))

real_temp = np.reshape(real_temp,(-1,1))
feel_temp = np.reshape(feel_temp,(-1,1))

reg_line.fit(real_temp[:25],feel_temp[:25])
feel_temp_predict = reg_line.predict(real_temp[25:])

print "Coefficients:", reg_line.coef_
#mean squared error
print"Mean squared error: %.2f" % mean_squared_error(feel_temp[25:], feel_temp_predict)
#variance score: 1 is perfect prediction
print "variance: %.2f" % r2_score(feel_temp[25:], feel_temp_predict)

#probably a more concise way than this...
residual_errors = []
for idx,predicted in enumerate(feel_temp_predict):
  residual_errors.append(check_residuals(feel_temp[idx],predicted))

sse_mean = sum([check_residuals(x,np.mean(feel_temp)) ** 2 for x in feel_temp]) #SSE
sse_regression = sum(([x **2 for x in residual_errors])) #SSR

print "sum squared errors (mean): ", sse_mean
print "sum squared errors (regression): ", sse_regression
print "SST (mean): ", sse_mean
print "SST (with regression taken away)", sse_mean - sse_regression

#plot the test points
plt.scatter(real_temp[25:],feel_temp[25:],color='green')
#plot 'best fit' line
plt.plot(real_temp[25:],feel_temp_predict,color='blue')
#mean line
plt.plot(real_temp[25:],([np.mean(feel_temp[25:]) for x in feel_temp[25:]]),color='orange')

plt.xlabel('Actual Temperature (C)')
plt.ylabel('Feels like Temperature (C)')
plt.title('Actual Temp / Feels Like Temp Regression Line')
plt.show()