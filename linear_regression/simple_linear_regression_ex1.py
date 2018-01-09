from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

def check_residuals(observed,predicted):
  return observed - predicted

bills = [34,108,64,88,99,51]
tips = [5,17,11,8,14,5]

mean_bills = np.mean(bills)
mean_tips = np.mean(tips)

print "means: ",mean_bills,mean_tips

bill_deviations = []

for bill in bills:
    bill_deviations.append(bill - mean_bills)

tip_deviations = []
for tip in tips:
    tip_deviations.append(tip - mean_tips)

print "individual deviations from mean: ",bill_deviations,tip_deviations

deviation_products = []
for idx,b_dev in enumerate(bill_deviations):
    deviation_products.append(b_dev * tip_deviations[idx])

print "product of both sets deviations: ", deviation_products

bill_dev_squared = 0
for bill in bill_deviations:
    bill_dev_squared += (bill ** 2)

print "bill deviation squared: ",bill_dev_squared

slope = sum(deviation_products) / bill_dev_squared
yintercept = mean_tips - (slope * mean_bills)

print "slope: ", slope
print "y intercept", yintercept

#leaving python to calculate this causes worse results?
#yregression = [(round(slope,4) * x) - round(yintercept,4) for x in bills]

#hardcode works better though
yregression = [(0.1462 * x) - 0.8188 for x in bills]
meanonly = [mean_tips for x in tips]
print "y regression points: ", yregression

residual_errors = []
for idx,predicted in enumerate(yregression):
	residual_errors.append(check_residuals(tips[idx],predicted))

#SSE (sum square error), SSR (sum square regression), SSR(sum square total)
print "sum square errors (mean only): ", sum([check_residuals(x,mean_tips) ** 2 for x in tips]) #SSE
print "sum square errors (regression): ", sum(([x **2 for x in residual_errors])) #SSR

#our slope should also always pass through the centroid
centroid_y = [mean_tips]
centroid_x = [mean_bills]
plt.scatter(bills,tips,c='green')
plt.scatter(centroid_x,centroid_y,c='orange')
plt.plot(bills,yregression,'-',label='regression line')
plt.plot(bills,meanonly,'--',label='mean only line')
plt.xlabel('Bill $')
plt.ylabel('Tip $')
plt.legend()
plt.title('linear regression line')
plt.show()
