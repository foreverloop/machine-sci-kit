"""
IMPORTANT: this code is largely based on 'https://github.com/llSourcell/linear_regression_live'
but I have added plots to show this visually, and a few things to make it clearer to myself 
-----
The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
this is just to demonstrate gradient descent

"""

import numpy as np
from matplotlib import pyplot as plt
"""
y = mx + b
m is slope, b is y-intercept
only used much later on after gradient done it's thing
so we can check the improvement in a quantifiable manner
"""
def compute_error_for_line_given_points(b, m, points):
    totalError = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

"""
the smallest error also gives the ideal b and m values
e.g the ideal values for the equation which will
give the line of best fit

to calculate this gradient, we need the partial deriviatives

"""
def step_gradient(b_current, m_current, points, learningRate):
    #b = intercept m = slope
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        #current point set (does for all points in entire dataset)
        x = points[i, 0]
        y = points[i, 1]
        
        #partial deriviative of b and m at their current point, summed for all points
        # intercept calculaion, this is, apparently, the same every time
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        # slope calculation
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    #learning rate determines how fast we want to approach our minimum
    #it could be worth looking at making this dynamic to improve efficiency
    #the reason we are deducting the partial deriviative for b and m is we're finding the minimum error
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    #row,column
    hours = points[0:,:1]
    scores = points[0:,1:2]
    plt.scatter(hours,scores,color='green')
    for idx,i in enumerate(range(num_iterations)):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        
        y_predicts = [x * m + b for x in hours]
        if idx % 2 == 0:
          plt.plot(hours,y_predicts,color='purple')
        else:
          plt.plot(hours,y_predicts,color='pink')
    #our best fit is to be the red line
    plt.plot(hours,y_predicts,color='red')
    plt.show()
    return [b, m]

#highest level
def run():
    plt.title('Gradient Descent to Find Linear Regression Line')
    plt.xlabel('Hours Studied')
    plt.ylabel('Test Score')
    points = np.genfromtxt("datanotitle.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess

    #the higher this number, the lower our eventual error, but we don't want to go on forever
    #particularly if we are not improving much anymore
    num_iterations = 100
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))


if __name__ == '__main__':
    run()

