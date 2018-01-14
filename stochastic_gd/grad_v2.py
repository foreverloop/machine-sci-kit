"""
REFERENCE ONLY -- this was mostly not written by myself, from the book 'Data science from scratch'

but I have added a run of the stochastic gradient descent methods to test it
as well as notes to clarify what is happening

"""

from __future__ import division
from collections import Counter
from matplotlib import pyplot as plt
import math, random

def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
   return math.sqrt(squared_distance(v, w))

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

def partial_difference_quotient(f, v, i, h):

    # add h to just the i-th element of v
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]
         
    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)] 

def step(v, direction, step_size):
    """move step_size in the direction from v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v): 
    return [2 * v_i for v_i in v]

def safe(f):
    """define a new function that wraps f and return it"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf') #infinity
    return safe_f


# minimize / maximize batch
# standard gradient descent
#

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""
    #target_fn = sum_of_squares - returns one figure
    #gradient_fn = sum_of_squares_gradient - returns 3 values in a list
    #theta_0 =  [-1,-3,4] (it's random, this is just an example)

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    theta = theta_0                           # set theta to initial value
    target_fn = safe(target_fn)               # safe version of target_fn
    value = target_fn(theta)                  # value we're minimizing

    #value  = total size of the error we're going to minimise
    #print value

    #repeat the following set of instructions until we converge:
    # 1. find the gradient (like a derivative) for the current theta values
    # 2. generate a list of potential theta values, using different step sizes (learning rates)
    # 3. choose the smallest theta, based on the function we're trying to minimise
    # 4. feed that theta into the function we're minimising to give the new error size
    # 5. if the new error size isn't significantly smaller (specified by tolerance), we should leave
    # 6.  otherwise, assign the values of the error and theta to our closest values and repeat the loop

    while True:
        #print "theta: ", theta
        gradient = gradient_fn(theta) 
        #print "gradient: ", gradient
        #print "gradient interation: {0} ".format(gradient)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]
        #print "next thetas {0}".format(len(next_thetas))

        #out of all attempted step sizes 
        #choose the one that minimizes the error function (sum square errors in this instance)
        next_theta = min(next_thetas, key=target_fn)

        #the smallest one should be compared with our current best value
        next_value = target_fn(next_theta)
        
        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta #return our best fit
        else:
            #use our best fit as new values for the next iteration
            theta, value = next_theta, next_value

def negate(f):
    """return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)
    
def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0, 
                          tolerance)

#
# minimize / maximize stochastic
#

def in_random_order(data):
    """generator that returns the elements of data in random order"""
    indexes = [i for i, _ in enumerate(data)]  # create a list of indexes
    random.shuffle(indexes)                    # shuffle them
    for i in indexes:                          # return the data in that order
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    #target = stoc_squared_error 
    #gradient = stoc_squared_error_gradient
    #x = list of number of friends
    #y = list of number of minutes spent on site
    #theta = initial guess of min error
    #alpha = initial learning rate

    data = zip(x, y)
    theta = theta_0                             # initial guess
    alpha = alpha_0                             # initial step size
    min_theta, min_value = None, float("inf")   # the minimum so far
    iterations_with_no_improvement = 0
    """
    #repeat the following steps to find the coefficients of the simple, singular,linear regression:
    #
    # 1. continue to try to find a smaller values as long as we haven't gone 100 attempts dry
    # 2. compute the new error using the sum of the squared errors for both data features
    # 3. if the new error is smaller than our previous minimum re-assign our new min and the inputs we used to get to it
    # 4. otherwise, +1 to the no improvement counter and shrink step size slightly
    # 5. for each data point in the list, get the partial derivates
    #    then assign theta to it's new value for the next iteration by
    #    subtracting the new partial derivaties * step size, from the existing theta value 
    #
    #    this method means we take a guess at the next step for each data point
    #    NOT the entire data set, so could be faster for a large dataset
    #    which is approximately normally distributed
    """
    total_iter = 0
    while iterations_with_no_improvement < 100:

        #value of sum sq errors.
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )
        
        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9

        total_iter += 1
        # and take a gradient step for each of the data points    
        for x_i, y_i in in_random_order(data):

            #compute the partial derivatives for this datapoint
            #theta is our current slope and intercept (alpha,beta)
            gradient_i = gradient_fn(x_i, y_i, theta)

            #we're descending, so remove the learning rate * the guess we made from theta
            #this is additive which is why it's occuring in a loop
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
            
            #plot some of the lines as we go to visually show the guesses
            if total_iter % 100 == 0 and random.randint(1,10) == 7:
                ygr = [theta[1] * n + theta[0] for n in num_friends_good]
                plt.plot(num_friends_good,ygr,color='orange',linewidth='0.2')


    ygr = [theta[1] * n + theta[0] for n in num_friends_good]

    plt.title('Stochastic Gradient Descent')
    plt.xlabel('Number of Friends')
    plt.ylabel('Number of Minutes Spent on Site')
    plt.plot(num_friends_good,ygr,color='green')
    plt.scatter(num_friends_good,daily_minutes_good,color='purple')
    plt.show()
    return min_theta

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    print "dot args: ", v,w
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def predict(x_i, beta):
    print "predict:  ", x_i
    return dot(x_i, beta)

def error(x_i, y_i, beta):
    print "error:", x_i
    return y_i - predict(x_i, beta)
    
def squared_error(x_i, y_i, beta):
    print "squared_error: ",x_i,y_i
    return error(x_i, y_i, beta) ** 2

def stoc_predict(alpha,beta,x_i):
    return beta * x_i + alpha

def stoc_error(alpha,beta,x_i,y_i):
    return y_i - stoc_predict(alpha,beta,x_i)

def stoc_squared_error(x_i,y_i,theta):
    alpha,beta = theta
    return stoc_error(alpha,beta,x_i,y_i) ** 2

def stoc_squared_error_gradient(x_i,y_i,theta):
    #get alpha and beta partial derivatives
    #presume -2 because stoc_squared_error will always be a positive result
    #but we are trying to lower alpha/beta
    alpha,beta = theta
    return [-2 * stoc_error(alpha,beta,x_i,y_i),
    -2 * stoc_error(alpha,beta,x_i,y_i) * x_i]

def squared_error_gradient(x_i, y_i, beta):
    """the gradient corresponding to the ith squared error term"""
    print "squared error gradient: ", x_i
    return [-2 * x_ij * error(x_i, y_i, beta)
            for x_ij in x_i]

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn),
                               negate_all(gradient_fn),
                               x, y, theta_0, alpha_0)

if __name__ == "__main__":

    print "using the gradient"

    v = [random.randint(-10,10) for i in range(3)]
    print "minimise these values: ", v
    tolerance = 0.0000001

    while True:
        #print v, sum_of_squares(v)
        gradient = sum_of_squares_gradient(v)   # compute the gradient at v
        next_v = step(v, gradient, -0.01)       # take a negative gradient step
        if distance(next_v, v) < tolerance:     # stop if we're converging
            break
        v = next_v                              # continue if we're not

    print "minimum v", v
    print "minimum value", sum_of_squares(v)
    print


    print "using minimize_batch"

    v = [random.randint(-10,10) for i in range(3)]
    print "minimise these values: ",v
    v = minimize_batch(sum_of_squares, sum_of_squares_gradient, v)

    print "minimum v", v
    print "minimum value", sum_of_squares(v)
    print

    num_friends_good = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    daily_minutes_good = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]
    
    print "using minimize_stochastic"

    random.seed(0)
    theta = [random.random(),random.random()]

    alpha,beta = minimize_stochastic(stoc_squared_error,
                                    stoc_squared_error_gradient,
                                    num_friends_good,
                                    daily_minutes_good,
                                    theta,0.0001)

    print "alpha: {0}, beta: {1}".format(alpha,beta)
    