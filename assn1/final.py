import numpy as np
import random as rnd
import time as tm

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def solver( X, y, C, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0

	# w is the normal vector and b is the bias
	# These are the variables that will get returned once timeout happens
	w = np.zeros( (d,) )
	b = 0
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
	w = np.random.randn(d)
	b = rnd.random()

	#X_new contains the data X and it's first column is set to 1
	X_new = np.ones((n,d+1))
	X_new[:,1:] = X
	#W_new included to bias term. W_new[0] is effectively equal to the bias term.
	W_new = np.append([0],w)
	#j is the weight number to be changed.
	j=0

	# You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc

################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w, b, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

		prediction = X_new.dot(W_new) #prediction contains the inner product of X and weights
		pred_correct = y*prediction # pred_correct contains y_j <W,x_j> as its elements
		filter = (1-pred_correct)>=0 # filter[i]  contains if ith sample is predicted correcly or not.

		cache1 = np.sum(y*X_new[:,j]*filter) #cache1 is summation y_j*x_i^j
		cache2 = np.sum(X_new[:,j]*X_new[:,j]*filter) #cache2 is summation x_i^j squared
		cache3 = np.sum(prediction*X_new[:,j]*filter) #cache3 is summation x_i^j <w,x^j>

		#updating jth component of weight according to the update rule, derived in the report
		W_new[j] = (2*C*cache1 - 2*C*cache3 + 2*C*W_new[j]*cache2)/(1+2*C*cache2)
		#Extracting w and b from W_new
		w = W_new[1:]
		b = W_new[0]
		#choosing next co-ordinate cyclically
		j = (j+1)%(d+1)

		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses - severe penalties await

		# Please note that once timeout is reached, the code will simply return w, b
		# Thus, if you wish to return the average model (as we did for GD), you need to
		# make sure that w, b store the averages at all times
		# One way to do so is to define two new "running" variables w_run and b_run
		# Make all GD updates to w_run and b_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w and b
		# w = (w * (t-1) + w_run)/t
		# b = (b * (t-1) + b_run)/t
		# This way, w and b will always store the average and can be returned at any time
		# w, b play the role of the "cumulative" variable in the lecture notebook
		# w_run, b_run play the role of the "theta" variable in the lecture notebook

	return (w, b, totTime) # This return statement will never be reached
