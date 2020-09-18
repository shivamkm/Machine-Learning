import numpy as np
import random as rnd
import time as tm
import sys

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def getObj( X, y, w, b , C ):
        hingeLoss = np.maximum(1 -  np.multiply( (X.dot( w ) + b), y ), 0 )
        return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )

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

        # You may reinitialize w, b to your liking here
        # You may also define new variables here e.g. eta, B etc

################################
# Non Editable Region Starting #
################################
        X_new = np.ones((n,d+1))
        X_new[:,1:] = X
        alpha = np.zeros((n,1))
        #alpha = np.zeros((n,))
        #alpha = np.expand_dims(alpha,axis=1)
        curr_variable = 0
        y_new = np.expand_dims(y,axis=1)
        X_pre = (y_new)*X_new
        '''pre_process = np.matmul(X_pre,X_pre.T)'''
        W_new = np.sum(alpha*X_pre,0)
        #print(W_new)
        while True:
                print(getObj(X,y,w,b,C))
                t = t + 1
                if t % spacing == 0:
                        #print(totTime)
                        toc = tm.perf_counter()
                        totTime = totTime + (toc - tic)
                        if totTime > timeout:
                                return (w, b, totTime)
                        else:
                                tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
                #print(alpha[curr_variable]*y[curr_variable]*X_new[curr_variable,:])
                W_new_c = W_new - alpha[curr_variable]*y[curr_variable]*X_new[curr_variable,:]
                '''numerator = (1-np.sum(alpha*pre_process[curr_variable,:])+alpha[curr_variable]*pre_process[curr_variable,curr_variable])
                denominator = 1/(2*C) + np.dot(X_new[curr_variable,:],X_new[curr_variable,:])
                alpha[curr_variable] = numerator/denominator'''
                '''cumulate = 0
                for i in range(n):
                        if i != curr_variable:
                                cumulate = cumulate - (alpha[i]*y[i]*y[curr_variable]*np.dot(X[i,:],X[curr_variable,:]))
                #alpha[curr_variable] = 2*C*(1-y[curr_variable]*np.dot(W_new,X_new[curr_variable,:]))
                alpha[curr_variable] = (1-cumulate)/'''
                norm_xi = np.dot(X_new[curr_variable,:],X_new[curr_variable,:])
                numerator = 1 - y[curr_variable]*(np.dot(W_new,X_new[curr_variable,:]) - alpha[curr_variable]*y[curr_variable]*norm_xi)
                denominator = norm_xi + (1/(2*C))
                alpha[curr_variable] = numerator/denominator
                if alpha[curr_variable] < 0:
                        alpha[curr_variable] = 0

                #print(alpha[curr_variable]*y[curr_variable]*X_new[curr_variable,:])
                W_new = W_new_c + alpha[curr_variable]*y[curr_variable]*X_new[curr_variable,:]
                #W_new = np.sum(alpha*X_pre,0)
                curr_variable = (curr_variable + 1)%(n)

                w = W_new[1:]
                b = W_new[0]

#                w = ( w * (t-1) + wrun)/t
#                b = ( b * (t-1) + brun)/t




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
