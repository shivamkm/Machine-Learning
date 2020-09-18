import numpy as np
import random as rnd
import time as tm

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def steplength( t ):
        return 0.000005/pow(t,0.5)
        return 0.000005/np.sqrt(t)

def getObj( X, y, w, b,C ):
        hingeLoss = np.maximum(1 - np.multiply( (X.dot( w ) + b), y ), 0 )
        return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )
################################
# Non Editable Region Starting #
################################
def solver( X, y, C, timeout, spacing ):
        (n, d) = X.shape
        t = 0
        totTime = 0
        B = 10 
        # w is the normal vector and b is the bias
        # These are the variables that will get returned once timeout happens
        w = np.zeros( (d,) )
       # print ( type(w))
        b = 0
        tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

        # You may reinitialize w, b to your liking here
        # You may also define new variables here e.g. eta, B etc
        #wnew = np.zeros(d+1)
        wnew = np.append([0],w)
        Xnew = np.ones((n,d+1))
        Xnew[:,1:] = X
        alpha = np.zeros( (n,) )
        #rows = np.random.choice(n, B, replace=False)
        #print(rows)
        #print (Xnew)
################################
# Non Editable Region Starting #
################################
        while True:
                print(getObj(X,y,w,b,C))
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
#                rows = np.random.choice(d+1, B, replace=False)
#                xb = Xnew[rows]
#                yb = y[rows]
#                #print(np.maximum(1 -  y_new*np.dot(X_new,w_new),0 ))
#                temp =  ((n+0.0)/(B+0.0))*(-2*C*np.maximum((1-yb*(np.dot(xb,wnew))),0)*yb)
#                #print(temp)
#                temp.resize(B,1)
#                #print(temp*X_new)
#                sq_hinge_grad = np.sum(temp*xb , 0)
                k = rnd.randint( 0,d )   
                xb = Xnew[:,k]
                grad = 1 - d * wnew[rows] * y * xb
                grad = grad - alpha /(2*C + 0.0)
                nt = steplength(t)
               # hingeloss = np.maximum(1-discriminant,0)
               # gradhingeloss = g*y[i]*x
                alpha = np.maximum(alpha + nt * grad,0)

                temp = alpha * y
                temp.resize(n,1)
                wnew = np.sum(temp*Xnew,0)
                w = wnew[1:]
                b = wnew[0]
                #w = ( w * (t-1) + wrun)/t
                #b = ( b * (t-1) + brun)/t

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

