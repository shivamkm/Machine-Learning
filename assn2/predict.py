import numpy as np
from numpy import random as rand
import os
import operator
import subprocess
import sys
import bisect

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# X: n x d matrix in csr_matrix format containing d-dim (sparse) features for n test data points
# k: the number of recommendations to return for each test data point in ranked order

# OUTPUT CONVENTION
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of labels with the i-th row
# containing k labels which it thinks are most appropriate for the i-th test point. Labels must be returned in
# ranked order i.e. the label yPred[i][0] must be considered most appropriate followed by yPred[i][1] and so on

# CAUTION: Make sure that you return (yPred below) an n x k numpy (nd) array and not a numpy/scipy/sparse matrix
# The returned matrix will always be a dense matrix and it terribly slows things down to store it in csr format
# The evaluation code may misbehave and give unexpected results if an nd-array is not returned

def getReco( X, k ):
    # Find out how many data points we have
    n = X.shape[0]
    f = open('my_test_feats','w')
    superstring = ""
    superstring += str(n)+" 16385\n"
    for i in range(0,n):
        string  = ""
        X_temp = X[i].toarray()
        indices = np.argwhere(X_temp > 0.0)
        for j in indices:
            string += str(j[1])+":"+str(X_temp[0][j[1]])+" "
        string  = string[:-1]
        string+= "\n"
        superstring += string
    f.write(superstring)
    f.close()
    output = subprocess.check_output(["./bonsai_predict2","my_test_feats","./sandbox/results/CS771/model/"])
    output = output.decode().split('\n')
    yPred = np.ndarray((n,k))
    j = 0
    count = 0
    for i in output:
        if j is 0:
            j +=1
        else:
            i = i[:-1]
            a = i.strip().split()
            b=[]
            for value in a:
                bisect.insort(b,[float(value.split(":")[1]),int(value.split(":")[0])]) #[prob,index]
            if len(b) > 0 and count < n:
                for i in range(k):
                    yPred[count][i] = b[len(b)-i-1][1]
                count+=1
    os.system("rm my_test_feats")
    return yPred
