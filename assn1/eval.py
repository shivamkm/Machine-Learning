import numpy as np
from final import solver

def getObj( X, y, w, b ):
        hingeLoss = np.maximum(1 -  np.multiply( (X.dot( w ) + b), y ), 0 )
        return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )

Z = np.loadtxt( "data" )

y = Z[:,0]
X = Z[:,1:]
C = 1

avgTime = 0
avgPerf = 0

# To avoid unlucky outcomes try running the code several times
numTrials = 10
# 30 second timeout for each run
timeout = 5
# Try checking for timeout every 100 iterations
spacing = 100

for t in range( numTrials ):
        (w, b, totTime) = solver( X, y, C, timeout, spacing )
        avgTime = avgTime + totTime
        #print (w.shape)
        #print (X.shape)
        loss = getObj( X, y, w, b )
        print ( loss )
        avgPerf = avgPerf + loss

print( avgPerf/numTrials, avgTime/numTrials )
