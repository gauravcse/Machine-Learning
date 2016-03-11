import scipy as sp
import numpy as np
data=sp.genfromtxt("data.txt",delimiter=" ")
X=data[:,0]
Y=data[:,1]
#Default Values of Theta 1 and Theta 2 is 0.0
theta1,theta2=0.0,0.0
error_margin=0.0001
alpha=0.001
def gradient_descent() :
    old_theta1,old_theta2=0.0,0.0
    while True :
        old_theta1=theta1
        old_theta2=theta2
        temp1=theta1-(alpha*partial_wrt_theta1)
        temp2=theta2-(alpha*partial_wrt_theta2)
        theta1=temp1
        theta2=temp2
        if(abs(theta1-old_theta1)<error and abs(theta2-old_theta2)<error) :
            break
def partial_wrt_theta1() :
    s=0.0
    for i in range(len(X)) :
        x=X[i]
        y=Y[i]
        s+=(theta1+(theta2)*x-y)
    return s
def partial_wrt_theta2() :
    s=0.0
    for i in range(len(X)) :
        x=X[i]
        y=Y[i]
        s+=((theta1+(theta2)*x-y)*x)
    return s   
gradient_descent()
print '{0}  {1}'.format(theta1,theta2)
