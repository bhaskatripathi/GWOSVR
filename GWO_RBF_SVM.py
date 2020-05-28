# -*- coding: utf-8 -*-
"""
Created on Thursday May 22,2020
@author: BHASKAR TRIPATHI

This code Optimizes a Support Vector machine by using Grey Wolf Optimizer. The original algorithm of GWO was developed by Dr Ali Mirjalili in Matlab. I was inspired by his method and Hybridized it with an SVR objective function (or SVC depending upon what you need). 
Intution - https://www.youtube.com/watch?v=B3jqn9lCNxI

"""
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.svm import SVR
import sklearn.model_selection
import numpy.random as rd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import warnings, pandas as pd,numpy as np, time, math, configparser,random



## 1. GWO optimization algorithm
def gwo(X_train,X_test,y_train,y_test,SearchAgents_no,T,dim,lb,ub):
    Alpha_position=[0,0] # Initialize the position of Alpha Wolf
    Beta_position=[0,0]
    Delta_position=[0,0]  

    Alpha_score = float("inf") # Initialize the value of Alpha Wolf's objective function 
    Beta_score = float("inf")
    Delta_score = float("inf")
 
    Positions = np.dot(rd.rand(SearchAgents_no,dim),(ub-lb))+lb # initialize the first search position
    
    Convergence_curve=np.zeros((1,T))# initialization fusion curve

    iterations = []
    accuracy = []

    # Main Loop
    t = 0 
    while t < T:
        
        # Iterate over each wolf
        for i in range(0,(Positions.shape[0])):
            #If the search position exceeds the search space, you need to return to the search space 
            for j in range(0,(Positions.shape[1])): 
                Flag4ub=Positions[i,j]>ub
                Flag4lb=Positions[i,j]<lb
                #If the wolf's position is between the maximum and minimum, the position does not need to be adjusted, if it exceeds the maximum, the maximum returns to the maximum value boundary

                if Flag4ub:                   
                    Positions[i,j] = ub
                if Flag4lb:                   
                    Positions[i,j] = lb
            '''SVM MODEL TRAINING - FOR CLASSIFICATION PROBLEM DATASET''' 
            #rbf_svm = svm.SVC(kernel = 'rbf', C = Positions[i][0], gamma = Positions[i][1]).fit(X_train, y_train)  #svm
            #cv_accuracies = cross_val_score(rbf_svm,X_test,y_test,cv =3,scoring = 'accuracy')
            
            '''SVR MODEL TRAINING - FOR REGRESSION PROBLEM DATASET'''
            rbf_regressor = svm.SVR(kernel = 'rbf', C = Positions[i][0], gamma = Positions[i][1]).fit(X_train, y_train)  #svm        
            cv_accuracies = cross_val_score(rbf_regressor,X_test,y_test,cv =3,scoring = 'neg_mean_squared_error') # Taking negated value of MSE
            
            
            #To minimize the error rate
            accuracies = cv_accuracies.mean()            
            fitness_value = (1 - accuracies)*100
            if fitness_value<Alpha_score: # If the objective function value is less than the objective function value of Alpha Wolf
                Alpha_score=fitness_value # Then update the target function value of Alpha Wolf to the optimal target function value
                Alpha_position=Positions[i] # At the same time update the position of the Alpha wolf to the optimal position
            if fitness_value>Alpha_score and fitness_value<Beta_score:  # If the objective function value is between the objective function value of Alpha Wolf and Beta Wolf
                Beta_score=fitness_value # Then update the target function value of Beta Wolf to the optimal target function value
                Beta_position=Positions[i]
            if fitness_value>Alpha_score and fitness_value>Beta_score and fitness_value<Delta_score: #If the target function value is between the target function value of Beta Wolf and Delta Wolf
                Delta_score=fitness_value  # Then update the target function value of Delta Wolf to the optimal target function value
                Delta_position=Positions[i]


        a=2-t*(2/T)
        
        # Iterate over each wolf
        for i in range(0,(Positions.shape[0])):
            #Traverse through each dimension
            for j in range(0,(Positions.shape[1])): 
                #Surround prey, location update                 
                r1=rd.random(1)#Generate a random number between 0 ~ 1
                r2=rd.random(1)               
                A1=2*a*r1-a # calculation factor A
                C1=2*r2 # calculation factor C

                #Alphawolf location update
                D_alpha=abs(C1*Alpha_position[j]-Positions[i,j])
                X1=Alpha_position[j]-A1*D_alpha
                       
                r1=rd.random(1)
                r2=rd.random(1)

                A2=2*a*r1-a
                C2=2*r2

                # Beta wolf location update
                D_beta=abs(C2*Beta_position[j]-Positions[i,j])
                X2=Beta_position[j]-A2*D_beta
                r1=rd.random(1)
                r2=rd.random(1)

                A3=2*a*r1-a
                C3=2*r2

                # Delta Wolf Location Update
                D_delta=abs(C3*Delta_position[j]-Positions[i,j])
                X3=Delta_position[j]-A3*D_delta

                # Location update
                Positions[i,j]=(X1+X2+X3)/3

        
        t = t + 1
        iterations.append(t)
        accuracy.append((100-Alpha_score)/100)
        print('----------------Count of iterations----------------' + str(t))
        print(Positions)
        print('C and gamma:' + str(Alpha_position))
        print('accuracy:' + str((100-Alpha_score)/100))

    best_C=Alpha_position[0]
    best_gamma=Alpha_position[1]

    return best_C,best_gamma,iterations,accuracy

# Plot Convergence Curve
def plot(iterations,accuracy):
    plt.plot(iterations,accuracy)
    plt.xlabel('Count of iterations',size = 20)
    plt.ylabel('Accuracy',size = 20)
    plt.title('GWO-SVM parameter optimization')
    plt.show()

if __name__ == '__main__':
    print('----------------1. Load data-------------------')
    file_name='BTC_Jan2020_IQR.csv'
    df=pd.read_csv(file_name)
    df = df.sort_index()
    y=df.USD_Exchange_Price
    X=df.drop('USD_Exchange_Price',axis=1)
    X=X.drop('Date',axis=1)
    X_train,X_test,y_train,y_test= train_test_split(X, y, test_size = 0.2, random_state = 0)

    print('----------------2. Parameter setting------------')
    SearchAgents_no=20 #Number of Wolfs
    T=25 # maximum number of iterations
    dim=2 #Need to optimize two variables - Cost and Gamma
    lb=0.01 #lower bound Parameter
    ub=10 #upper bound Parameter

    print('----------------3.LARGE-----------------')
    best_C,best_gamma,iterations,accuracy = gwo(X_train,X_test,y_train,y_test,SearchAgents_no,T,dim,lb,ub)

    print('----------------4. The result shows-----------------')
    print("The best C is " + str(best_C))
    print("The best gamma is " + str(best_gamma))
    plot(iterations,accuracy)
    
#Apply Optimal Parameters to SVR
svr_regressor= SVR(kernel='rbf', C = best_C,gamma=best_gamma )
svr_regressor.fit(X_train,y_train)
y_pred = svr_regressor.predict(X_test)
# APPLYING K-FOLD CROSS VALIDATION on RF model
accuracies = cross_val_score(svr_regressor, X = X_train, y = y_train, cv = 10)
accuracy_mean= accuracies.mean()
accuracies.std()*100

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2=r2_score(y_test, y_pred)
nrmse=rmse/(y_test.max() - y_test.min())
print("SVR RESULTS - C AND GAMMA PARAMETERS OPTIMIZED BY GRAY WOLF OPTIMIZATION")
print("RMSE =", rmse)
print("MSE =", mse)
print("Normalized RMSE=",nrmse)
print("R Square =",r2)
print("K-fold accuracy mean",accuracy_mean)
print('Misclassified Samples: %d'%(y_test!=y_pred).sum())
