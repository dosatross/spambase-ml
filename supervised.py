# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:52:19 2018

@author: Hayden Warmington
"""
from matplotlib.pyplot import (figure, plot, subplot, title, xlabel, ylabel, 
                               hold, contour, contourf, cm, colorbar, show,
                               legend)
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import model_selection, tree
from sklearn.neighbors import KNeighborsClassifier
import neurolab as nl


def ANN(X,Y):
    # Parameters for neural network classifier
    max_hidden_units = 32      # max number of hidden units
    n_train = 2             # number of networks trained in each k-fold
    
    # These parameters are usually adjusted to: (1) data specifics, (2) computational constraints
    learning_goal = 10     # stop criterion 1 (train mse to be reached) : error goal
    max_epochs = 64        # stop criterion 2 (max epochs in training)
    
    # K-fold CrossValidation
    K = 2
    CV = model_selection.KFold(K,shuffle=True)
    
    mean_hidden_units = []
    
    for l in range(1,max_hidden_units+1):
        print('\nHidden Layers: {0}/{1}'.format(l,max_hidden_units))    
        # Variable for classification error
        errors = np.zeros((K))
        error_hist = np.zeros((max_epochs,K))
        bestnet = list()
        
        k = 0
        
        for train_index, test_index in CV.split(X,Y):
            print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
            
            # extract training and test set for current CV fold
            X_train = X[train_index,:]
            y_train = Y[train_index,:]
            X_test = X[test_index,:]
            y_test = Y[test_index,:]
        
            best_train_error = 1e100
            for i in range(n_train):
                # Create 3 layer feed forward ann
                ann = nl.net.newff([[-1,1]]*M, [l, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
                
                # train network
                train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=round(max_epochs))
                if train_error[-1]<best_train_error:
                    bestnet.append(ann)
                    best_train_error = train_error[-1]
                    error_hist[range(len(train_error)),k] = train_error
            
            y_est = bestnet[k].sim(X_test)
            y_est = (y_est>.5).astype(int)
            errors[k] = (y_est!=y_test).sum().astype(float)/y_test.shape[0]
            
            k+=1
            
        mean_hidden_units.append(np.mean(errors)) 
    min_units = np.argmin(mean_hidden_units)
    
    x_axis = np.arange(1,max_hidden_units+1)
    # Plot the classification error rate
    figure()
    plot(x_axis,np.array(mean_hidden_units))
    xlabel('Number of hidden units')
    ylabel('Classification error rate (%)')
    show()

    return min_units + 1
    

def KNN(X,Y):
    N, M = X.shape
    K = 10  # K-nearest neighbors
    CV = model_selection.KFold(n_splits=K,shuffle=True)
    
    L=40 # Maximum number of neighbors
    
    errors = np.zeros((N,L))
    i=0
    for train_index, test_index in CV.split(X):
        print('Crossvalidation fold: {0}/{1}'.format(i+1,K))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = np.ravel(Y[train_index])
        X_test = X[test_index,:]
        y_test = np.ravel(Y[test_index])
    
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train, y_train);
            y_est = knclassifier.predict(X_test);
            errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
    
        i+=1
        
    # find optimal number of neighbours
    error_mean = sum(errors,0)/K
    min_neighbors = np.argmin(error_mean)
    
    # plot error over number of neighbours
    figure()
    plot(sum(errors,0)/N)
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    show()
    
    return min_neighbors + 1

def DTC(X,Y):
    N, M = X.shape
    K = 10 
    CV = model_selection.KFold(n_splits=K,shuffle=True)
    
    # Tree complexity parameter - constraint on maximum depth
    tc = np.arange(2, 21, 1)
    
    Error_train = np.empty((len(tc),K))
    Error_test = np.empty((len(tc),K))

    k=0
    for train_index, test_index in CV.split(X):
        print('Computing CV fold: {0}/{1}..'.format(k+1,K))
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = np.ravel(Y[train_index])
        X_test = X[test_index,:]
        y_test = np.ravel(Y[test_index])
    
        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train,y_train)
            y_est_test = dtc.predict(X_test)
            y_est_train = dtc.predict(X_train)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = sum(np.abs(y_est_test - y_test)) / float(len(y_est_test))
            misclass_rate_train = sum(np.abs(y_est_train - y_train)) / float(len(y_est_train))
            Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
        k+=1
        
    error_mean = Error_test.mean(1)
    min_depth = np.argmin(error_mean)
    
    # boxplot test error spread in each fold over max tree depth
    f = figure()
    boxplot(Error_test.T)
    xlabel('Model complexity (max tree depth)')
    ylabel('Test error across CV folds, K={0})'.format(K))
    
    # plot test error and training error over max tree depth
    f = figure()
    plot(tc, Error_train.mean(1))
    plot(tc, Error_test.mean(1))
    xlabel('Model complexity (max tree depth)')
    ylabel('Error (misclassification rate, CV K={0})'.format(K))
    legend(['Error_train','Error_test'])
        
    show()
    
    return min_depth + 1

#calculates error of majority class label
def largest_class(X,Y):
    counts = np.bincount(Y)
    largest_class=np.argmax(counts)
    error = 0
    for i in range(len(Y)):
        if abs(Y[i]-largest_class) > 0:
            error+=1
    error = error/float(len(Y))
    return error

# Compares ANN, KNN and majority class predictor and tests if classifiers are 
# significantly different by computing credibility interval
# figure: boxplot comparing performance of ANN, KNN and majority class predictor
def ANN_KNN_compare(ann,knn,majority,K):
    largest_class = np.full((K, ), majority) #majority class predictor array
    
    # convert to array
    annarr = np.array(ann) 
    knnarr = np.array(knn)
    
    #compute credibility interval
    z = (annarr-knnarr)
    zb = z.mean()
    nu = K-1
    sig =  (z-zb).std()  / np.sqrt(K-1)
    alpha = 0.05
    
    zL = zb + sig * stats.t.ppf(alpha/2, nu);
    zH = zb + sig * stats.t.ppf(1-alpha/2, nu);
    
    if zL <= 0 and zH >= 0 :
        print('Classifiers are not significantly different')        
    else:
        print('Classifiers are significantly different.')
        
    # Boxplot to compare classifier error distributions
    figure()
    boxplotarr = [annarr, knnarr, largest_class]
    boxplot(boxplotarr)
    xticks([1,2,3],('ANN','KNN','Largest Class'))
    ylabel('Cross-validation error [%]')
    show()
    


# Outer CV
if __name__ == "__main__":
    # Load data from .data file to a DataFrame
    spamdata = pd.read_csv('spambase.data', sep=",", header=None)
    
    # Set attribute names
    attributeNames = ['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail','word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business','word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!','char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total','spam_or_not']
    spamdata.columns = attributeNames
    
    # Check for data issues i.e. No negativeValues and no missingValues
    missingValues = spamdata.isnull().sum().sum()
    print('Number of missing values = ', missingValues)
    negativeValues = (spamdata < 0).sum().sum()
    print('Number of negative values = ',negativeValues)
    
    # Encode class name with dict
    classLabels  = sorted(set(spamdata['spam_or_not']))
    classNames = ['not_spam','spam']
    classDict = dict(zip(classNames, range(3)))
    
    # Get the class values
    y = np.mat(spamdata.iloc[:,-1]).T #matrix
    y_array = np.array(spamdata.iloc[:,-1]).T
    
    # Preallocate memory, set data to matrix X
    X = spamdata[attributeNames[:-1]].as_matrix()

    # Compute values of N, M and C.
    N = spamdata.shape[0]
    M = spamdata[attributeNames[:-1]].shape[1]
    C = len(classNames)

    # Transform
    X = (X - np.ones((N,1))*X.mean(0))/X.std(0) # normalize
    X = np.log10((X - np.mean(X,axis = 0))/np.std(X,axis = 0)+1) # add 1 to avoid zeros and apply log
    
    # Outer cross validation
    K = 10
    CV = model_selection.KFold(n_splits=K,shuffle=True)
    
    # Optimal parameters for each  fold and corresponding error rates
    opt_param_ann = []
    opt_param_knn = []
    opt_param_dtc = []
    error_ann = []
    error_knn = []
    error_dtc = []
    
    i=0
    for train_index, test_index in CV.split(X):
        print('Crossvalidation fold: {0}/{1}'.format(i+1,K))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index,:]
        X_test = X[test_index,:]
        y_test = y[test_index,:]
    
        # Inner CV for optimal parameters
        opt_param_knn.append(KNN(X_train,y_train))
        opt_param_dtc.append(DTC(X_train,np.ravel(y_train)))
        opt_param_ann.append(ANN(X_train,y_train))
    
        #Outer DTC
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_param_dtc[i])
        dtc = dtc.fit(X_train,np.ravel(y_train))
        y_est_test = dtc.predict(X_test)
        test_error = sum(np.abs(y_est_test - np.ravel(y_test))) / float(len(y_est_test))
        
        error_dtc.append(test_error.item(0))
        
        #Outer KNN
        knclassifier = KNeighborsClassifier(n_neighbors=opt_param_knn[i]);
        knclassifier.fit(X_train, np.ravel(y_train));
        y_est = knclassifier.predict(X_test);
        test_error = sum(np.abs(y_est - np.ravel(y_test))) / float(len(y_est))
        
        error_knn.append(test_error.item(0))
        
        #Outer ANN
        n_train = 2             # number of networks trained in each k-fold
        max_epochs = 64        # stop criterion 2 (max epochs in training)
        
        ann = nl.net.newff([[-1,1]]*M, [opt_param_ann[i], 1], [nl.trans.TanSig(), nl.trans.PureLin()])
        train_error = ann.train(X_train, y_train, epochs=max_epochs, show=round(max_epochs))
        y_est = ann.sim(X_test)
        y_est = (y_est>.5).astype(int)
        test_error = sum(np.abs(y_est - y_test)) / float(len(y_est))
        
        error_ann.append(test_error.item(0))
        
        i+=1
    
    
    # compare ANN and KNN
    ANN_KNN_compare(error_ann,error_knn,largest_class(X,np.ravel(y)),K)
    
    # plot performance comparison
    f = figure()
    plot(range(1,K+1),error_ann)
    plot(range(1,K+1),error_knn)
    plot(range(1,K+1),error_dtc)
    xlabel('Crossvalidation Fold')
    ylabel('Error (misclassification rate, CV K={0})'.format(K))
    legend(['ANN','KNN','DTC'])
    show()