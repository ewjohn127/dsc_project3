import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix,plot_confusion_matrix, roc_auc_score, plot_roc_curve, roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import graphviz 

#Function to plot ROC curves of every model
def all_roc_curves(X, y, dummy, logreg, dtree, rforest, lr_keep_list, dt_keep_list, rf_keep_list):

    #Get the FPRs and TPRs of each model
    dc_test_fpr, dc_test_tpr, dc_test_thresholds = roc_curve(y, dummy.predict_proba(X)[:, 1])
    lr_test_fpr, lr_test_tpr, lr_test_thresholds = roc_curve(y, logreg.predict_proba(X[lr_keep_list[0]])[:, 1])
    dt_test_fpr, dt_test_tpr, dt_test_thresholds = roc_curve(y, dtree.predict_proba(X[dt_keep_list[4]])[:, 1])
    RF_test_fpr, RF_test_tpr, RF_test_thresholds = roc_curve(y, rforest.predict_proba(X[rf_keep_list[19]])[:, 1])

    plt.figure(figsize=(10, 8))
    lw = 2

    #Plot each curve
    plt.plot(dc_test_fpr, dc_test_tpr, color='blue',
            lw=lw, label='Dummy Regressor ROC curve')
    plt.plot(lr_test_fpr, lr_test_tpr, color='darkorange',
            lw=lw, label='Logistic Regressor ROC curve')
    plt.plot(dt_test_fpr, dt_test_tpr, color='red',
            lw=lw, label='Decision Tree ROC curve')
    plt.plot(RF_test_fpr, RF_test_tpr, color='green',
            lw=lw, label='Random Forest ROC curve')

    #Decorations
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    return plt.show()


#Function to plot decision tree with depth of 2
def tree(model, keep_list, y):
        dot_data = tree.export_graphviz(model, out_file=None,
                              max_depth = 2,  
                              feature_names=keep_list[4],  
                              class_names=np.unique(y).astype('str'),  
                              filled=True, rounded=True,  
                              special_characters=True)  
        graph = graphviz.Source(dot_data)  
        graph 
        # graph.format = 'png'
        # graph.render('dtree_render',view=True)
    

#Function to plot roc scores as the number of features in the model increase
def roc_score_plot(n_features, cv_rfe):

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(range(1,n_features+1), cv_rfe)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Mean Cross Val ROC AUC Score for Random Forest')
    
    return plt.show()

