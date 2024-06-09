# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 01:03:07 2024

@author: krsto
"""

from sys import argv
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef,confusion_matrix,accuracy_score
import statistics

seq = "GAVPLIMFWYSTCNQHDEKR"
Benchmark="Benchmark_set"
PSWMN="PSWM_12345.npy"



def testing(testing_set,Train_matrix,th):
     
     fajlerino=open(testing_set, 'r')
     fajlerino.readline()
     entries=[]
     true_classification=[]
     values=[]
     Train_matrix=np.load(Train_matrix)
     
     
     for index, line in enumerate(fajlerino):
         if index < 10000000:
             wline = []
             line = line.split(sep='\t')
             
             sequence = line[5]
             Uniprot_code = line[0]
             Clevage_seq = line[9].rstrip()
             
             target_sequence = sequence[:90]
             lenght_target = len(target_sequence)
             scores = []
             
             for number in range(lenght_target - 15):
                 sliding_seq = target_sequence[number:15 + number]
                 value=0
                 for c,letter in enumerate(sliding_seq):
                     search_seq = seq.find(letter)
                     if search_seq==-1:
                         #print("error",Uniprot_code,line[6],letter)
                         continue
                     value += Train_matrix[search_seq, c]
                 scores.append(value)
                 
             entries.append(Uniprot_code)
             
             values.append(float(max(scores)))
             
             if line[6] == "Pos":
                 true_classification.append(1)
                 
             elif line[6] == "Neg":
                 true_classification.append(0)
             else:
                 print("There is some error here", Uniprot_code)
                 
             #print(wline)
             #print("this is the max score", max(scores))
             #print("this is lenght of target seq",lenght_target,"this is lenght of list_scores", len(scores))
             #print("this is len of values",len(values))
             #print("this is len of classification", len(true_classification))
             
         else:
             break
     
         #else:
           # break
     #print("this is values",values)
     testing_class=np.array(true_classification)
     y_test_scores = np.array(values)
     y_pred_test = np.array([int(t_s >= th) for t_s in y_test_scores])   # interesting way to store boolean with int this is list that stores if its True put 1 in list if False put 0
     
     precision = precision_score(testing_class, y_pred_test)
     recall = recall_score(testing_class, y_pred_test)
     f1 = f1_score(testing_class, y_pred_test)
     mcc = matthews_corrcoef(testing_class, y_pred_test)
     conf_matrix = confusion_matrix(testing_class, y_pred_test)
     accuracy=accuracy_score(testing_class, y_pred_test)    
         
     
     
     #file2.close()
     return precision,recall,f1,mcc,conf_matrix,accuracy,testing_class,y_pred_test
 
    
 
if __name__=="__main__":
    precision,recall,f1,mcc,conf_matrix,acc,y_true,y_pred=testing(Benchmark,PSWMN,th=6.258323631069117)
    print("Accuracy",acc,"Recall",recall,"PRECISION",precision,"f1",f1,"MCC",mcc)
    print(conf_matrix)
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]

    print(f'Indices of False Positives (FP): {false_positives}')
    print(f'Indices of False Negatives (FN): {false_negatives}')

    # Example to show misclassified examples
    print('Examples of False Positives:')
    for idx in false_positives:
        print(f'Index: {idx}, True Label: {y_true[idx]}, Predicted Label: {y_pred[idx]}')

    print('Examples of False Negatives:')
    for idx in false_negatives:
        print(f'Index: {idx}, True Label: {y_true[idx]}, Predicted Label: {y_pred[idx]}')
    