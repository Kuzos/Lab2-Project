# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 01:52:03 2024

@author: krsto
"""

import statistics
import os
import sklearn
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn import preprocessing
# Import the svm module
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef,confusion_matrix,accuracy_score

SVM_file="SVM_Train12345"
Bencmhark_file="Benchmark_set"



def creating_feature(filename,k):
    file = open(filename, 'r')
    
    if "i" in filename:
        
        composition = np.zeros((8329, 20))
    else:
        composition=np.zeros((2088,20))
        
    # print(file_lines[0][0],file_lines[0][1])
    header = file.readline().split(sep=("\t"))
    # print(header)
    seq = "GAVPLIMFWYSTCNQHDEKR"
    matrix_lenght = []
    class_list = []
    # file_lines[get_index]-1 because header is always first line, its for the number of rows in matrix
    
    for index, line in enumerate(file):
        if index < 100000:

            line = line.split(sep=("\t"))  # Seperating the tabs
            sequence = line[5][:k]  # Getting sequence of 25 residues
        # print(sequence)
            true_class = line[6]  # Getting the class for sequence
            if true_class == "Neg":
                class_list.append(0)

            elif true_class == "Pos":
                class_list.append(1)

            else:
                print("error with class", index)

            c = 0
            for letter in sequence:

                if letter in seq:
                    search_seq = seq.find(letter)  # Checking letters a

                    # Writing in the good order of matrix
                    composition[index, search_seq] += 1
                else:
                    c += 1  # Setting counter for proper lenght

                    continue
            matrix_lenght.append(k-c)
            # print(letter,search_seq)
        else:
            break
    #print("this is lenght of the matrix_lenght", len(matrix_lenght), "this is y_class", len(class_list))
    for index, lenght in enumerate(matrix_lenght):
        composition[index] = composition[index]/lenght
    summation = np.sum(composition, axis=1)
    print(summation, "values should be 1")
    #rint(np.shape(composition),"this is shape of composition matrix ")
    y_class=np.array(class_list)
    file.close()
    return composition, y_class
# print(index,sequence)
# print(matrix_lenght[-1])
#print(composition[-1], "last row")
#print(composition[0],"first row")


#print(composition[581], "last row comp")
#print(composition[0],"first row comp")


def hydrophobicity(filename, k):

   if "i" in filename:
       
       matrix_phobicity= np.zeros((8329, 3))
   else:
       
       matrix_phobicity= np.zeros((2088, 3))
       

   kd = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
          "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
          "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
          "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2, "X": 0, 'Z': -3.5}
   file = open(filename, 'r')
   header = file.readline().split(sep=("\t"))
   scores = []
   for index, line in enumerate(file):
       
       if index < 10000000:
           

           line = line.split(sep=("\t"))
           sequence = "XX"+line[5][:k]+"XX"  # adding for the sliding window!
           pa = ProteinAnalysis(sequence)
           hp = pa.protein_scale(kd, 5)
           scores.append(hp)  # appending scores as list of lists

       else:
           raise ValueError("Error in lenght of file")
    #print("THIS IS HP")
    #print(hp)
    #print(scores[-2],"Last seq")
    #print(scores[-1],"lasst seq -1")

    #print(len(hp), len(hp), "This is HP and lenght of HP! which should be same as K")
    
   
   for index, each in enumerate(scores):
       
        #print(each," this is each from scores")
        
       highest_value = max(each)
        #print(highest_value, "highest value from each from scores")
       avg_value = sum(each)/len(each)
        # print(avg)
       pos_of_max = each.index(highest_value)
        #print(pos_of_max, "position of max")
       matrix_phobicity[index, 0] = highest_value
       matrix_phobicity[index, 1] = avg_value
       matrix_phobicity[index, 2] = pos_of_max
        
       min_val_first = np.min(matrix_phobicity[:, 0])
       max_val_first = np.max(matrix_phobicity[:, 0])
       normalized_first_column = (matrix_phobicity[:, 0] - min_val_first) / (max_val_first - min_val_first)

    # Normalize the second column using Min-Max scaling
       min_val_second = np.min(matrix_phobicity[:, 1])
       max_val_second = np.max(matrix_phobicity[:, 1])
       normalized_second_column = (matrix_phobicity[:, 1] - min_val_second) / (max_val_second - min_val_second)
       
       #Normalize the third column
       min_val_third = np.min(matrix_phobicity[:, 2])
       max_val_third = np.max(matrix_phobicity[:, 2])
       normalized_third_column = (matrix_phobicity[:, 2] - min_val_third) / (max_val_third - min_val_third)
    # Replace the first and second columns in the feature matrix with the normalized values
       normalized_feature_matrix = np.column_stack((normalized_first_column, normalized_second_column,normalized_third_column))
        #print(normalized_feature_matrix)
    #print(each.index(highest_value),"this is position of the without divide by k")
    #print(highest_value, "highest value from each from scores")
    #print(each," this is each from scores")
    #print(pos_of_max, "position of max")
    # print(matrix_phobicity)
   return normalized_feature_matrix


def concatenate_matrices(matrix1, matrix2):
    """
    Concatenate two matrices along the second axis.

    Parameters:
        matrix1 (numpy.ndarray): First matrix to concatenate.
        matrix2 (numpy.ndarray): Second matrix to concatenate.

    Returns:
        numpy.ndarray: Concatenated matrix.
    """
    if matrix1.shape[0] != matrix2.shape[0]:
        raise ValueError("Matrices must have the same number of rows for concatenation.")

    return np.concatenate((matrix1, matrix2), axis=1)


if __name__ == "__main__":
    
    training_matrix,training_class=creating_feature(SVM_file, k=20)
    feature1=hydrophobicity(SVM_file,k=20)
    final_matrix =concatenate_matrices(training_matrix,feature1)
    
    benchmark_matrix,b_class=creating_feature(Bencmhark_file, k=20)
    benchmark_feature=hydrophobicity(Bencmhark_file, k=20)
    benchmark_set=concatenate_matrices(benchmark_matrix,benchmark_feature)
    
    
    mySVC = svm.SVC(C=2, kernel='rbf', gamma=2)
    # Train (fit) the model on training data
    mySVC.fit(final_matrix, training_class)
    #mySVC.fit(final_matrix, training_class)
    # Predict classes on testing data
                
    y_pred = mySVC.predict(benchmark_set)
    
    mcc = matthews_corrcoef(b_class, y_pred)
    conf_matrix = confusion_matrix(b_class, y_pred)
    precision = precision_score(b_class, y_pred)
    recall = recall_score(b_class, y_pred)
    f1 = f1_score(b_class, y_pred)
    accuracy=accuracy_score(b_class, y_pred)
    print("Accuracy",accuracy,"Recall",recall,"PRECISION",precision,"f1",f1,"MCC",mcc)
    print(conf_matrix)
    false_positives = np.where((b_class == 0) & (y_pred == 1))[0]
    false_negatives = np.where((b_class == 1) & (y_pred == 0))[0]

    print(f'Indices of False Positives (FP): {false_positives}')
    print(f'Indices of False Negatives (FN): {false_negatives}')

    # Example to show misclassified examples
    print('Examples of False Positives:')
    for idx in false_positives:
        print(f'Index: {idx}, True Label: {b_class[idx]}, Predicted Label: {y_pred[idx]}')

    print('Examples of False Negatives:')
    for idx in false_negatives:
        print(f'Index: {idx}, True Label: {b_class[idx]}, Predicted Label: {y_pred[idx]}')
    
    