import pandas as pd
import numpy
import os
import joblib as jl
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

maxFrame = 232

def build_featureVector(filepath):  # build a feature vector with multiple features from ONE CSV file
    ################################# STEP 0： Read rawData ####################################################
    # rawData = pd.read_csv(r'C:\Users\Dunchuan\Desktop\CSV\fun\FUN_1_BAKRE.csv', sep=',', skiprows=1, header=None)
    rawData = pd.read_csv(filepath, sep=',', skiprows=1, header=None)

    left_wrist_X = rawData[30]
    left_wrist_Y = rawData[31]
    right_wrist_X = rawData[33]
    right_wrist_Y = rawData[34]
    left_eye_X = rawData[6]
    left_eye_Y = rawData[7]
    right_eye_X = rawData[9]
    right_eye_Y = rawData[10]
    nose_X = rawData[3]
    nose_Y = rawData[4]
    left_shoulder_X = rawData[18]
    left_shoulder_Y = rawData[19]
    right_shoulder_X = rawData[21]
    right_shoulder_Y = rawData[22]

    #################################### STEP 1：Universal Normalization ########################
    # X normalization for wrist #
    left_wrist_X_normalized = (left_wrist_X - nose_X) / abs(left_eye_X - right_eye_X)
    right_wrist_X_normalized = (right_wrist_X - nose_X) / abs(left_eye_X - right_eye_X)

    # Y normalization for wrist #
    left_wrist_Y_normalized = (left_wrist_Y - nose_Y) / abs(nose_Y - left_shoulder_Y)
    right_wrist_Y_normalized = (right_wrist_Y - nose_Y) / abs(nose_Y - right_shoulder_Y)

    ##################################### STEP 2: Extract Features ################################
    #### The first 2 features: right_wrist_X_normalized_filled, right_wrist_Y_normalized_filled ##########
    right_wrist_X_normalized_filled = right_wrist_X_normalized
    right_wrist_Y_normalized_filled = right_wrist_Y_normalized

    if len(right_wrist_X_normalized) != maxFrame:
        for i in range(maxFrame - len(right_wrist_X_normalized)):
            right_wrist_X_normalized_filled = numpy.append(right_wrist_X_normalized_filled, [0])

    if len(right_wrist_Y_normalized) != maxFrame:
        for i in range(maxFrame - len(right_wrist_Y_normalized)):
            right_wrist_Y_normalized_filled = numpy.append(right_wrist_Y_normalized_filled, [0])

    ##### The next 3 features: right_wrist_X_Difference_filled, right_wrist_Y_Difference_filled, slopeArray#####################
    right_wrist_X_Difference = numpy.diff(right_wrist_X_normalized)
    right_wrist_Y_Difference = numpy.diff(right_wrist_Y_normalized)

    right_wrist_X_Difference_filled = right_wrist_X_Difference
    right_wrist_Y_Difference_filled = right_wrist_Y_Difference

    if len(right_wrist_X_Difference_filled) != maxFrame:
        for i in range(maxFrame - len(right_wrist_X_Difference_filled)):
            right_wrist_X_Difference_filled = numpy.append(right_wrist_X_Difference_filled, [0])
    if len(right_wrist_Y_Difference_filled) != maxFrame:
        for i in range(maxFrame - len(right_wrist_Y_Difference_filled)):
            right_wrist_Y_Difference_filled = numpy.append(right_wrist_Y_Difference_filled, [0])

    slopeArray = right_wrist_Y_Difference / right_wrist_X_Difference
    slopeArray_filled = slopeArray
    if len(slopeArray_filled) != maxFrame:
        for i in range(maxFrame - len(slopeArray_filled)):
            slopeArray_filled = numpy.append(slopeArray_filled, [0])
    ################# The last 2 features: zeroCrossingArray, maxDiffArray#####################
    zeroCrossingArray = numpy.array([])
    maxDiffArray = numpy.array([])

    if right_wrist_Y_Difference[0] > 0:
        initSign = 1
    else:
        initSign = 0

    windowSize = 5;
    for x in range(1, len(right_wrist_Y_Difference)):
        if right_wrist_Y_Difference[x] > 0:
            newSign = 1
        else:
            newSign = 0

        if initSign != newSign:
            zeroCrossingArray = numpy.append(zeroCrossingArray, x)
            initSign = newSign
            maxIndex = numpy.minimum(len(right_wrist_Y_Difference), x + windowSize)
            minIndex = numpy.maximum(0, x - windowSize)
            maxVal = numpy.amax(right_wrist_Y_Difference[minIndex:maxIndex])
            minVal = numpy.amin(right_wrist_Y_Difference[minIndex:maxIndex])
            maxDiffArray = numpy.append(maxDiffArray, (maxVal - minVal))
    index = numpy.argsort(-maxDiffArray)


    ##################### Plus 5 features for left hand similar to right hand ##########################
    left_wrist_X_normalized_filled = left_wrist_X_normalized
    left_wrist_Y_normalized_filled = left_wrist_Y_normalized

    if len(left_wrist_X_normalized) != maxFrame:
        for i in range(maxFrame - len(left_wrist_X_normalized)):
            left_wrist_X_normalized_filled = numpy.append(left_wrist_X_normalized_filled, [0])

    if len(left_wrist_Y_normalized) != maxFrame:
        for i in range(maxFrame - len(left_wrist_Y_normalized)):
            left_wrist_Y_normalized_filled = numpy.append(left_wrist_Y_normalized_filled, [0])

    ##### The next 3 features: right_wrist_X_Difference_filled, right_wrist_Y_Difference_filled, slopeArray#####################
    right_wrist_X_Difference = numpy.diff(right_wrist_X_normalized)
    right_wrist_Y_Difference = numpy.diff(right_wrist_Y_normalized)

    right_wrist_X_Difference_filled = right_wrist_X_Difference
    right_wrist_Y_Difference_filled = right_wrist_Y_Difference

    if len(right_wrist_X_Difference_filled) != maxFrame:
        for i in range(maxFrame - len(right_wrist_X_Difference_filled)):
            right_wrist_X_Difference_filled = numpy.append(right_wrist_X_Difference_filled, [0])
    if len(right_wrist_Y_Difference_filled) != maxFrame:
        for i in range(maxFrame - len(right_wrist_Y_Difference_filled)):
            right_wrist_Y_Difference_filled = numpy.append(right_wrist_Y_Difference_filled, [0])

    ############## Build left hand slope and 2 params########################################
    ##### The next 3 features: left_wrist_X_Difference_filled, left_wrist_Y_Difference_filled, slopeArray2#####################
    left_wrist_X_Difference = numpy.diff(left_wrist_X_normalized)
    left_wrist_Y_Difference = numpy.diff(left_wrist_Y_normalized)

    left_wrist_X_Difference_filled = left_wrist_X_Difference
    left_wrist_Y_Difference_filled = left_wrist_Y_Difference

    if len(left_wrist_X_Difference_filled) != maxFrame:
        for i in range(maxFrame - len(left_wrist_X_Difference_filled)):
            left_wrist_X_Difference_filled = numpy.append(left_wrist_X_Difference_filled, [0])
    if len(left_wrist_Y_Difference_filled) != maxFrame:
        for i in range(maxFrame - len(left_wrist_Y_Difference_filled)):
            left_wrist_Y_Difference_filled = numpy.append(left_wrist_Y_Difference_filled, [0])

    # slopeArray for left hand
    slopeArray2 = left_wrist_Y_Difference / left_wrist_X_Difference
    slopeArray2_filled = slopeArray2
    if len(slopeArray2_filled) != maxFrame:
        for i in range(maxFrame - len(slopeArray2_filled)):
            slopeArray2_filled = numpy.append(slopeArray2_filled, [0])
    # print(len(left_wrist_X_Difference),len(left_wrist_Y_Difference),len(slopeArray2))

    featureVector1 = numpy.array([])  ######## Empty Vector. Needs 12 Features to Form a featureVector
    # print(featureVector.shape)
    featureVector1 = numpy.append(featureVector1, right_wrist_X_normalized_filled)
    featureVector1 = numpy.append(featureVector1, right_wrist_Y_normalized_filled)
    featureVector1 = numpy.append(featureVector1, left_wrist_X_normalized_filled)
    featureVector1 = numpy.append(featureVector1, left_wrist_Y_normalized_filled)
    featureVector1 = numpy.append(featureVector1, slopeArray_filled)
    featureVector1 = numpy.append(featureVector1, slopeArray2_filled)

    featureVector2 = numpy.array([])
    featureVector2 = numpy.append(featureVector2, right_wrist_X_normalized_filled)
    featureVector2 = numpy.append(featureVector2, right_wrist_Y_normalized_filled)
    featureVector2 = numpy.append(featureVector2, left_wrist_X_normalized_filled)
    featureVector2 = numpy.append(featureVector2, left_wrist_Y_normalized_filled)
    featureVector2 = numpy.append(featureVector2, right_wrist_Y_Difference_filled)
    featureVector2 = numpy.append(featureVector2, left_wrist_Y_Difference_filled)
    featureVector2 = numpy.append(featureVector2, maxDiffArray[index[0:3]])

    featureVector3 = numpy.array([])
    featureVector3 = numpy.append(featureVector3, right_wrist_Y_normalized_filled)
    featureVector3 = numpy.append(featureVector3, left_wrist_Y_normalized_filled)

    return featureVector1,featureVector2,featureVector3

trainlabel = numpy.array([])
traintotal = 0

path = r"C:\Users\37026\Desktop\assignment2\CSV" #改一下路径

for file in os.listdir(path):
    print(os.path.join(path, file))
    for name in os.listdir(os.path.join(path, file)):
        cvs = os.path.join(os.path.join(path, file), name)
        featureVector1,featureVector2,featureVector3 = build_featureVector(cvs)  # 返回vector
        trainlabel = numpy.append(trainlabel, file)  # 添加到label
        if traintotal == 0:
            trainfeatureMatrix1 = numpy.array([featureVector1])
            trainfeatureMatrix2 = numpy.array([featureVector2])
            trainfeatureMatrix3 = numpy.array([featureVector3])
            traintotal = traintotal + 1
        else:
            trainfeatureMatrix1 = numpy.concatenate((trainfeatureMatrix1, [featureVector1]), axis=0)  # 添加到matrix
            trainfeatureMatrix2 = numpy.concatenate((trainfeatureMatrix2, [featureVector2]), axis=0)  # 添加到matrix
            trainfeatureMatrix3 = numpy.concatenate((trainfeatureMatrix3, [featureVector3]), axis=0)  # 添加到matrix

ramodel=RandomForestClassifier()
ramodel.fit(trainfeatureMatrix1,trainlabel)

lrmodel=LogisticRegression()
lrmodel.fit(trainfeatureMatrix2,trainlabel)

dtmodel=DecisionTreeClassifier(max_leaf_nodes=6)
dtmodel.fit(trainfeatureMatrix3,trainlabel)

clf = svm.SVC()
clf.fit(trainfeatureMatrix3,trainlabel)

jl.dump(ramodel, 'C:/Users/37026/Desktop/assignment2/modelmodel/model_1.pkl')
jl.dump(lrmodel, 'C:/Users/37026/Desktop/assignment2/modelmodel/model_2.pkl')
jl.dump(dtmodel, 'C:/Users/37026/Desktop/assignment2/modelmodel/model_3.pkl')
jl.dump(clf, 'C:/Users/37026/Desktop/assignment2/modelmodel/model_4.pkl')

print('save successful')

