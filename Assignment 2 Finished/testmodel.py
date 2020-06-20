import pandas as pd
import numpy
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

maxFrame = 232

def normalize01(data):
	data = ( data - numpy.min(data) ) / ( numpy.max(data)  - numpy.min(data) );
	return data;

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
    # print(maxFrame == len(right_wrist_X_normalized_filled))
    # print("x filled is ", len(right_wrist_X_normalized_filled))

    if len(right_wrist_Y_normalized) != maxFrame:
        for i in range(maxFrame - len(right_wrist_Y_normalized)):
            right_wrist_Y_normalized_filled = numpy.append(right_wrist_Y_normalized_filled, [0])
    # print(maxFrame == len(right_wrist_Y_normalized_filled))
    # print("y filled is ", len(right_wrist_Y_normalized_filled))

    # print(len(right_wrist_Y_normalized_filled), len(right_wrist_X_normalized_filled))

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
    # print(len(right_wrist_X_Difference),len(right_wrist_Y_Difference),len(slopeArray))

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

    # print("zeroCrossing and maxDiff length are", len(zeroCrossingArray[index[0:5]]), len(maxDiffArray[index[0:5]]))

    ##################### Plus 5 features for left hand similar to right hand ##########################
    left_wrist_X_normalized_filled = left_wrist_X_normalized
    left_wrist_Y_normalized_filled = left_wrist_Y_normalized

    if len(left_wrist_X_normalized) != maxFrame:
        for i in range(maxFrame - len(left_wrist_X_normalized)):
            left_wrist_X_normalized_filled = numpy.append(left_wrist_X_normalized_filled, [0])
    # print(maxFrame == len(left_wrist_X_normalized_filled))
    # print("x filled is ", len(left_wrist_X_normalized_filled))

    if len(left_wrist_Y_normalized) != maxFrame:
        for i in range(maxFrame - len(left_wrist_Y_normalized)):
            left_wrist_Y_normalized_filled = numpy.append(left_wrist_Y_normalized_filled, [0])
    # print(maxFrame == len(left_wrist_Y_normalized_filled))
    # print("y filled is ", len(left_wrist_Y_normalized_filled))

    # print(len(left_wrist_Y_normalized_filled), len(left_wrist_X_normalized_filled))

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

    ######## Empty Vector. Needs 12 Features to Form a featureVector
    featureVector = numpy.array([])  
    # print(featureVector.shape)
    #featureVector = numpy.append(featureVector, right_wrist_X_normalized_filled)
    featureVector = numpy.append(featureVector, right_wrist_Y_normalized_filled)
    #featureVector = numpy.append(featureVector, left_wrist_X_normalized_filled)
    featureVector = numpy.append(featureVector, left_wrist_Y_normalized_filled)
    
    #featureVector = numpy.append(featureVector, right_wrist_X_Difference_filled)  ########## fill to maxFrame size
    featureVector = numpy.append(featureVector, right_wrist_Y_Difference_filled)  ########## fill to maxFrame size
    #featureVector = numpy.append(featureVector, left_wrist_X_Difference_filled)  ########## fill to maxFrame size
    #featureVector = numpy.append(featureVector, left_wrist_Y_Difference_filled)  ########## fill to maxFrame size
    #featureVector = numpy.append(featureVector, slopeArray_filled)
    #featureVector = numpy.append(featureVector, slopeArray2_filled)
    
    #featureVector = numpy.append(featureVector, zeroCrossingArray[index[0:3]])  
    #featureVector = numpy.append(featureVector, maxDiffArray[index[0:3]])  

    '''
    #featureVector = numpy.append(featureVector, right_wrist_X_normalized_filled)  ###### fill to maxFrame size
    # print(featureVector.shape)

    #featureVector = numpy.append(featureVector, right_wrist_Y_normalized_filled)  ###### fill to maxFrame size
    # print(featureVector.shape)

    #featureVector = numpy.append(featureVector, right_wrist_X_Difference_filled)  ########## fill to maxFrame size
    # print(featureVector.shape)

    #featureVector = numpy.append(featureVector, right_wrist_Y_Difference_filled)  ########## fill to maxFrame size
    # print(featureVector.shape)

    #featureVector = numpy.append(featureVector, slopeArray_filled)  ######################## fill to maxFrame size
    # print(featureVector.shape)

    ###################################### __________________________ ###################################
    #featureVector = numpy.append(featureVector, left_wrist_X_normalized_filled)  ###### fill to maxFrame size
    # print(featureVector.shape)

    #featureVector = numpy.append(featureVector, left_wrist_Y_normalized_filled)  ###### fill to maxFrame size
    # print(featureVector.shape)

    #featureVector = numpy.append(featureVector, left_wrist_X_Difference_filled)  ########## fill to maxFrame size
    # print(featureVector.shape)

    #featureVector = numpy.append(featureVector, left_wrist_Y_Difference_filled)  ########## fill to maxFrame size
    # print(featureVector.shape)

    #featureVector = numpy.append(featureVector, slopeArray2_filled)  ######################## fill to maxFrame size
    # print(featureVector.shape)

    ################################### Might drop these 2 features, depending on accruracy ################

    #featureVector = numpy.append(featureVector, zeroCrossingArray[index[0:5]])  ##### size of 5
    # print(featureVector.shape)
    #featureVector = numpy.append(featureVector, maxDiffArray[index[0:5]])  ######### size of 5
    # print(featureVector.shape)

    maxLen = 2500
    if len(featureVector) < maxLen:
        featureVector = numpy.pad(featureVector, (0, maxLen - len(featureVector)), 'constant')

    #return featureVectorMother[0:150]
    return featureVector[0:maxLen]
    '''
    return featureVector




trainlabel = numpy.array([])
traintotal = 0
testlabel = numpy.array([])
testtotal = 0
#maxFrame = 232
path = r"C:\Users\Dunchuan\Desktop\CSV" #改一下路径

for file in os.listdir(path):
	# print(os.path.join(path, file))
	count = 0
	for name in os.listdir(os.path.join(path, file)):
		cvs = os.path.join(os.path.join(path, file), name) #cvs filename 
		featureVector = build_featureVector(cvs)  # return vector
		if count < 60: # take the first 60 samples to train
			trainlabel = numpy.append(trainlabel, file)  # add to label
			if traintotal==0:
				trainfeatureMatrix=numpy.array([featureVector])
				traintotal=traintotal+1
			else:
				trainfeatureMatrix = numpy.concatenate((trainfeatureMatrix, [featureVector]), axis=0)#添加到matrix

			count += 1
		else: # the rest is for testing
			testlabel=numpy.append(testlabel, file)
			if testtotal == 0:
				testfeatureMatrix = numpy.array([featureVector])
				testtotal = testtotal + 1
			else:
				testfeatureMatrix = numpy.concatenate((testfeatureMatrix, [featureVector]), axis=0)  # 添加到matrix

clf = svm.SVC()
clf.fit(trainfeatureMatrix,trainlabel)

ramodel=RandomForestClassifier()
ramodel.fit(trainfeatureMatrix,trainlabel)

knnmodel=neighbors.KNeighborsClassifier(n_neighbors=6)
knnmodel.fit(trainfeatureMatrix,trainlabel)

lrmodel=LogisticRegression()
lrmodel.fit(trainfeatureMatrix,trainlabel)

gnbmodel=GaussianNB()
gnbmodel.fit(trainfeatureMatrix,trainlabel)

dtmodel=DecisionTreeClassifier(max_leaf_nodes=6)
dtmodel.fit(trainfeatureMatrix,trainlabel)

print('\nSVM.Score: %.2f' % clf.score(testfeatureMatrix, testlabel))
print('\nRamdomForest.Score: %.2f' % ramodel.score(testfeatureMatrix, testlabel))
print('\nKNN.Score: %.2f' % knnmodel.score(testfeatureMatrix, testlabel))
print('\nLR.Score: %.2f' % lrmodel.score(testfeatureMatrix, testlabel))
print('\nNB.Score: %.2f' % gnbmodel.score(testfeatureMatrix, testlabel))
print('\nDTree.Score: %.2f' % dtmodel.score(testfeatureMatrix, testlabel))

