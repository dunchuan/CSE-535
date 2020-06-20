import pandas as pd
import numpy
import matplotlib.pyplot as plt
import os
from sklearn import svm

def getMaxFrame(rootdir):
	maxFrame = float("-INF")
	for parent,dirnames,filenames in os.walk(rootdir):
		for filename in filenames:
			rawDataTemp=pd.read_csv(os.path.join(parent,filename), sep=',', skiprows=1, header=None)
			# print(filename, len(rawDataTemp)) # FUN_1_BAKRE.csv 150 
			if len(rawDataTemp) > maxFrame:
				maxFrame = len(rawDataTemp)
	#print(maxFrame)
	return maxFrame

def build_featureVector(rawData, maxFrame): # build a feature vector with multiple features from ONE CSV file	
	################################# STEP 0： Read rawData ####################################################
	# rawData = pd.read_csv(r'C:\Users\Dunchuan\Desktop\CSV\fun\FUN_1_BAKRE.csv', sep=',', skiprows=1, header=None)
	
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
			right_wrist_X_normalized_filled = numpy.append(right_wrist_X_normalized_filled,[0]) 
	#print(maxFrame == len(right_wrist_X_normalized_filled))
	#print("x filled is ", len(right_wrist_X_normalized_filled))
	
	if len(right_wrist_Y_normalized) != maxFrame:
		for i in range(maxFrame - len(right_wrist_Y_normalized)):
			right_wrist_Y_normalized_filled = numpy.append(right_wrist_Y_normalized_filled,[0])
	#print(maxFrame == len(right_wrist_Y_normalized_filled))
	#print("y filled is ", len(right_wrist_Y_normalized_filled))
	
	#print(len(right_wrist_Y_normalized_filled), len(right_wrist_X_normalized_filled))

	##### The next 3 features: right_wrist_X_Difference_filled, right_wrist_Y_Difference_filled, slopeArray#####################
	right_wrist_X_Difference = numpy.diff(right_wrist_X_normalized)
	right_wrist_Y_Difference = numpy.diff(right_wrist_Y_normalized)
	
	right_wrist_X_Difference_filled = right_wrist_X_Difference
	right_wrist_Y_Difference_filled = right_wrist_Y_Difference
	
	if len(right_wrist_X_Difference_filled) != maxFrame:
		for i in range(maxFrame - len(right_wrist_X_Difference_filled)):	
			right_wrist_X_Difference_filled = numpy.append(right_wrist_X_Difference_filled,[0]) 
	if len(right_wrist_Y_Difference_filled) != maxFrame:
		for i in range(maxFrame - len(right_wrist_Y_Difference_filled)):	
			right_wrist_Y_Difference_filled = numpy.append(right_wrist_Y_Difference_filled,[0]) 
		
	slopeArray = right_wrist_Y_Difference / right_wrist_X_Difference
	slopeArray_filled = slopeArray
	if len(slopeArray_filled) != maxFrame:
		for i in range(maxFrame - len(slopeArray_filled)):	
			slopeArray_filled = numpy.append(slopeArray_filled,[0]) 
	#print(len(right_wrist_X_Difference),len(right_wrist_Y_Difference),len(slopeArray))
	
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
	
	#print("zeroCrossing and maxDiff length are", len(zeroCrossingArray[index[0:5]]), len(maxDiffArray[index[0:5]]))
		
	##################### Plus 5 features for left hand similar to right hand ##########################
	left_wrist_X_normalized_filled = left_wrist_X_normalized
	left_wrist_Y_normalized_filled = left_wrist_Y_normalized
	
	if len(left_wrist_X_normalized) != maxFrame:
		for i in range(maxFrame - len(left_wrist_X_normalized)):	
			left_wrist_X_normalized_filled = numpy.append(left_wrist_X_normalized_filled,[0]) 
	#print(maxFrame == len(left_wrist_X_normalized_filled))
	#print("x filled is ", len(left_wrist_X_normalized_filled))
	
	if len(left_wrist_Y_normalized) != maxFrame:
		for i in range(maxFrame - len(left_wrist_Y_normalized)):
			left_wrist_Y_normalized_filled = numpy.append(left_wrist_Y_normalized_filled,[0])
	#print(maxFrame == len(left_wrist_Y_normalized_filled))
	#print("y filled is ", len(left_wrist_Y_normalized_filled))
	
	#print(len(left_wrist_Y_normalized_filled), len(left_wrist_X_normalized_filled))
	
	##### The next 3 features: right_wrist_X_Difference_filled, right_wrist_Y_Difference_filled, slopeArray#####################
	right_wrist_X_Difference = numpy.diff(right_wrist_X_normalized)
	right_wrist_Y_Difference = numpy.diff(right_wrist_Y_normalized)
	
	right_wrist_X_Difference_filled = right_wrist_X_Difference
	right_wrist_Y_Difference_filled = right_wrist_Y_Difference
	
	if len(right_wrist_X_Difference_filled) != maxFrame:
		for i in range(maxFrame - len(right_wrist_X_Difference_filled)):	
			right_wrist_X_Difference_filled = numpy.append(right_wrist_X_Difference_filled,[0]) 
	if len(right_wrist_Y_Difference_filled) != maxFrame:
		for i in range(maxFrame - len(right_wrist_Y_Difference_filled)):	
			right_wrist_Y_Difference_filled = numpy.append(right_wrist_Y_Difference_filled,[0]) 
	

	############## Build left hand slope and 2 params########################################
	##### The next 3 features: left_wrist_X_Difference_filled, left_wrist_Y_Difference_filled, slopeArray2#####################
	left_wrist_X_Difference = numpy.diff(left_wrist_X_normalized)
	left_wrist_Y_Difference = numpy.diff(left_wrist_Y_normalized)
	
	left_wrist_X_Difference_filled = left_wrist_X_Difference
	left_wrist_Y_Difference_filled = left_wrist_Y_Difference
	
	if len(left_wrist_X_Difference_filled) != maxFrame:
		for i in range(maxFrame - len(left_wrist_X_Difference_filled)):	
			left_wrist_X_Difference_filled = numpy.append(left_wrist_X_Difference_filled,[0]) 
	if len(left_wrist_Y_Difference_filled) != maxFrame:
		for i in range(maxFrame - len(left_wrist_Y_Difference_filled)):	
			left_wrist_Y_Difference_filled = numpy.append(left_wrist_Y_Difference_filled,[0]) 
	
	# slopeArray for left hand
	slopeArray2 = left_wrist_Y_Difference / left_wrist_X_Difference
	slopeArray2_filled = slopeArray2
	if len(slopeArray2_filled) != maxFrame:
		for i in range(maxFrame - len(slopeArray2_filled)):	
			slopeArray2_filled = numpy.append(slopeArray2_filled,[0]) 
	#print(len(left_wrist_X_Difference),len(left_wrist_Y_Difference),len(slopeArray2))

	featureVector = numpy.array([]) ######## Empty Vector. Needs 12 Features to Form a featureVector
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, right_wrist_X_normalized_filled) ###### fill to maxFrame size
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, right_wrist_Y_normalized_filled) ###### fill to maxFrame size
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, right_wrist_X_Difference_filled) ########## fill to maxFrame size
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, right_wrist_Y_Difference_filled) ########## fill to maxFrame size
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, slopeArray_filled) ######################## fill to maxFrame size
	#print(featureVector.shape)
	
	###################################### __________________________ ###################################
	featureVector = numpy.append(featureVector, left_wrist_X_normalized_filled) ###### fill to maxFrame size
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, left_wrist_Y_normalized_filled) ###### fill to maxFrame size
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, left_wrist_X_Difference_filled) ########## fill to maxFrame size
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, left_wrist_Y_Difference_filled) ########## fill to maxFrame size
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, slopeArray2_filled) ######################## fill to maxFrame size
	#print(featureVector.shape)
	
	################################### Might drop these 2 features, depending on accruracy ################
	
	featureVector = numpy.append(featureVector, zeroCrossingArray[index[0:5]]) ##### size of 5
	#print(featureVector.shape)
	
	featureVector = numpy.append(featureVector, maxDiffArray[index[0:5]])  ######### size of 5
	#print(featureVector.shape)
	return featureVector 

def allPaths(fileDir):
	# fileDir = "C:" + os.sep + "Users\Dunchuan\Desktop\CSV"
	pathList = numpy.array([])
	for root, dirs, files in os.walk(fileDir):
		#print(root)
		pathList = numpy.append(pathList, root)
		#print(files)
	# index = 0, root path
	# index = 1 2 3 4 5 6, child paths
	#print("pathList size", len(pathList)) 
	return pathList

def printAllFiles_singleClass(path): # print a path from a single class
	rootdir = path
	for parent,dirnames,filenames in os.walk(rootdir):
			for filename in filenames:
				print(filename)
				rawData=pd.read_csv(os.path.join(parent,filename), sep=',', skiprows=1, header=None)

def printAllFiles(pathList): # Print all files in 6 gestures folders at once 
	for i in range(1, len(pathList)):
		inputDir = pathList[i]
		rootdir = inputDir
		for parent,dirnames,filenames in os.walk(rootdir):
			for filename in filenames:
				print(filename)
				rawData=pd.read_csv(os.path.join(parent,filename), sep=',', skiprows=1, header=None)

## Begin ##
fileDir = "C:" + os.sep + "Users\Dunchuan\Desktop\CSV"
pathList = allPaths(fileDir)

# printAllFiles(pathList)
# printAllFiles_singleClass(pathList[3]) 
##### 需要修改语法 gesture = [1 "buy", 2 "communicate", 3 "fun", 4 "hope", 5 "mother", 6 "really"]

# get max frame for each gesture class
maxFrameList = numpy.array([])
for i in range(1, len(pathList)) :
	maxFrameList = numpy.append(maxFrameList, getMaxFrame(pathList[i])) #	Get the max frame for ONE gesture class
	# build_featureVector()
# print(maxFrameList)


########################################## Build 6 feature vectors #############################

def combine_feature_vectors_single(pathList, i, maxFrame): # given pathList at i, build a feature vector	
	featureVectorGesture = numpy.array([])
	inputDir = pathList[i]
	rootdir = inputDir
	for parent,dirnames,filenames in os.walk(rootdir):
		for filename in filenames:
			rawDataTemp = pd.read_csv(os.path.join(parent,filename), sep=',', skiprows=1, header=None)
			#print(filename)
			featureVectorGesture = build_featureVector(rawDataTemp, maxFrame)
			#print(featureVector.shape)
	return featureVectorGesture

frameParam = int(max(maxFrameList))
fv1 = combine_feature_vectors_single(pathList, 1, frameParam)	# params: pathList, i = 1, int(maxFrameList[0])	
#print("fv1 shape",fv1.shape)
#print(fv1)
fv2 = combine_feature_vectors_single(pathList, 2, frameParam)
fv3 = combine_feature_vectors_single(pathList, 3, frameParam)
fv4 = combine_feature_vectors_single(pathList, 4, frameParam)
fv5 = combine_feature_vectors_single(pathList, 5, frameParam)
fv6 = combine_feature_vectors_single(pathList, 6, frameParam)

print(fv1.shape, fv2.shape, fv3.shape, fv4.shape, fv5.shape, fv6.shape)
############### Build a feature matrix by 6 feature vectors
featureMatrix = numpy.array([])
featureMatrix = numpy.zeros((1, len(fv1)))
print(featureMatrix.shape)
featureMatrix = numpy.concatenate((featureMatrix, [fv1]),axis=0)
print(featureMatrix.shape)
featureMatrix = numpy.concatenate((featureMatrix, [fv2]),axis=0)
print(featureMatrix.shape)

featureMatrix = numpy.concatenate((featureMatrix, [fv3]),axis=0)
print(featureMatrix.shape)

featureMatrix = numpy.concatenate((featureMatrix, [fv4]),axis=0)
print(featureMatrix.shape)

featureMatrix = numpy.concatenate((featureMatrix, [fv5]),axis=0)
print(featureMatrix.shape)

featureMatrix = numpy.concatenate((featureMatrix, [fv6]),axis=0)
print(featureMatrix.shape)

featureMatrix = numpy.delete(featureMatrix, 0, axis=0) # Delete the first row 
print(featureMatrix.shape)
#print(featureMatrix)

############## Train ###################
# SVM
svm_model = svm.SVC() # SVM, Naive Bayes || KNN, Decision Tree,
# trainingSamples = numpy.concatenate((featureMatrix, ),axis=0)

interval = len(fv1)
# print(interval) # 1170
# print(featureMatrix.shape[1]) # 1170

trainingSamples = featureMatrix
labelVector = [0] * interval* interval # first 1170 x 6 number of 0's
labelVector[interval:] = [1] * interval #
labelVector[2*interval:] = [2] * interval
labelVector[3*interval:] = [3] * interval
labelVector[4*interval:] = [4] * interval
labelVector[5*interval:] = [5] * interval 

# print( trainingSamples[0]==fv1)
#testTrain = numpy.zeros((1, interval))
#testTrain = numpy.concatenate((testTrain, [fv1]),axis=0)
#testTrain = numpy.concatenate((testTrain, [fv2]),axis=0)
#testTrain = numpy.delete(testTrain, 0, axis=0)

testTrain = numpy.concatenate((trainingSamples[0],trainingSamples[0]), axis=0)
testTrain = numpy.concatenate((testTrain,trainingSamples[0]), axis=0)
testTrain = numpy.concatenate((testTrain,trainingSamples[0]), axis=0)
testTrain = numpy.concatenate((testTrain,trainingSamples[0]), axis=0)
testTrain = numpy.concatenate((testTrain,trainingSamples[0]), axis=0)
testTrain = numpy.concatenate((testTrain,trainingSamples[1]), axis=0)
testTrain = numpy.concatenate((testTrain,trainingSamples[1]), axis=0)
testTrain = numpy.concatenate((testTrain,trainingSamples[1]), axis=0)
testTrain = numpy.concatenate((testTrain,trainingSamples[1]), axis=0)
testTrain = numpy.concatenate((testTrain,trainingSamples[1]), axis=0)

#print("Look", trainingSamples[0])
print(" trainingSamples[0] shape is ", trainingSamples[0].shape)
testTrain = numpy.array([])
testTrain = numpy.concatenate([[fv1], [fv1]], axis=0)
testTrain = numpy.concatenate([testTrain, [fv2]], axis=0)
testTrain = numpy.concatenate([testTrain, [fv3]], axis=0)
testTrain = numpy.concatenate([testTrain, [fv4]], axis=0)
testTrain = numpy.concatenate([testTrain, [fv5]], axis=0)
testTrain = numpy.concatenate([testTrain, [fv6]], axis=0)

testTrain = numpy.delete(testTrain, 0, axis=0)
#print("Look Test Train", testTrain)
print("test Train shape is ", testTrain.shape)

#print("testTrain.shape", testTrain.shape)
#testLabelVector = [0] * interval * 6
#testLabelVector[interval:] = [1] * interval
# print(len(testLabelVector))

#testLabelVector = [0] *  2
#testLabelVector[1] = 1
testLabelVector = [0,1,2,3,4,5]
print("testLabelVector", testLabelVector)
svm_model.fit(testTrain, testLabelVector)
print(svm_model.predict([testTrain[5]]))
####################################
# Naive Bayes
# nb_model = 
