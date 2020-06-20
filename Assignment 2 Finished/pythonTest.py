import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn import svm

rawData = pd.read_csv(r'C:\Users\Dunchuan\Desktop\CSV\fun\FUN_1_BAKRE.csv', sep=',', skiprows=1, header=None)

rY = rawData[34]

#plt.plot(rY)
#plt.show()

normRawData = (rY - numpy.mean(rY))/(numpy.max(rY-numpy.mean(rY))-numpy.min(rY-numpy.mean(rY)))

diffNormRawData = numpy.diff(normRawData)
print(len(diffNormRawData))
#plt.plot(diffNormRawData)
#plt.show()

zeroCrossingArray = numpy.array([])
maxDiffArray = numpy.array([])

if diffNormRawData[0] > 0:
 	initSign = 1
else:
 	initSign = 0

windowSize = 5;

for x in range(1, len(diffNormRawData)):
	if diffNormRawData[x] > 0:
		newSign = 1
	else:
		newSign = 0	 
	
	if initSign != newSign:
		zeroCrossingArray = numpy.append(zeroCrossingArray, x)
		initSign = newSign
		maxIndex = numpy.minimum(len(diffNormRawData),x+windowSize)
		minIndex = numpy.maximum(0,x - windowSize)
	
		maxVal = numpy.amax(diffNormRawData[minIndex:maxIndex])
		minVal = numpy.amin(diffNormRawData[minIndex:maxIndex])

		maxDiffArray = numpy.append(maxDiffArray, (maxVal - minVal))
		


	

index = numpy.argsort(-maxDiffArray)


featureVectorMother = numpy.array([])
print("vector size shape is", featureVectorMother.shape)
featureVectorMother = numpy.append(featureVectorMother, diffNormRawData)
print("vector size shape is", featureVectorMother.shape)
featureVectorMother = numpy.append(featureVectorMother, zeroCrossingArray[index[0:5]])
print("vector size shape is", featureVectorMother.shape)
featureVectorMother = numpy.append(featureVectorMother, maxDiffArray[index[0:5]])	
print("vector size shape is", featureVectorMother.shape)

featureMatrixMother = numpy.array([])
print("matrix shape is", featureMatrixMother.shape)
featureMatrixMother = numpy.concatenate([[featureVectorMother], [featureVectorMother]])
print("matrix shape is", featureMatrixMother.shape)
featureMatrixMother = numpy.concatenate((featureMatrixMother, [featureVectorMother]),axis=0)
print("matrix shape is", featureMatrixMother.shape)
featureMatrixMother = numpy.concatenate((featureMatrixMother, [featureVectorMother]),axis=0)
print("matrix shape is", featureMatrixMother.shape)
featureMatrixMother = numpy.concatenate((featureMatrixMother, [featureVectorMother]),axis=0)
featureMatrixMother = numpy.concatenate((featureMatrixMother, [featureVectorMother]),axis=0)
featureMatrixMother = numpy.concatenate((featureMatrixMother, [featureVectorMother]),axis=0)
featureMatrixMother = numpy.concatenate((featureMatrixMother, [featureVectorMother]),axis=0)
featureMatrixMother = numpy.concatenate((featureMatrixMother, [featureVectorMother]),axis=0)
featureMatrixMother = numpy.concatenate((featureMatrixMother, [featureVectorMother]),axis=0)
print("matrix shape is", featureMatrixMother.shape)

#print(featureMatrixMother)

featureMatrixNotMother = featureMatrixMother - numpy.random.rand(10,159)
print("flag", featureMatrixMother.shape,featureMatrixNotMother.shape)
TrainingSamples = numpy.concatenate((featureMatrixMother, featureMatrixNotMother),axis=0)

print(TrainingSamples.shape)

labelVector = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

clf = svm.SVC()
clf.fit(TrainingSamples,labelVector)
print(clf.predict([TrainingSamples[11]]))
