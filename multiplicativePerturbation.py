#############################################################
# 		 Multiplicatie Data Perturbation (GDP)
#############################################################

import os, sys, random, math
from scipy.stats import ortho_group
import numpy as np
import statistics

#Funtion to read the dataSet
#Referenced from the folowing source: http://cecs.wright.edu/~keke.chen/cloud/labs/privacy/readsvm.py
def readDataSet(fileName):
	recordords = []
	ncols = 0
	file = open(fileName)
	for line in file.readlines():
		fields = line.strip().split(",")
	record = []
	for field in fields[0:]:
		if isFloat(field):
			record.append(float(field))
	
	recordords.append(record)
	
	ncols = len(record)
	newRecords=[]
	for r in recordords:
		m = statistics.mean(r)
		sd = statistics.stdev(r)
		
		v = [0]*ncols
		for i in r:
			v.append(float((i-m)/sd))
		
		newRecords.append(v)
	
	return newRecords

#Funtion to generate perturbated data from original dataSet
#Parameter: { dataSet, sigma, fileName }
def geoDataPerturbation(record,sig,fileName):
	shapeTMatrixRecord = np.shape(np.transpose(record)) #Create the Transpose of the dataRecords, along with indexing using Numpy Shape function
	arrayTMatrixRecord = np.array(np.transpose(record)) #Create the Transpose of the dataRecords
	sigma = sig
	R = ortho_group.rvs(dim=shapeTMatrixRecord[0]) #Returns a random orthogonal matrix of dataSet's transpose
	t = np.random.rand(shapeTMatrixRecord[0],shapeTMatrixRecord[1]) #Creates an array of dataSet shape and populate it #with random samples from the dataSet's transpose (0, 1)
	d = np.random.rand(shapeTMatrixRecord[0],shapeTMatrixRecord[1]) #Creates another array of dataSet shape and populate #it with random samples from the dataSet's transpose (0, 1)
	D = d * sigma #Introduce sigma to the original dataset
	identityMatrix = matrixMultipliction(R,arrayTMatrixRecord)
	firstOutput = matrixAddition(identityMatrix,t) #(Rx + t)
	secondOutput = matrixAddition(firstOutput,D) #((Rx + t) + D)
	Y = np.transpose(secondOutput) #OUTPUT Purturbated Data Matrix { Y = (Rx + t + D) }
	Yshape = np.shape(Y)
	N =[[0 for row in range(Yshape[0])] for col in range(Yshape[0])]
	for i in range(Yshape[0]):
		for j in range(Yshape[1]):
			N[i][j] = str(Y[i][j])
	np.savetxt((os.path.splitext(fileName)[0] + "_" + str(sigma) + os.path.splitext(fileName)[1]),N,fmt='%s',delimiter=" ")
	print ("File Generated: " + ((os.path.splitext(fileName)[0] + "_" + str(sigma) + os.path.splitext(fileName)[1])))

#Funtion for Matrix Multiplication
#Parameter: { Matrix1, Matrix2 }
def matrixMultipliction (martrix1, martrix2):
	martrix2shape = np.shape(martrix2)
	martrix2Row = martrix2shape[0]
	martrix2Column = martrix2shape[1]
	result = [[0 for row in range(martrix2Column)] for col in range(martrix2Row)]
	for i in range(martrix2Row):
		for j in range(martrix2Column):
			for k in range(martrix2Row):
				if isFloat(martrix1[i][k]) and isFloat(martrix2[k][j]):
					result[i][j] += (float(martrix1[i][k])*float(martrix2[k][j]))
	return result

#Funtion for Matrix Addition
#Parameter: { Matrix1, Matrix2 }
def matrixAddition (martrix1, martrix2):
	martrix2shape = np.shape(martrix2)
	martrix2Row = martrix2shape[0]
	martrix2Column = martrix2shape[1]
	result = [[0 for row in range(martrix2Column)] for col in range(martrix2Row)]
	for i in range(martrix2Row):
		for j in range(martrix2Column):
			result[i][j] = martrix1[i][j]+martrix2[i][j]
	return result

#Funtion to check if the string value is float
#Parameter: { value }
def isFloat(value):
	try:
		float(value)
		return True
	except:
		return False

#Main method
if __name__ == '__main__':
	filePath = "{dataset path}" #Path of the original dataset
	fileName = os.path.basename(filePath) #Obtain the fineName from file
	sigList = [0.10, 0.20, 0.30, 0.40] #List of Noise Vector, higher noise vector = more data preserving
	for sig in sigList:
		record = readDataSet(filePath) #Read the dataset
		geoDataPerturbation(record,sig,fileName) #Generate the perturbated data. Parameters {record, sigma, #fileName}
	print ("Dataset Perturbation Completed")







