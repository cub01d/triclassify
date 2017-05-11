#!/usr/bin/env python3
# Kenneth Chan
# kc135576@cs.ucsb.edu
# CS165B Spring 2017
import sys

# file names
training_data_file = sys.argv[1]
testing_data_file = sys.argv[2]

# training data parameters
dim = None
classA = []
classB = []
classC = []
centroids = []

def loadTrainingData():
	"""load training data from file"""
	global dim, classA, classB, classC
	try:
		with open(training_data_file, "r") as train:
			dim, n1, n2, n3 = map(int, train.readline()[:-1].split())
			
			for i in range(n1):
				point = tuple(map(float, train.readline()[:-1].split()))
				classA.append(point)
			for i in range(n2):
				point = tuple(map(float, train.readline()[:-1].split()))
				classB.append(point)
			for i in range(n3):
				point = tuple(map(float, train.readline()[:-1].split()))
				classC.append(point)

			# make sure our data is good
			assert(len(classA) == n1)
			assert(len(classB) == n2)
			assert(len(classC) == n3)

	except IOError:
		print("training file '{}' not found.".format(training_data_file))
		sys.exit(1)

def calculateCentroids(c):
	"""Calculates the centroids of class c."""
	centroid = [0] * dim
	for i in range(len(c)):
		for j in range(dim):
			centroid[j] += c[i][j]
	for i in range(len(centroid)):
		centroid[i] /= float(len(c))    
	return centroid

def calculateLinearDiscriminants(a, b):
	"""
	Calculates the linear discriminant function and the threshold for
	the discriminant function between centroid a and centroid b. 
	"""
	global dim
	w = [a[i] - b[i] for i in range(dim)]
	x0 = [(a[i]+b[i])/2 for i in range(dim)]
	t = 0
	for i in range(dim):
		t += w[i]*x0[i]

	return w,t

def trainClassifiers():
	global classA, classB, classC, centroids
	# calculate the centroids
	for c in (classA, classB, classC):
		centroids.append(calculateCentroids(c))

	# calculate the linear discriminants for every pair
	# ld: ((w_ab, t_ab), (w_ac, t_ac), (w_bc, t_bc))
	# w_ij = dim dimensional point
	# t_ij = scalar
	ld = [
		calculateLinearDiscriminants(centroids[0], centroids[1]),
		calculateLinearDiscriminants(centroids[0], centroids[2]),
		calculateLinearDiscriminants(centroids[1], centroids[2])
	]
	return ld

class Data(object):

	# array indexes in table, stored as Data.table
	# 					actual C
	# predicted			A 	B 	C
	# 				A |	0	1	2		P hat
	# C hat			B |	3	4	5		/ 
	# 				C |	6 	7	8		N hat
	#					P / N			

	def __init__(self):
		self.table = [0]*9
		self.tpr = None
		self.fpr = None
		self.er = None
		self.acc = None
		self.prec = None
		self.total = 0

	def setData(self, actual, prediction):
		"""Updates data table with values"""
		self.table[(ord(prediction)-97)*3 + (ord(actual)-97)] += 1
		self.total += 1


	def done(self):
		"""Done setting data, update the stats"""
		# update self.tpr
		tpr_a = self.table[0] / (self.table[0] + self.table[3] + self.table[6])
		tpr_b = self.table[4] / (self.table[1] + self.table[4] + self.table[7])
		tpr_c = self.table[8] / (self.table[2] + self.table[5] + self.table[8])
		self.tpr = (tpr_a + tpr_b + tpr_c) / 3

		# update self.fpr
		fpr_a = (self.table[1] + self.table[2]) / (	self.table[1] + self.table[2] + 
													self.table[4] + self.table[5] +
													self.table[7] + self.table[8])
		fpr_b = (self.table[3] + self.table[5]) / (	self.table[0] + self.table[2] + 
													self.table[3] + self.table[5] +
													self.table[6] + self.table[8])
		fpr_c = (self.table[6] + self.table[7]) / (	self.table[0] + self.table[1] + 
													self.table[3] + self.table[4] +
													self.table[6] + self.table[7])
		self.fpr = (fpr_a + fpr_b + fpr_c) / 3

		# update self.er
		er_a = (self.table[1] + self.table[2] + self.table[3] + self.table[6]) / self.total
		er_b = (self.table[1] + self.table[3] + self.table[5] + self.table[7]) / self.total
		er_c = (self.table[2] + self.table[5] + self.table[6] + self.table[7]) / self.total
		self.er = (er_a + er_b + er_c) / 3
		# update self.acc
		acc_a = (self.table[0] + self.table[4] + self.table[5] + self.table[7] + self.table[8]) / self.total
		acc_b = (self.table[4] + self.table[0] + self.table[2] + self.table[6] + self.table[8]) / self.total
		acc_c = (self.table[0] + self.table[1] + self.table[3] + self.table[4] + self.table[8]) / self.total
		self.acc = (acc_a + acc_b + acc_c) / 3

		# update self.prec
		prec_a = self.table[0] / (self.table[0] + self.table[1] + self.table[2])
		prec_b = self.table[4] / (self.table[3] + self.table[4] + self.table[5])
		prec_c = self.table[8] / (self.table[6] + self.table[7] + self.table[8])
		self.prec = (prec_a + prec_b + prec_c) / 3

def update(actual, x, w_ab, w_ac, w_bc, t_ab, t_ac, t_bc, data):
	# a vs b
	if sum([x[i]*w_ab[i] for i in range(len(x))]) >= t_ab:
		# a vs c
		if sum([x[i]*w_ac[i] for i in range(len(x))]) >= t_ac:
			# a
			data.setData(actual=actual, prediction="a")
		else:
			# c
			data.setData(actual=actual, prediction="c")
	else:
		# b vs c
		if sum([x[i]*w_bc[i] for i in range(len(x))]) >= t_bc:
			# b
			data.setData(actual=actual, prediction="b")
		else:
			# c
			data.setData(actual=actual, prediction="c")

# data structure for updating stats
data = Data()

def testClassifiers(ld):
	# test distances from linear discriminants
	# resolve ties in order A,B,C
	global dim, testing_data_file, data
	try:
		with open(testing_data_file, "r") as f:
			dim2, n1, n2, n3 = map(int, f.readline()[:-1].split())
			assert(dim2 == dim)

			# learned parameters from training
			w_ab = ld[0][0]
			t_ab = ld[0][1]
			w_ac = ld[1][0]
			t_ac = ld[1][1]
			w_bc = ld[2][0]
			t_bc = ld[2][1]

			for i in range(n1):
				x = tuple(map(float, f.readline()[:-1].split()))
				update('a', x, w_ab, w_ac, w_bc, t_ab, t_ac, t_bc, data)
			for i in range(n2):
				x = tuple(map(float, f.readline()[:-1].split()))
				update('b', x, w_ab, w_ac, w_bc, t_ab, t_ac, t_bc, data)
			for i in range(n3):
				x = tuple(map(float, f.readline()[:-1].split()))
				update('c', x, w_ab, w_ac, w_bc, t_ab, t_ac, t_bc, data)

			data.done()

	except IOError:
		print("testing file '{}' not found.".format(testing_data_file))
		sys.exit(1)

def printStats():
	global data
	print("True positive rate = {}".format(round(data.tpr,2)))
	print("False positive rate = {}".format(round(data.fpr,2)))
	print("Error rate = {}".format(round(data.er,2)))
	print("Accuracy = {}".format(round(data.acc,2)))
	print("Precision = {}".format(round(data.prec,2)))

def main():
	loadTrainingData()			# load training data from file
	ld = trainClassifiers()		# calculate centroids, linear discriminants
	testClassifiers(ld)			# compare with testing file data
	printStats()

if __name__ == "__main__":
	main()
