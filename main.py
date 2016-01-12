from dtree import *
import sys
import math
import matplotlib.pyplot as plt 
from pylab import *
import csv

class Globals:
		noisyFlag = False
		pruneFlag = False
		valSetSize = 0
		dataset = None
		debug = False


##Classify
#---------

def classify(decisionTree, example):
		return decisionTree.predict(example)

##Learn
#-------
def learn(dataset):
		learner = DecisionTreeLearner()
		learner.train(dataset)
		return learner.dt

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
	"""Parses arguments vector, looking for switches of the form -key {optional value}.
	For example:
		parseArgs([ 'main.py', '-n', '-p', 5 ]) = { '-n':True, '-p':5 }"""
	args_map = {}
	curkey = None
	for i in xrange(1, len(args)):
		if args[i][0] == '-':
			args_map[args[i]] = True
			curkey = args[i]
		else:
			assert curkey
			args_map[curkey] = args[i]
			curkey = None
	return args_map

def validateInput(args):
		args_map = parseArgs(args)
		valSetSize = 0
		noisyFlag = False
		pruneFlag = False
		boostRounds = -1
		maxDepth = -1
		if '-n' in args_map:
			noisyFlag = True
		if '-p' in args_map:
			pruneFlag = True
			valSetSize = int(args_map['-p'])
		if '-d' in args_map:
			maxDepth = int(args_map['-d'])
		if '-b' in args_map:
			boostRounds = int(args_map['-b'])
		return [noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds]

def main():
		arguments = validateInput(sys.argv)
		noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds = arguments
		print noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds

		# Read in the data file. F_other is just for graphing purposes
		
		if noisyFlag:
				f = open("noisy.csv")
				f_other = open('data.csv')
		else:
				f = open("data.csv")
				f_other = open('noisy.csv')

		data = parse_csv(f.read(), " ")
		dataset = DataSet(data)

		data_other = parse_csv(f_other.read(), " ")
		dataset_other = DataSet(data_other)

		# Copy the dataset so we have two copies of it
		# Also copy the other dataset for graphing purposes later on
		examples = dataset.examples[:]
 
		dataset.examples.extend(examples)

		examples_other = dataset_other.examples[:]
 
		dataset_other.examples.extend(examples_other)

		dataset.max_depth = maxDepth
		if boostRounds != -1:
			dataset.use_boosting = True
			dataset.num_rounds = boostRounds


		# ====================================
		# WRITE CODE FOR YOUR EXPERIMENTS HERE
		# ====================================

		# 10-fold Cross Validation calculates score of learned tree on a given dataset
		avg_errors = tenfoldCrossValidate(dataset, valSetSize, maxDepth) 
		print "Cross-validated training performance: " + str(avg_errors[0])
		print "Cross-validated test performance:     " + str(avg_errors[1])

		if noisyFlag:
			datasets = [dataset_other, dataset]
		else:
			datasets = [dataset, dataset_other]
		
		###  Code for graphing. Comment out whichever one you want to graph. 
		# graph_2bi(datasets)
		# graph_3aB(datasets)
		# graph_3aD(datasets)

def graph_2bi(datasets):

		plt.clf()
		xs = []
		ys_train = []
		ys_test = []
		ys_train_n = []
		ys_test_n = []
		for val_size in range(0,81):
			avg_errors = tenfoldCrossValidate(datasets[0],val_size, -1)
			xs.append(val_size)
			ys_train.append((float)(avg_errors[0]))
			ys_test.append((float)(avg_errors[1]))
			avg_errors = tenfoldCrossValidate(datasets[1],val_size, -1)
			ys_train_n.append((float)(avg_errors[0]))
			ys_test_n.append((float)(avg_errors[1]))

		pl=plt.plot(xs,ys_train,color='b')
		pl=plt.plot(xs,ys_test,color='r')
		pl=plt.plot(xs,ys_train_n,color='g')
		pl=plt.plot(xs,ys_test_n,color='y')

		plt.title("Graph 2(b)(i) Varying validation size for both datasets")
		plt.xlabel('validation set size')
		plt.ylabel('cross-validated performance')
		plt.axis([0,85,0.55,1.00])
		plt.legend(['train','test','train_noisy','test_noisy'], loc='lower right')
		savefig('2bifigure.pdf')
		savefig('2bifigure.jpg')

		plt.show()

def graph_3aB(datasets):

		plt.clf()
		xs = []
		ys_test = []
		ys_test_n = []

		for num_boosting in range(1,30):
			for i in range(2):
				datasets[i].num_rounds = num_boosting
				datasets[i].use_boosting = True
			avg_errors = tenfoldCrossValidate(datasets[0],0,1)
			ys_test.append((float)(avg_errors[1]))
			xs.append(num_boosting)
			avg_errors = tenfoldCrossValidate(datasets[1],0,1)
			ys_test_n.append((float)(avg_errors[1]))

		pl=plt.plot(xs,ys_test,color='b')
		pl=plt.plot(xs,ys_test_n,color='r')

		plt.title("Graph 3(a)(B) Boosting performance for both datasets")
		plt.xlabel('number of boosting rounds')
		plt.ylabel('cross-validated test performance')
		plt.axis([0,35,0.7,1.05])
		plt.legend(['test','test_noisy'], loc='lower right')
		savefig('3aBfigure.pdf')
		savefig('3aBfigure.jpg')

		plt.show()

def graph_3aD(datasets):

		plt.clf()
		xs = []
		ys_train = []
		ys_test = []
		ys_train_n = []
		ys_test_n = []

		for num_boosting in range(1,16):
			for i in range(2):
				datasets[i].num_rounds = num_boosting
				datasets[i].use_boosting = True
			avg_errors = tenfoldCrossValidate(datasets[0],0,1)
			ys_train.append((float)(avg_errors[0]))
			ys_test.append((float)(avg_errors[1]))
			xs.append(num_boosting)
			avg_errors = tenfoldCrossValidate(datasets[1],0,1)
			ys_train_n.append((float)(avg_errors[0]))
			ys_test_n.append((float)(avg_errors[1]))

		pl=plt.plot(xs,ys_train,color='b')
		pl=plt.plot(xs,ys_test,color='r')
		pl=plt.plot(xs,ys_train_n,color='g')
		pl=plt.plot(xs,ys_test_n,color='y')
		title_plot = "Graph 3(a)(D) Boosting performance for both datasets"
		plt.title(title_plot)
		plt.xlabel('number of boosting rounds')
		plt.ylabel('cross-validated performance')
		plt.axis([0,16,0.6,1.05])
		plt.legend(['train','test','train_noisy','test_noisy'], loc='lower right')
		savefig('3aDfigure.pdf')
		savefig('3aDfigure.jpg')
		plt.show()

def tenfoldCrossValidate(dataset, valSetSize, maxDepth):
	"""
	Args:
		dataset: dataSet object to be learned on 
		valSetSize: specified size of the validation set
		maxDepth: maxDepth for regularization and for boosting
	Returns: 
		err[o]: cross-validated training performance
		err[1]: cross-validated test performance
	"""
	# Using 0-1 accuracy

	global debug
	debug = False
	num_folds = 10
	total_train_error = 0
	total_test_error = 0

	data_length = len(dataset.examples)
	fold_length = data_length / num_folds / 2 # because two copies

	for round_num in range(0,num_folds):

		# divide the dataset into test, train and validation sets
		test_set = dataset.examples[round_num*fold_length:(round_num+1)*fold_length]
		train_set = dataset.examples[(round_num+1)*fold_length:data_length/2+(round_num)*fold_length-valSetSize]
		val_set = dataset.examples[data_length/2+(round_num)*fold_length-valSetSize:data_length/2+(round_num)*fold_length]

		train_data = DataSet(train_set, dataset.attrs, dataset.target, dataset.values, dataset.attrnames, dataset.name, dataset.source)
		test_data = DataSet(test_set, dataset.attrs, dataset.target, dataset.values, dataset.attrnames, dataset.name, dataset.source)
		val_data = DataSet(val_set,dataset.attrs, dataset.target, dataset.values, dataset.attrnames, dataset.name, dataset.source)

		train_data.max_depth = maxDepth

		# generate decision tree from training/learning on the training set\
		# made boosting and pruning mutually exclusive
		if not(dataset.use_boosting):
			dtree = learn(train_data)
			if debug:
				print "Final dtree from fold "+ str(round_num) +": " + str(dtree)
			if valSetSize == 0:
				train_errors = performance(train_data, dtree)
				test_errors = performance(test_data, dtree)
			else:
				# prune tree if valSetSize was set
				pruned_dtree = prune(dtree, train_data, val_data)
				if debug:
					print "Final pruned tree from fold "+ str(round_num) +": " +str(pruned_dtree)
				train_errors = performance(train_data, pruned_dtree) 
				test_errors = performance(test_data, pruned_dtree)
		else:
			multi_dtree = adaBoost(train_data, dataset.num_rounds)
			train_errors = performance(train_data, multi_dtree, True, multi_dtree) 
			test_errors = performance(test_data, multi_dtree, True, multi_dtree)

		total_train_error += train_errors
		total_test_error += test_errors

	train_performance = (float) (len(train_set)*num_folds-total_train_error)/((len(train_set))*num_folds)
	test_performance =  (float) (len(test_set)*num_folds-total_test_error)/(len(test_set)*num_folds)

	return [train_performance, test_performance]

def performance(dataset, dtree, boosting=False, multi_dtree=[]):
	"""
	Returns number of errors on this dataset when the dtree is applied to the datset
	"""
	
	predictions = []
	for example in dataset.examples:
		if boosting:
			predictions.append(weightedVote(multi_dtree,example))
		else:
			predictions.append(dtree.predict(example))

	errors = 0
	for i in range(len(predictions)): 
		if predictions[i] != dataset.examples[i].attrs[dataset.target]:
			errors += 1
	return errors

def weightedVote(trees_weights, example):
	"""
	Weighted voting from the ensemble of hypotheses generated in boosting.
	Args:
		trees_weights = list of tuples of (dtrees, weights) in the ensemble of hypotheses
		example = example whose label we want to predict
	Returns: 
		prediction of the example's label
	"""

	sum = 0.0
	for (dtree, weight) in trees_weights:
		if dtree.predict(example)==0:
			sum-=weight 
		else:
			sum+=weight
	if sum>=0:
		return 1 
	else:
		return 0

def adaBoost(dataset, boostRounds):
	"""
	Implements adaBoost on the dataset
	Args:
		dataset 
		boostRounds - number of boosting rounds
	Returns: 
		tuples - list of tuples of (dtrees, weights) where weights are the alphas
				computed for each dtree in the ensemble
	"""

	for example in dataset.examples:
		example.weight = 1.0/(len(dataset.examples[:]))
	tuples = []
	for r in range(boostRounds):	
		# learns a different hypothesis	
		dtree = learn(dataset)
		# calculates error from that hypothesis
		epsilon = weightedError(dataset, dtree)
		if epsilon == 0.0: 
			return [(dtree, 1)]
		else:
			alpha = (1.0/2)*log2((1.0-epsilon)/epsilon)
			sum_new_weight = 0.0
			v_n = []
			for i in range(len(dataset.examples)):
				if dtree.predict(dataset.examples[i]) != dataset.examples[i].attrs[dataset.target]:
					v_n.append(dataset.examples[i].weight*(math.e ** alpha))
				else:
					v_n.append(dataset.examples[i].weight/(math.e ** alpha))
				sum_new_weight += v_n[i]
			# updates the weights of the examples
			for i in range(len(dataset.examples)):
				dataset.examples[i].weight = v_n[i]/sum_new_weight
		# adds the hypothesis and its alpha to the ensemble
		tuples.append((dtree, alpha))
	return tuples

def weightedError(dataset, dtree):
	"""
	Returns weighted errors on this dataset based on the predictions of dtree
	Measures weighted performance of dtree on the dataset
	"""

	predictions = []
	for example in dataset.examples:
		predictions.append(dtree.predict(example))

	errors = 0
	for i in range(len(predictions)): 
		if predictions[i] != dataset.examples[i].attrs[dataset.target]:
			errors += dataset.examples[i].weight
	return errors

def splitData(dataset, attr, val):
	"""
	Used in pruning when selecting a subset of the training or validation dataset 
	that reaches a certain subtree on which prune() is being called
	Args: 
		dataset = training or validation set to be split
		attr = attribute we're splitting on
		val = value of that attribute
	Returns:
		subset of the dataset 
	"""
	subset = DataSet([],dataset.attrs, dataset.target, dataset.values, dataset.attrnames)
	for example in dataset.examples[:]: 
		if example.attrs[attr] == val:
			subset.add_example(example)
	return subset

def prune(dtree, dataset, valSet):
	"""
	Carries out the validation pruning
	Args: 
		dtree = tree or subtree to be pruned
		dataset = the training dataset that reaches that tree or subtree
		valSet = the validation set that reaches that tree or subtree
	Returns:
		either a pruned subtree (a leaf) or the original subtree
	"""

	sub_valSet = DataSet([],dataset.attrs, dataset.target, dataset.values, dataset.attrnames)
	for i in range(len(dtree.branches.items())):
		(attrval, subtree) = dtree.branches.items()[i]
		if subtree.nodetype != DecisionTree.LEAF:
			# this is not a leaf! recurse on subtree, splitting data according to tree traversal
			sub_set = splitData(dataset, dtree.attrname, attrval)
			sub_valSet = splitData(valSet, dtree.attrname, attrval)
			dtree.branches.pop(attrval) # remove the entry
			dtree.branches[attrval] = prune(subtree, sub_set, sub_valSet)

	# if we get here, all children are leaves or all subtrees have gone through the prune() function- do prune stuff on the root node of subtree
	# use Majority Learner to get the majority class of this subset of data
	learner = MajorityLearner()
	learner.train(dataset)
	# majorityClass is the value of the most popular target, in this case 0 or 1
	majorityClass = learner.predict(dataset.examples[0]) # input to predict doesn't matter

	prunedLeafTree = DecisionTree(DecisionTree.LEAF, None, majorityClass)

	oldErrors = performance(valSet, dtree)
	majorityErrors = performance(valSet, prunedLeafTree)

	if (oldErrors < majorityErrors): # T' performs worse
		return dtree
	else: 
		return prunedLeafTree

main()


		
