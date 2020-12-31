import numpy as np
import math
import random

class Leaf:
    def __init__(self,data):
        self.less = None #less than
        self.greater = None #greater than
        self.data = data
        self.feature = None
        self.valid = True
        self.predict = -1
        self.split_value = None
        self.valid_error = None
        self.predict_true = 0
        self.predict_false = 0

    def insert(self,feature,split, less, greater):
        self.less = Leaf(less)
        self.greater = Leaf(greater)
        self.feature = feature
        self.split_value = split

    def prune(self):
        self.valid = False
        if self.less:
            self.less.prune()
        if self.greater:
            self.greater.prune()

def readData(filename, feature_size):
    data_array = []
    #Read the file name that was passed in as paramter
    with open(filename,'r') as testing:
        for eachline in testing:
            split_line = eachline.split()
            data_array.append([float(x) for x in split_line])
    return data_array

def readFeature(filename):
    data_array = []
    with open(filename,'r') as feature:
        for eachline in feature:
            data_array.append(eachline)
    return data_array

def SplitDataByColumn(data_set,feature_size):
    data_array = []
    for i in range(feature_size):
        data_array.append([])
    for i in range(len(data_set)):
        for y in range(feature_size):
            data_array[y].append([data_set[i][y], data_set[i][feature_size]])
    return data_array

#Find Decision
def FindBestFeature(train_set,data_set,root,feature_size):
    MaxInfoGain = [([0.0,0.0]),0.0]
    common_label = FindCommonLabel(data_set[0])
    lessthan = []
    greaterthan = []
    t = 0
    H_X = EntropyOfLabel(data_set[0], common_label)
    if(H_X == 0):
        root.predict = common_label
        # print("predict : ", common_label)
        return
    else:
        for i in data_set: #feature
            t = t + 1
            result,lessthan,greaterthan = FindIG(i,common_label,H_X)
            
            if MaxInfoGain[0][0] < result[0]:
                MaxInfoGain = [result,t]
        # print(MaxInfoGain)
        # print(MaxInfoGain[0][1])
        lessthan,greaterthan = splitDataBythreshold(train_set, MaxInfoGain[1], MaxInfoGain[0][1])
        root.insert(MaxInfoGain[1],MaxInfoGain[0][1],lessthan,greaterthan)
        FindBestFeature(lessthan, SplitDataByColumn(lessthan,feature_size),root.less,feature_size)
        FindBestFeature(greaterthan, SplitDataByColumn(greaterthan,feature_size),root.greater,feature_size)
    return

def splitDataBythreshold(data_set, feature, threshold):
    lessthan = []
    greaterthan = []
    for i in data_set:
        if(i[feature-1] < threshold):
            lessthan.append(i)
        else:
            greaterthan.append(i)
    return lessthan,greaterthan

#Find IG
def FindIG(data_set, common_label, H_X):
    in_between = []
    common_data = FindCommonData(data_set)
    in_between = FindInBetween(common_data)
    split_IB,split_index,lessthan,greaterthan = SplitByInBetween(data_set, in_between, common_label)
    IG = H_X - split_IB
    return [IG,in_between[split_index]],lessthan,greaterthan

#Find common data
def FindCommonData(data_set):
    common_Data = []
    for i in data_set:
        if i[0] not in common_Data:
            common_Data.append(i[0])
    return common_Data

#Find the amount of labels
def FindCommonLabel(data_set):
    common_Data = []
    for i in data_set:
        if i[1] not in common_Data:
            common_Data.append(i[1])
    return common_Data


#Find entropy of the data given label H(X)
def EntropyOfLabel(data_set, common_label):
    labels = {}
    for i in data_set:
        if i[1] not in labels:
            labels[i[1]] = 1
        elif i[1] in common_label:
            labels[i[1]] = labels[i[1]] + 1
    final_entropy = 0
    for label in labels:
        labels[label] = labels[label]/ len(data_set)
        final_entropy = final_entropy - 1*labels[label]*(math.log2(labels[label]))
    return final_entropy


#Finds the value we may want to split the data by.
def FindInBetween(data_set):
    in_between = []
    if (len(data_set) == 1):
        return data_set
    else:
        data_set.sort()
        for i in range(len(data_set)-1):
            in_between.append( (data_set[i] + data_set[i + 1])/ 2 )
        return in_between


#Finds H(X|Z)
def SplitByInBetween(data_set, inbetween, common_label):
    min_entropy = 1
    min_index_split = 0
    t = 0
    greater_equal_split = []
    less_than_split = []
    for i in inbetween:
        greater_equal_split = []
        less_than_split = []
        for j in data_set:
            if j[0] < i:
                less_than_split.append(j)
            else:
                greater_equal_split.append(j)
        
        prob_one = len(less_than_split) / len(data_set)
        prob_zero = len(greater_equal_split) / len(data_set)
        H_X_Zis1 = EntropyOfLabel(less_than_split, common_label)
        H_X_Zis0 = EntropyOfLabel(greater_equal_split, common_label)
        entropy = prob_zero*(H_X_Zis0) + prob_one*(H_X_Zis1)
        if(min_entropy != min(min_entropy, entropy)):
            min_entropy = min(min_entropy, entropy)
            min_index_split = t
        t = t + 1
    return min_entropy, min_index_split,less_than_split,greater_equal_split

def test_for_all_data(data_set, root):
    count = 0
    k = 0
    for data in data_set:
        k = test_for_one_data(data, root)
        if(k == data[22]):
            count = count + 1

    return  count/len(data_set)

def test_for_one_data(data, root):
    prediction = -1
    if(root.predict == -1):
        if(data[root.feature-1] < root.split_value):
            prediction = test_for_one_data(data, root.less)
        else:
            prediction = test_for_one_data(data, root.greater)
    else:
        return root.predict
    return prediction

def pruning(root, validation_set, origin_root,testing_set):
    root.valid_error = 1 - (test_for_all_data(validation_set, origin_root))
    if root.less == None and root.greater == None:
        return
    else: 
        if root.predict_false > root.predict_true:
            root.predict = [0.0]
        elif root.predict_true > root.predict_false:
            root.predict = [1.0]
        elif root.predict_true == root.predict_false:
            pruning(root.less,validation_set, origin_root, testing_set)
            pruning(root.greater,validation_set, origin_root, testing_set)
        valid_error2 = 1-test_for_all_data(validation_set,origin_root)
        if(root.valid_error < valid_error2 and root.predict != -1):
            root.predict = -1
            pruning(root.less,validation_set, origin_root, testing_set)
            pruning(root.greater,validation_set, origin_root, testing_set)
        else:
            CountMajority(origin_root)
            print("Prune Occured : validation error ", valid_error2, ", testing error ", 1-test_for_all_data(testing_set,origin_root))
            return
    return
 
def CountMajority(root):
    root.predict_false = 0
    root.predict_true = 0 
    if root.predict != -1:
        if root.predict == [0.0]:
            root.predict_false = 1
        elif root.predict == [1.0]:
            root.predict_true = 1
        return
    if root.less != None:
        CountMajority(root.less)
        root.predict_false = root.predict_false + root.less.predict_false
        root.predict_true = root.predict_true + root.less.predict_true
    if root.greater != None:
        CountMajority(root.greater)
        root.predict_false = root.predict_false + root.greater.predict_false
        root.predict_true = root.predict_true + root.greater.predict_true
    # print(root.predict_false, " ", root.predict_true)
    return

def printTree(root, i):
    print("Parent : feature :", root.feature, " value :", root.split_value, " dataSize: ", len(root.data), " predict: ", root.predict)
    i = i + 1
    if i == 4:
        return
    if root.less != None:
        print("Left child: feature: ", root.less.feature, " value : ", root.less.split_value, "dataSize: ", len(root.less.data))
        printTree(root.less,i)
    if root.greater != None:
        print("right child: feature: ", root.greater.feature, " Greater : ", root.greater.split_value, "dataSize: ", len(root.greater.data))
        printTree(root.greater,i)
    return

def main():

    features = readFeature('features.txt')
    feature_size = len(features)

    train_set = readData('training_data_pa2.txt', feature_size)
    train_set = np.array(train_set)
    test_set = readData('testing_data_pa2.txt', feature_size)
    test_set = np.array(test_set)
    validation_set = readData('validation_data_pa2.txt', feature_size)
    validation_set = np.array(validation_set)

    feature_train_Set = SplitDataByColumn(train_set, feature_size)
    feature_train_Set = np.array(feature_train_Set)

    root = Leaf(feature_train_Set)
    FindBestFeature(train_set,feature_train_Set, root,feature_size)
    print("training error = ", 1-test_for_all_data(train_set,root))
    print("test error = ", 1-test_for_all_data(test_set, root))
    print("validation error = ", 1-test_for_all_data(validation_set, root))

    #pruning
    CountMajority(root)
    pruning(root,validation_set, root, test_set)
    print("Final testing error :", 1-test_for_all_data(test_set,root))
    i = 0
    printTree(root, i)
    # pruning(validation_set, root)
    # returning_root = Leaf([1])
    # findMinValidInTree(root, min_valid,returning_root)
    # print(min_valid)
    # print("Second prune root node: feature ", returning_root.feature, " value ", returning_root.split_value)






    return

if __name__== "__main__" :
    main()