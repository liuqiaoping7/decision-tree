# coding=utf-8

from __future__ import division
import math
import operator
import time
import random
import copy
import sys
import ast
import csv
from collections import Counter



##################################################
# data class to hold csv data
##################################################
class data():
    def __init__(self, classifier):
        self.examples = [] #样本集
        self.attributes = [] #属性集
        self.attr_types = [] #属性值类型
        self.classifier = classifier #二值attributes分类
        self.class_index = None #二值attributes分类的index号

##################################################
# function to read in data from the .csv files
##################################################
def read_data(dataset, datafile, datatypes):
    print "Reading data..."
    f = open(datafile)
    original_file = f.read()
    rowsplit_data = original_file.splitlines() #返回包含各行的列表
    dataset.examples = [rows.split(',') for rows in rowsplit_data] #使用notepad++之类打开就发现，excel其实是以,分隔的

    #list attributes
    dataset.attributes = dataset.examples.pop(0) #属性是第一行，pop返回列表中删除的对象

    
    #create array that indicates whether each attribute is a numerical value or not
    attr_type = open(datatypes)  #属性值类型在另一个文件中，只有一行数据
    orig_file = attr_type.read()
    dataset.attr_types = orig_file.split(',') #使用notepad++之类打开就发现，excel其实是以,分隔的

##################################################
# Preprocess dataset
##################################################
def preprocess2(dataset):
    print "Preprocessing data..."

    class_values = [example[dataset.class_index] for example in dataset.examples] #example的分类值class_values
    class_mode = Counter(class_values) #分类值class_values频率字典
    class_mode = class_mode.most_common(1)[0][0] #分类值频率元组列表第一项也就是最频繁的class_values
                         
    for attr_index in range(len(dataset.attributes)):

        ex_0class = filter(lambda x: x[dataset.class_index] == '0', dataset.examples) #过滤得到分类值class_values为0的项
        values_0class = [example[attr_index] for example in ex_0class]  #过滤之后的属性值
                           
        ex_1class = filter(lambda x: x[dataset.class_index] == '1', dataset.examples) #过滤得到分类值class_values为1的项
        values_1class = [example[attr_index] for example in ex_1class]  #过滤之后的属性值
                
        values = Counter(values_0class)  #过滤之后的属性值频率字典
        value_counts = values.most_common() #过滤之后的属性值频率元组列表
        
        mode0 = values.most_common(1)[0][0] #过滤之后的属性值频率元组列表第一项也就是最频繁的values
        if mode0 == '?':
            mode0 = values.most_common(2)[1][0] #缺失值则取第二

        values = Counter(values_1class)
        mode1 = values.most_common(1)[0][0]
        
        if mode1 == '?':
            mode1 = values.most_common(2)[1][0]

        mode_01 = [mode0, mode1] #取0 和 1 的 属性值频率元组列表第一项也就是最频繁的values列表

        attr_modes = [0]*len(dataset.attributes) #初始化0 
        attr_modes[attr_index] = mode_01 #属性(0,1)列表
        
        for example in dataset.examples:
            if (example[attr_index] == '?'): #缺失值处理，根据本身类别来取众数
                if (example[dataset.class_index] == '0'):
                    example[attr_index] = attr_modes[attr_index][0] 
                elif (example[dataset.class_index] == '1'):
                    example[attr_index] = attr_modes[attr_index][1]
                else:
                    example[attr_index] = class_mode

        #convert attributes that are numeric to floats
        for example in dataset.examples:
            for x in range(len(dataset.examples[0])):
                #if dataset.attributes[x] == 'True': 
				#修正 这里是不是写错了 attr_types
                if dataset.attr_types[x] == 'True':
				    example[x] = float(example[x])

##################################################
# tree node class that will make up the tree
##################################################
class treeNode():
    def __init__(self, is_leaf, classification, attr_split_index, attr_split_value, parent, upper_child, lower_child, height):
        self.is_leaf = True #该节点是否为叶节点
        self.classification = None #该节点的样本归类
        self.attr_split = None #最优划分属性
        self.attr_split_index = None #最优划分属性index
        self.attr_split_value = None #最优划分属性value
        self.parent = parent #父节点
        self.upper_child = None #上子树节点
        self.lower_child = None #下子树节点
        self.height = None #节点深度，根为0

##################################################
# compute tree recursively
##################################################

# initialize Tree
    # if dataset is pure (all one result) or there is other stopping criteria then stop
    # for all attributes a in dataset
        # compute information-theoretic criteria if we split on a
    # abest = best attribute according to above
    # tree = create a decision node that tests abest in the root
    # dv (v=1,2,3,...) = induced sub-datasets from D based on abest
    # for all dv
        # tree = compute_tree(dv)
        # attach tree to the corresponding branch of Tree
    # return tree 

def compute_tree(dataset, parent_node, classifier):
    node = treeNode(True, None, None, None, parent_node, None, None, 0)
    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1 #树高度+1

    ones = one_count(dataset.examples, dataset.attributes, classifier) #统计属性值为1的样本数
	#递归返回，情形(1) 样本全为同一类别
    if (len(dataset.examples) == ones): #全部为1
        node.classification = 1 #该节点的样本归类
        node.is_leaf = True
        return node
    elif (ones == 0): #全部非1
        node.classification = 0 #该节点的样本归类
        node.is_leaf = True
        return node
    else:
        node.is_leaf = False
    attr_to_split = None # The index of the attribute we will split on 最优划分属性index
    max_gain = 0 # The gain given by the best attribute 最大收益
    split_val = None #属性分割值
    min_gain = 0.01
    dataset_entropy = calc_dataset_entropy(dataset, classifier) #计算数据集信息熵
    for attr_index in range(len(dataset.examples[0])):

        if (dataset.attributes[attr_index] != classifier):
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in dataset.examples] # these are the values we can split on, now we must find the best one 列表
            attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values 集合
            if(len(attr_value_list) > 100): #属性值分散
                attr_value_list = sorted(attr_value_list) #排序
                total = len(attr_value_list)
                ten_percentile = int(total/10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x*ten_percentile]) #换序每隔十取一
                attr_value_list = new_list

            for val in attr_value_list:
                # calculate the gain if we split on this value
                # if gain is greater than local_max_gain, save this gain and this value
                local_gain = calc_gain(dataset, dataset_entropy, val, attr_index) # calculate the gain if we split on this value
  
                if (local_gain > local_max_gain):
                    local_max_gain = local_gain
                    local_split_val = val
            #到这里已经找到最优属性切割值
            if (local_max_gain > max_gain):
                max_gain = local_max_gain
                split_val = local_split_val
                attr_to_split = attr_index
    #到这已经找到最优划分属性和最优属性切割值
    #attr_to_split is now the best attribute according to our gain metric
	#递归返回，情形(2) 
    if (split_val is None or attr_to_split is None):
        print "Something went wrong. Couldn't find an attribute to split on or a split value."
    elif (max_gain <= min_gain or node.height > 20): #信息增益趋于0或者树高度超过20 这里算正则项

        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier) #叶节点直接以多数类作为标记

        return node

    node.attr_split_index = attr_to_split #确定该节点划分属性index
    node.attr_split = dataset.attributes[attr_to_split] #确定该节点划分属性index
    node.attr_split_value = split_val #确定该节点划分属性val
    # currently doing one split per node so only two datasets are created
	#以下切分出上下分支数据集
    upper_dataset = data(classifier) #data对象
    lower_dataset = data(classifier) #data对象
    upper_dataset.attributes = dataset.attributes
    lower_dataset.attributes = dataset.attributes
    upper_dataset.attr_types = dataset.attr_types
    lower_dataset.attr_types = dataset.attr_types
    for example in dataset.examples:
        if (attr_to_split is not None and example[attr_to_split] >= split_val): 
            upper_dataset.examples.append(example)
        elif (attr_to_split is not None):
            lower_dataset.examples.append(example)
    #以下递归的以新的节点产生子树
    node.upper_child = compute_tree(upper_dataset, node, classifier)
    node.lower_child = compute_tree(lower_dataset, node, classifier)

    return node

##################################################
# Classify dataset
##################################################
def classify_leaf(dataset, classifier):
    ones = one_count(dataset.examples, dataset.attributes, classifier)
    total = len(dataset.examples)
    zeroes = total - ones
    if (ones >= zeroes):
        return 1
    else:
        return 0

##################################################
# Calculate the entropy of the current dataset
#二分类标准信息熵计算
##################################################
def calc_dataset_entropy(dataset, classifier):
    ones = one_count(dataset.examples, dataset.attributes, classifier)
    total_examples = len(dataset.examples);

    entropy = 0
    p = ones / total_examples
    if (p != 0):
        entropy += p * math.log(p, 2)
    p = (total_examples - ones)/total_examples
    if (p != 0):
        entropy += p * math.log(p, 2)

    entropy = -entropy
    return entropy

##################################################
# Calculate the gain of a particular attribute split
##################################################
def calc_gain(dataset, entropy, val, attr_index):
    classifier = dataset.attributes[attr_index]
    attr_entropy = 0
    total_examples = len(dataset.examples);
    gain_upper_dataset = data(classifier) #data对象
    gain_lower_dataset = data(classifier) #data对象
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    gain_upper_dataset.attr_types = dataset.attr_types
    gain_lower_dataset.attr_types = dataset.attr_types
    for example in dataset.examples:
        if (example[attr_index] >= val): #二分切割
            gain_upper_dataset.examples.append(example)
        elif (example[attr_index] < val):
            gain_lower_dataset.examples.append(example)

    if (len(gain_upper_dataset.examples) == 0 or len(gain_lower_dataset.examples) == 0): #Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
        return -1
    #以下是标准的信息增益计算公式
    attr_entropy += calc_dataset_entropy(gain_upper_dataset, classifier)*len(gain_upper_dataset.examples)/total_examples
    attr_entropy += calc_dataset_entropy(gain_lower_dataset, classifier)*len(gain_lower_dataset.examples)/total_examples
    #'''#修正
    attr_iv = 0
    attr_p=len(gain_upper_dataset.examples)/total_examples
    attr_iv += attr_p * math.log(attr_p, 2)
    attr_p=len(gain_lower_dataset.examples)/total_examples
    attr_iv += attr_p * math.log(attr_p, 2)
    attr_iv = -attr_iv
    return (entropy - attr_entropy)/attr_iv
	#'''#修正
	
    #return entropy - attr_entropy
	
##################################################
# count number of examples with classification "1"
##################################################
def one_count(instances, attributes, classifier):
    count = 0
    class_index = None
    #find index of classifier
    for a in range(len(attributes)):
        if attributes[a] == classifier:
            class_index = a
        else:
            class_index = len(attributes) - 1
    for i in instances:
        if i[class_index] == "1":
            count += 1
    return count

##################################################
# Prune tree
##################################################
def prune_tree(root, node, dataset, best_score):
    # if node is a leaf
    if (node.is_leaf == True):
        # get its classification
        classification = node.classification
        # run validate_tree on a tree with the nodes parent as a leaf with its classification
        node.parent.is_leaf = True
        node.parent.classification = node.classification
        if (node.height < 20):
            new_score = validate_tree(root, dataset)
        else:
            new_score = 0
  
        # if its better, change it
        if (new_score >= best_score):
            return new_score
        else:
            node.parent.is_leaf = False
            node.parent.classification = None
            return best_score
    # if its not a leaf
    else:
        # prune tree(node.upper_child)
        new_score = prune_tree(root, node.upper_child, dataset, best_score)
        # if its now a leaf, return
        if (node.is_leaf == True):
            return new_score
        # prune tree(node.lower_child)
        new_score = prune_tree(root, node.lower_child, dataset, new_score)
        # if its now a leaf, return
        if (node.is_leaf == True):
            return new_score

        return new_score

##################################################
# Validate tree
##################################################
def validate_tree(node, dataset):
    total = len(dataset.examples)
    correct = 0
    for example in dataset.examples:
        # validate example
        correct += validate_example(node, example)
    return correct/total

##################################################
# Validate example
##################################################
def validate_example(node, example):
    if (node.is_leaf == True):
        projected = node.classification
        actual = int(example[-1])
        if (projected == actual): 
            return 1
        else:
            return 0
    value = example[node.attr_split_index]
    if (value >= node.attr_split_value):
        return validate_example(node.upper_child, example)
    else:
        return validate_example(node.lower_child, example)

##################################################
# Test example
##################################################
def test_example(example, node, class_index):
    if (node.is_leaf == True):
        return node.classification
    else:
        if (example[node.attr_split_index] >= node.attr_split_value):
            return test_example(example, node.upper_child, class_index)
        else:
            return test_example(example, node.lower_child, class_index)

##################################################
# Print tree
##################################################
def print_tree(node):
    if (node.is_leaf == True):
        for x in range(node.height):
            print "\t",
        print "Classification: " + str(node.classification)
        return
    for x in range(node.height):
            print "\t",
    print "Split index: " + str(node.attr_split)
    for x in range(node.height):
            print "\t",
    print "Split value: " + str(node.attr_split_value)
    print_tree(node.upper_child)
    print_tree(node.lower_child)

##################################################
# Print tree in disjunctive normal form
##################################################
def print_disjunctive(node, dataset, dnf_string):
    if (node.parent == None):
        dnf_string = "( "
    if (node.is_leaf == True):
        if (node.classification == 1):
            dnf_string = dnf_string[:-3]
            dnf_string += ") ^ "
            print dnf_string,
        else:
            return
    else:
        upper = dnf_string + str(dataset.attributes[node.attr_split_index]) + " >= " + str(node.attr_split_value) + " V "
        print_disjunctive(node.upper_child, dataset, upper)
        lower = dnf_string + str(dataset.attributes[node.attr_split_index]) + " < " + str(node.attr_split_value) + " V "
        print_disjunctive(node.lower_child, dataset, lower)
        return

##################################################
# main function, organize data and execute functions based on input
# need to account for missing data
##################################################

def main():
    args = str(sys.argv)
    print args
    args = ast.literal_eval(args)
    print args
    if (len(args) < 2):
        print "You have input less than the minimum number of arguments. Go back and read README.txt and do it right next time!"
    elif (args[1][-4:] != ".csv"):
        print "Your training file (second argument) must be a .csv!"
    else:
        datafile = args[1]
        dataset = data("")
        if ("-d" in args):
            datatypes = args[args.index("-d") + 1]
        else:
            datatypes = 'datatypes.csv'
        read_data(dataset, datafile, datatypes)
        arg3 = args[2]
        if (arg3 in dataset.attributes):
            classifier = arg3
        else:
            classifier = dataset.attributes[-1]

        dataset.classifier = classifier

        #find index of classifier
        for a in range(len(dataset.attributes)):
            if dataset.attributes[a] == dataset.classifier:
                dataset.class_index = a
            else:
                dataset.class_index = range(len(dataset.attributes))[-1]
                
        unprocessed = copy.deepcopy(dataset)
        preprocess2(dataset)

        print "Computing tree..."
        root = compute_tree(dataset, None, classifier) 
        #print_tree(root)
        if ("-s" in args):
            print_disjunctive(root, dataset, "")
            print "\n"
        if ("-v" in args):
            datavalidate = args[args.index("-v") + 1]
            print "Validating tree..."

            validateset = data(classifier)
            read_data(validateset, datavalidate, datatypes)
            for a in range(len(dataset.attributes)):
                if validateset.attributes[a] == validateset.classifier:
                    validateset.class_index = a
                else:
                    validateset.class_index = range(len(validateset.attributes))[-1]
            preprocess2(validateset)
            best_score = validate_tree(root, validateset)
            all_ex_score = copy.deepcopy(best_score)
            print "Initial (pre-pruning) validation set score: " + str(100*best_score) +"%"
        if ("-p" in args):
            if("-v" not in args):
                print "Error: You must validate if you want to prune"
            else:
                post_prune_accuracy = 100*prune_tree(root, root, validateset, best_score)
                #print_tree(root)
                print "Post-pruning score on validation set: " + str(post_prune_accuracy) + "%"
        if ("-t" in args):
            datatest = args[args.index("-t") + 1]
            testset = data(classifier)
            read_data(testset, datatest, datatypes)
            for a in range(len(dataset.attributes)):
                if testset.attributes[a] == testset.classifier:
                    testset.class_index = a
                else:
                    testset.class_index = range(len(testset.attributes))[-1]
            print "Testing model on " + str(datatest)
            for example in testset.examples:
                example[testset.class_index] = '0'
            testset.examples[0][testset.class_index] = '1'
            testset.examples[1][testset.class_index] = '1'
            testset.examples[2][testset.class_index] = '?'
            preprocess2(testset)
            b = open('results.csv', 'w')
            a = csv.writer(b)
            for example in testset.examples:
                example[testset.class_index] = test_example(example, root, testset.class_index)
            saveset = testset
            saveset.examples = [saveset.attributes] + saveset.examples
            a.writerows(saveset.examples)
            b.close()
            print "Testing complete. Results outputted to results.csv"
            
if __name__ == "__main__":
	main()