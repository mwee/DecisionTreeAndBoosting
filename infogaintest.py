from utils import *
from dtree import *

def information_gain(attr, examples):
   def I(examples):
       target = dataset.target
       # new
       weightSumList = []
       for v in dataset.values[target]:
           weightSumList.append(addWeights(examples, target, v))
       return information_content(weightSumList)
       #return information_content([count(target, v, examples)
       #                            for v in dataset.values[target]])
   def addWeights(examples, attr=-1, val=-1):
       total = 0.0
       if attr!=-1:
           for example in examples:
               if example.attrs[attr] == val:
                   total += example.weight
       else:
           for example in examples:
               total+=example.weight
       #print total
       #print examples
       return total    
   #N = float(len(examples))
   W = float(addWeights(examples))
   #print W
   #W = sum(weightSumList)
   remainder = 0
   for (v, examples_i) in split_by(attr, examples):
       weightSumAttr = addWeights(examples, attr, v)
       remainder += (weightSumAttr / W) * I(examples_i)
   return I(examples) - remainder

def split_by(attr, examples=None):
   "Return a list of (val, examples) pairs for each val of attr."
   if examples == None:
       examples = dataset.examples
   return [(v, [e for e in examples if e.attrs[attr] == v])
           for v in dataset.values[attr]]
    
def information_content(values):
    "Number of bits to represent the probability distribution in values."
    # If the values do not sum to 1, normalize them to make them a Prob. Dist.
    values = removeall(0, values)
    W = float(sum(values))
    if W != 1.0: values = [v/W for v in values] # normalize weights
    return sum([- v * log2(v) for v in values])

def main():

    f = open("data_test.csv")
    data = parse_csv(f.read(), " ")
    dataset = DataSet(data)

    print str(information_gain(0,dataset.examples))

main() 