from datagen import *
from networks2 import *

import pandas




def generate():
    result = gen_basic_addition(1000, [0,50], [0,50])
    save("testing.csv", result)

    result = gen_foolproof(1000)
    save("foolproof.csv", result)


generate()

data = pandas.read_csv("foolproof.csv",header=None)

inputs = []
out = []

for i in range(len(data[0])):
    inputs.append([data[0][i], data[1][i]])
    out.append([data[2][i], data[3][i]])

#import kiss
#kiss.run(inputs, out)

#run(inputs, out, activation=tf.nn.relu, regression=False, name='test/foolproof')


data_add = pandas.read_csv("testing.csv", header=None)
inputs = []
out = []

for i in range(len(data_add[0])):
    inputs.append([data_add[0][i], data_add[1][i]])
    out.append([data_add[2][i]])


run(inputs,out,activation=tf.nn.relu,name='test/addition', out_size=1)

'''
data = pandas.read_csv("foolproof.csv",header=None)
#in1 = list(data[0])
#in2 = list(data[1])
out = list(data[2])

#inputs = list(zip(data[0], data[1]))
#out = [(j,) for j in out]

inputs = []
out = []

for i in range(len(data[0])):
    inputs.append([data[0][i], data[1][i]])
    out.append([data[2][i]])

#print(str(inputs[0]) + " = " + str(out[0]))
#print(inputs)

net = FFNet([2,4,1], 2, False)
net.train(inputs, out, 250, 10)
result = net.predict([[1,1]])
print(result)


#result = gen_basic_addition(50, [0,10], [0,10])
##save("test.csv", result)
#result = gen_basic_exp(10, [0,10], [0,10])
#print(result)
#result = gen_quadratic(10, [0,10],[0,10],[0,10])
#result = gen_binary_exp(10, 4,4,8)
#print(result)
'''
