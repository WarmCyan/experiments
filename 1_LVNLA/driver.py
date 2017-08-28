from datagen import *
from networks import *

import pandas

def generate():
    result = gen_basic_addition(1000, [0,50], [0,50])
    save("testing.csv", result)

    result = gen_foolproof(1000)
    save("foolproof.csv", result)


generate()

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
