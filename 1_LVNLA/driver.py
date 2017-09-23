from datagen import *
from networks2 import *
from networks import *

import pandas


'''
def generate():
    result = gen_foolproof(1000)
    save("foolproof.csv", result)
    
    result = gen_basic_addition(1000, [0,50], [0,50])
    save("addition.csv", result)

    result = gen_basic_exp(1000, [0,50], [0, 4])
    save("exp.csv", result)
    
    result = gen_circle_class(1000, [-50, 50], [-50, 50], [-10,10], [-10,10], [1,20])
    save("circle.csv", result)

    result = gen_quadratic(1000, [-10, 10], [-10, 10], [10, 10])
    save("quad.csv", result)

generate()
'''


data = pandas.read_csv("foolproof.csv",header=None)

inputs = []
out = []

for i in range(len(data[0])):
    inputs.append([data[0][i], data[1][i]])
    out.append([data[2][i], data[3][i]])

net = FNN([2, 4, 2], "summary/", "simple", tf.nn.sigmoid, learning_rate=.05)
net.train(inputs, out, 1000, 100)
#print(net.predict([[0,0], [0,1]]))
net.stopSession()



data_add = pandas.read_csv("addition.csv", header=None)
inputs = []
out = []

for i in range(len(data_add[0])):
    inputs.append([data_add[0][i], data_add[1][i]])
    out.append([data_add[2][i]])

net = FNN([2, 4, 1], "summary/", "addition", tf.nn.relu, regression=True)
net.train(inputs, out, 1000, 100)
#print(net.predict([[7,6]]))
net.stopSession()


data_exp = pandas.read_csv("exp.csv", header=None)
inputs = []
out = []

for i in range(len(data_exp[0])):
    inputs.append([data_exp[0][i], data_exp[1][i]])
    out.append([data_exp[2][i]])

net = FNN([2, 4, 1], "summary/", "exp", tf.nn.relu, regression=True)
net.train(inputs, out, 1000, 100)
#print(net.predict([[7,6]]))
net.stopSession()


data_circle = pandas.read_csv("circle.csv", header=None)
inputs = []
out = []

for i in range(len(data_circle[0])):
    inputs.append([data_circle[0][i], data_circle[1][i], data_circle[2][i], data_circle[3][i], data_circle[4][i]])
    out.append([data_circle[5][i], data_circle[6][i]])

net = FNN([5, 4, 2], "summary/", "circle", tf.nn.relu, regression=False)
net.train(inputs, out, 1000, 100)
#print(net.predict([[7,6]]))
net.stopSession()


data_quad = pandas.read_csv("quad.csv", header=None)
inputs = []
out = []

for i in range(len(data_quad[0])):
    inputs.append([data_quad[0][i], data_quad[1][i], data_quad[2][i]])
    out.append([data_quad[3][i], data_quad[4][i]])

net = FNN([3, 4, 2], "summary/", "quad", tf.nn.relu, regression=False)
net.train(inputs, out, 1000, 100)
#print(net.predict([[7,6]]))
net.stopSession()
