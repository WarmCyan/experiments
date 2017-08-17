from datagen import *

result = gen_basic_addition(1000, [0,50], [0,50])
save("testing.csv", result)

#result = gen_basic_addition(50, [0,10], [0,10])
##save("test.csv", result)
#result = gen_basic_exp(10, [0,10], [0,10])
#print(result)
#result = gen_quadratic(10, [0,10],[0,10],[0,10])
#result = gen_binary_exp(10, 4,4,8)
#print(result)
