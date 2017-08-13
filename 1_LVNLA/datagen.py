import csv
import math
import random as r

# NOTE: domain format: [min,max] (array)
# NOTE: size refers to number of bits

def rand(domain):
    return r.randint(domain[0], domain[1])

def randbinary(size):
     maxint = size**2 - 1  
     number = rand([0, maxint])
     binnumber = binarify(number, size)
     return (binnumber, number)

def binarify(number, size=-1):
    binarystring = "{0:b}".format(number)
    entry = []

    # pad with zeros as needed
    if size != -1:
        for i in range(0, size-len(binarystring)):
            entry.append(0)
        
    for char in binarystring:
        entry.append(int(char))
    return entry

def save(filename, data):
    print("Saving data to '" + str(filename) + "'...")
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)

        for entry in data:
            writer.writerow(entry)
    print("Saved!")
        

def gen_basic_addition(count, i1_domain, i2_domain):
    print("Generating basic addition dataset of size " + str(count) + "...")
    print("Domains: " + str(i1_domain) + "," + str(i2_domain))
    
    data = []
    
    for i in range(0,count):
        i1 = rand(i1_domain)
        i2 = rand(i2_domain)
        output = i1+i2

        data.append([i1, i2, output])

    print("Generated!")
    return data

def gen_basic_exp(count, i1_domain, i2_domain):
    print("Generating basic addition dataset of size " + str(count) + "...")
    print("Domains: " + str(i1_domain) + "," + str(i2_domain))
    
    data = []
    
    for i in range(0,count):
        i1 = rand(i1_domain)
        i2 = rand(i2_domain)
        output = i1**i2

        data.append([i1, i2, output])

    print("Generated!")
    return data

def gen_binary_addition(count, i1_size, i2_size, output_size):
    print("Generating basic binary addition dataset of size " + str(count) + "...")
    print("Sizes: " + str(i1_size) + "," + str(i2_size))
    
    data = []
    
    for i in range(0,count):

        validsize = False
        i1 = None 
        i2 = None 
        output = None 

        while not validsize:
            i1 = randbinary(i1_size)
            i2 = randbinary(i2_size)

            outputsum = i1[1]+i2[1]
            if outputsum <= output_size**2 - 1:
                output = binarify(outputsum, output_size)
                validsize = True

        row = []
        for bit in i1[0]:
            row.append(bit)
        for bit in i2[0]:
            row.append(bit)
        for bit in output:
            row.append(bit)
            
        data.append(row)

    print("Generated!")
    return data

def gen_binary_exp(count, i1_size, i2_size, output_size):
    print("Generating basic binary exponential dataset of size " + str(count) + "...")
    print("Sizes: " + str(i1_size) + "," + str(i2_size))
    
    data = []
    
    for i in range(0,count):

        validsize = False
        i1 = None 
        i2 = None 
        output = None 

        while not validsize:
            i1 = randbinary(i1_size)
            i2 = randbinary(i2_size)

            outputpow = i1[1]**i2[1]
            if outputpow <= output_size**2 - 1:
                output = binarify(outputpow, output_size)
                validsize = True

        row = []
        for bit in i1[0]:
            row.append(bit)
        for bit in i2[0]:
            row.append(bit)
        for bit in output:
            row.append(bit)
            
        data.append(row)

    print("Generated!")
    return data

def gen_circle_class(count, i1_domain, i2_domain, h_domain, k_domain, r_domain):
    print("Generating quadratic classification dataset of size " + str(count) + "...")
    print("Domains: " + str(i1_domain) + "," + str(i2_domain) + "," + str(h_domain) + "," + str(k_domain) + "," + str(r_domain))
    
    data = []

    for i in range(0, count):
        i1 = rand(i1_domain)
        i2 = rand(i2_domain)
        h = rand(h_domain)
        k = rand(k_domain)
        r = rand(r_domain)
        
        point = (i1-h)**2 + (i2-k)**2
        output = 0
        print(str([i1,i2,h,k,r]))
        print(str(point) + " " + str(r**2))
        if point <= r**2:
            output = 1
        
        data.append([i1,i2,h,k,r,output])
        
    print("Generated!")
    return data

# NOTE: for now, only going to do non-imaginary
def gen_quadratic(count, a_domain, b_domain, c_domain):
    print("Generating quadratic dataset of size " + str(count) + "...")
    print("Domains: " + str(a_domain) + "," + str(b_domain) + "," + str(c_domain))

    data = []

    for i in range(0, count):
        a = 0
        b = 0
        c = 0

        desc = -1
        while desc < 0:
            a = rand(a_domain)
            b = rand(b_domain)
            c = rand(b_domain)
            desc = b**2 - 4*a*c
            if a == 0: desc = -1
        
        output1 = (-b - math.sqrt(desc))/(2*a)
        output2 = (-b + math.sqrt(desc))/(2*a)

        data.append([a,b,c,output1,output2])
    
    print("Generated!")
    return data
