import csv
import math
import random as r

# NOTE: domain format: [min,max] (array)
# NOTE: size refers to number of bits

def rand(domain):
    return r.randint(domain[0], domain[1])

def randbinary(size):
    pass

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
    pass

def gen_binary_exp(count, i1_size, i2_size, output_size):
    pass

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
