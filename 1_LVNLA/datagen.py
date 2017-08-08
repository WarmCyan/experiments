import csv
import random as r

# NOTE: domain format: [min,max] (array)
# NOTE: size refers to number of bits

def rand(domain):
    return r.randint(domain[0], domain[1])

def randbinary(size):
    

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
    pass

def gen_quadratic_class(count, a_domain, b_domain, c_domain):
    pass
