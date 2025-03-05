import os
import random
import time

def generate_quad_sample_mnist_model(fileName):
    model = open(fileName, "w")
    #Create input layer 28 * 28 grid
    model.write("Cost:0\n")
    model.write("Layer 0\n")
    for i in range(28):
        for j in range(28):
            model.write("input" + str(i) + "_" + str(j) + ":0.0\nEnd Inputs\n")
    
    # Create hidden layer 1
    model.write("Layer 1\n")
    for i in range(14):
        for j in range(14):
            bias = random.uniform(0, 1)
            model.write("hidden1_" + str(i) + "_" + str(j) + ":" + str(bias) + "\n")
            model.write("input" + str(i*2) + "_" + str(j*2) + ":" + str(random.uniform(-1,1))+ "\n")
            model.write("input" + str(i*2) + "_" + str(j*2+1) + ":" + str(random.uniform(-1,1))+ "\n")
            model.write("input" + str(i*2+1) + "_" + str(j*2) + ":" + str(random.uniform(-1,1))+ "\n")
            model.write("input" + str(i*2+1) + "_" + str(j*2+1) + ":" + str(random.uniform(-1,1))+ "\n")
            model.write("End Inputs\n")
    
    # Create hidden layer 2
    model.write("Layer 2\n")
    for i in range(7):
        for j in range(7):
            bias = random.uniform(0, 1)
            model.write("hidden2_" + str(i) + "_" + str(j) + ":" + str(bias) + "\n")
            for k in range(2):
                for l in range(2):
                    model.write("hidden1_" + str(i*2+k) + "_" + str(j*2+l) + ":" + str(random.uniform(-1,1))+ "\n")
            model.write("End Inputs\n")
            
    # Create output layer
    model.write("Layer 3\n")
    for i in range(10):
        bias = random.uniform(0, 1)
        model.write("output" + str(i) + ":" + str(bias) + "\n")
        
        for j in range(7):
            for k in range(7):
                model.write("hidden2_" + str(j) + "_" + str(k) + ":" + str(random.uniform(-1,1))+ "\n")
        model.write("End Inputs\n")
    model.close()

random.seed(time.time())
generate_quad_sample_mnist_model("models/quad_stack_mnist.nn")
        