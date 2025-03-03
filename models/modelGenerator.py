import os


def generate_quad_sample_mnist_model(fileName):
    model = open(fileName, "w")
    #Create input layer 28 * 28 grid
    for i in range(28):
        for j in range(28):
            model.write("input" + str(i) + "_" + str(j) + ":0.0\nEnd Inputs\n")
    
    # Create hidden layer 1
    for i in range(14):
        for j in range(14):
            model.write("hidden1_" + str(i) + "_" + str(j) + ":0.0\n")
            model.write("input" + str(i*2) + "_" + str(j*2) + ":0.5\n")
            model.write("input" + str(i*2) + "_" + str(j*2+1) + ":0.5\n")
            model.write("input" + str(i*2+1) + "_" + str(j*2) + ":0.5\n")
            model.write("input" + str(i*2+1) + "_" + str(j*2+1) + ":0.5\n")
            model.write("End Inputs\n")
    
    # Create hidden layer 2
    for i in range(7):
        for j in range(7):
            model.write("hidden2_" + str(i) + "_" + str(j) + ":0.0\n")
            model.write("hidden1_" + str(i*2) + "_" + str(j*2) + ":0.5\n")
            model.write("hidden1_" + str(i*2) + "_" + str(j*2+1) + ":0.5\n")
            model.write("hidden1_" + str(i*2+1) + "_" + str(j*2) + ":0.5\n")
            model.write("hidden1_" + str(i*2+1) + "_" + str(j*2+1) + ":0.5\n")
            model.write("End Inputs\n")
            
    # Create output layer
    for i in range(10):
        model.write("output" + str(i) + ":0.0\n")
        for j in range(7):
            for k in range(7):
                model.write("hidden2_" + str(j) + "_" + str(k) + ":0.5\n")
        model.write("End Inputs\n")
    model.close()
    
generate_quad_sample_mnist_model("quad_sample_mnist.nn")
        