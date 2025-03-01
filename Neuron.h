#ifndef NEURON_H
#define NEURON_H

#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>

class Neuron;

struct Connection
{
    double weight;
    Neuron *input;
};

enum class NeuronType
{
    input,
    hidden,
    output
};

class Neuron
{
public:
    Neuron(int batchSize, std::string name, std::vector<Neuron *> &inputs);                // Constructor with random weights
    Neuron(int batchSize, std::string name, std::vector<Connection> &inputs, double bias); // Constructor with given weights
    ~Neuron();
    std::string getRepresentation();
    double *outputs;
    std::string name;
    void feedForward();
    bool addError(double error);
    void backProp(double learningRate);
    void changeBatchSize(int batchSize);
    double getAverageOutput();

private:
    int batchSize;
    double error;
    double bias;
    int numOutputConnections;
    std::vector<Connection> inputConnections;
    double activationFunction(double x);
    double activationFunctionDerivative(double x);
};

#endif // NEURON_H