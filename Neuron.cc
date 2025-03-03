#include "Neuron.h"
#include <iostream>

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double randomWeight()
{
    return fRand(-1, 1);
}

Neuron::Neuron(int batchSize, std::string name, std::vector<Neuron *> &inputs)
{
    this->batchSize = batchSize;
    this->name = name;
    this->outputs = new double[batchSize];
    this->numOutputConnections = 0;
    this->error = 0;
    this->bias = (randomWeight() + 1.0) / 2.0;
    for (Neuron *input : inputs)
    {
        Connection connection;
        connection.weight = randomWeight();
        connection.input = input;
        this->inputConnections.push_back(connection);
    }
}

Neuron::Neuron(int batchSize, std::string name, std::vector<Connection> &inputs, double bias)
{
    this->batchSize = batchSize;
    this->name = name;
    this->outputs = new double[batchSize];
    this->numOutputConnections = 0;
    this->error = 0;
    this->bias = bias;
    for (Connection input : inputs)
    {
        this->inputConnections.push_back(input);
    }
}

Neuron::~Neuron()
{
    delete[] outputs;
}

std::string Neuron::getRepresentation()
{
    std::string representation = name + ":" + std::to_string(this->bias) + "\n";
    // representation += "Inputs:\n";
    for (Connection connection : inputConnections)
    {
        representation += "" + connection.input->name + ":" + std::to_string(connection.weight) + "\n";
    }
    representation += "End Inputs\n";
    return representation;
}

void Neuron::feedForward()
{
    for (int i = 0; i < batchSize; i++)
    {
        double sum = 0;
        for (Connection connection : inputConnections)
        {
            sum += connection.input->outputs[i] * connection.weight;
        }
        outputs[i] = activationFunction(sum + bias);
    }
    this->error = 0;
    this->numOutputConnections = 0;
}

bool Neuron::addError(double error)
{
    this->error += error;
    this->numOutputConnections++;
    return true;
}

void Neuron::backProp(double learningRate)
{
    double delta = activationFunctionDerivative(outputs[0]) * error;
    for (Connection &connection : inputConnections)
    {
        connection.input->addError(delta * connection.weight);
        connection.weight += delta * connection.input->getAverageOutput() * learningRate;
    }
    this->error = 0;
    this->numOutputConnections = 0;
}

void Neuron::changeBatchSize(int batchSize)
{
    delete[] outputs;
    this->batchSize = batchSize;
    this->outputs = new double[batchSize];
}

double Neuron::getAverageOutput()
{
    double sum = 0;
    for (int i = 0; i < batchSize; i++)
    {
        sum += outputs[i];
    }
    return sum / batchSize;
}

double Neuron::activationFunction(double x)
{
    return 1 / (1 + std::exp(-x));
}

double Neuron::activationFunctionDerivative(double x)
{
    return x * (1 - x);
}