#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>
#include <unordered_map>
#include <sstream>

#include "Neuron.h"

class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<int> *topology, double learningRate); // Create a neural network with the given topology
    NeuralNetwork(const std::string &filename);                     // Load a neural network from a file
    ~NeuralNetwork();                                               // Destructor
    void save(const std::string &filename);                         // Save a neural network to a file
    void train(const std::vector<std::vector<std::vector<double>>> &inputVals, const std::vector<std::vector<std::vector<double>>> &targetVals);
    double test(std::vector<std::vector<double>> *inputVals, std::vector<std::vector<double>> *resultVals, bool printResults = false);
    double getError() const { return m_error; }
    void printNetwork();
    void setLearningRate(double learningRate) { this->learningRate = learningRate; }
    unsigned long getComputations();
    void randomizeWeightsAndBias();
    unsigned long pruneNetwork(double proportion);

private:
    std::vector<std::vector<Neuron *>> m_layers;
    double m_error;
    double learningRate;
    void feedForward(std::vector<std::vector<double>> inputVals);
    void backProp(const std::vector<std::vector<double>> &targetVals);
    std::vector<double> getTestResults();
};

#endif // NEURALNETWORK_H