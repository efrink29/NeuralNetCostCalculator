#ifndef NEURON_H
#define NEURON_H

#include <cstdlib>
#include <string>
#include <vector>

class Neuron;

struct Connection
{
    double weight;
    Neuron *input;
};

class Neuron
{
public:
    double avgError;
    double *outputs;
    std::string name;
    void feedForward();

private:
    int batchSize;
    std::vector<Connection> inputConnections;
};

#endif // NEURON_H