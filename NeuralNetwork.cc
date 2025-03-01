#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<int> *topology, double learningRate)
{
    this->learningRate = learningRate;
    int numLayers = topology->size();
    for (int layerNum = 0; layerNum < numLayers; layerNum++)
    {
        std::vector<Neuron *> layer = std::vector<Neuron *>();
        std::vector<Neuron *> inputs;
        std::vector<Connection> inputConnections;
        if (layerNum > 0)
        {
            inputs = m_layers.back();
        }

        for (int neuronNum = 0; neuronNum < (*topology)[layerNum]; neuronNum++)
        {

            if (layerNum == 0)
            {
                std::string name = "Input-" + std::to_string(neuronNum);
                layer.push_back(new Neuron(1, name, inputConnections, 0.0));
            }
            else if (layerNum < numLayers - 1)
            {
                int hlayerNum = layerNum - 1;
                std::string name = "Hidden" + std::to_string(hlayerNum) + "-" + std::to_string(neuronNum);
                layer.push_back(new Neuron(1, name, inputs));
            }
            else
            {
                std::string name = "Output-" + std::to_string(neuronNum);
                layer.push_back(new Neuron(1, name, inputs));
            }
        }
        m_layers.push_back(layer);
    }
}

NeuralNetwork::NeuralNetwork(const std::string &filename)
{
}

NeuralNetwork::~NeuralNetwork()
{
    for (int i = 0; i < m_layers.size(); i++)
    {
        for (Neuron *neuron : m_layers[i])
        {
            delete neuron;
        }
    }
}
void NeuralNetwork::save(const std::string &filename) const
{
}

void NeuralNetwork::train(const std::vector<std::vector<std::vector<double>>> &inputVals, const std::vector<std::vector<std::vector<double>>> &targetVals)
{
    for (int i = 0; i < inputVals.size(); i++)
    {
        std::vector<std::vector<double>> input = inputVals[i];
        std::vector<std::vector<double>> output = targetVals[i];
        // Set batch size
        for (int layerNum = 0; layerNum < m_layers.size(); ++layerNum)
        {
            for (int neuronNum = 0; neuronNum < m_layers[layerNum].size(); ++neuronNum)
            {
                m_layers[layerNum][neuronNum]->changeBatchSize(input.size());
            }
        }

        feedForward(input);
        backProp(output);
    }
}

double NeuralNetwork::test(std::vector<std::vector<double>> *inputVals, std::vector<std::vector<double>> *resultVals, bool printResults)
{
    double avgError = 0.0;
    // Set batch size
    for (int layerNum = 0; layerNum < m_layers.size(); ++layerNum)
    {
        for (int neuronNum = 0; neuronNum < m_layers[layerNum].size(); ++neuronNum)
        {
            m_layers[layerNum][neuronNum]->changeBatchSize(1);
        }
    }
    std::cout.precision(4);
    for (int i = 0; i < inputVals->size(); i++)
    {
        std::vector<std::vector<double>> testInput = std::vector<std::vector<double>>();
        testInput.push_back((*inputVals)[i]);
        feedForward(testInput);
        std::vector<double> results = getTestResults();
        for (int r = 0; r < results.size(); r++)
        {
            avgError += abs((*resultVals)[i][r] - results[0]);
        }
        if (printResults)
        {
            std::cout << "Results Test #" + std::to_string(i) << "Expected : Actual" << std::endl;
            for (int r = 0; r < results.size(); r++)
            {
                std::cout << (*resultVals)[i][r] << " : " << results[0] << std::endl;
            }
            std::cout << std::endl;
        }
    }
    return avgError / (double)inputVals->size();
}

void NeuralNetwork::feedForward(std::vector<std::vector<double>> inputVals)
{
    for (int i = 0; i < inputVals.size(); i++)
    {
        std::vector<double> input = inputVals[i];
        for (int j = 0; j < input.size(); j++)
        {
            m_layers[0][j]->outputs[i] = input[j];
        }
    }
    for (int j = 1; j < m_layers.size(); j++)
    {
        for (Neuron *neuron : m_layers[j])
        {
            neuron->feedForward();
        }
    }
}

void NeuralNetwork::backProp(const std::vector<std::vector<double>> &targetVals)
{
    double avgError = 0.0;
    for (int neuronNum = 0; neuronNum < m_layers.back().size(); neuronNum++)
    {
        double error = 0.0;
        for (int i = 0; i < targetVals.size(); i++)
        {
            error += targetVals[i][neuronNum] - m_layers.back()[neuronNum]->outputs[i];
        }
        m_layers.back()[neuronNum]->addError(error / (double)targetVals.size());
        avgError += error / (double)targetVals.size();
    }
    m_error = avgError / (double)m_layers.back().size();

    for (int j = m_layers.size() - 1; j > 0; j--)
    {
        for (Neuron *neuron : m_layers[j])
        {
            neuron->backProp(learningRate);
        }
    }
}

std::vector<double> NeuralNetwork::getTestResults()
{
    std::vector<double> resultVals = std::vector<double>();
    for (int i = 0; i < m_layers.back().size(); i++)
    {
        resultVals.push_back(m_layers.back()[i]->outputs[0]);
    }
    return resultVals;
}

void NeuralNetwork::printNetwork()
{
    for (int i = 0; i < m_layers.size(); i++)
    {
        std::cout << "Layer " << i << std::endl;
        for (Neuron *neuron : m_layers[i])
        {
            std::cout << neuron->getRepresentation();
        }
    }
}