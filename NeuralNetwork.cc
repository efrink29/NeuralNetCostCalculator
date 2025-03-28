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
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    std::vector<Neuron *> layer;
    std::unordered_map<std::string, Neuron *> neuronLookup;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        if (line.find("Layer") != std::string::npos)
        {
            if (!layer.empty())
            {
                m_layers.push_back(layer);
                layer.clear();
            }
        }
        else
        {
            std::string name = line.substr(0, line.find(":"));
            double bias = std::stod(line.substr(line.find(":") + 1));

            std::vector<Connection> inputConnections;
            while (std::getline(file, line) && line.find("End Inputs") == std::string::npos)
            {
                std::string inputName = line.substr(0, line.find(":"));
                double weight = std::stod(line.substr(line.find(":") + 1));
                inputConnections.push_back({weight, neuronLookup[inputName]});
            }
            Neuron *neuron = new Neuron(1, name, inputConnections, bias);
            layer.push_back(neuron);
            neuronLookup[name] = neuron;
        }
    }

    // Push the last layer if needed
    if (!layer.empty())
    {
        m_layers.push_back(layer);
    }

    file.close();
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
void NeuralNetwork::save(const std::string &filename)
{
    std::ofstream file(filename);
    unsigned long computations = getComputations();
    file << "Cost: " << computations << std::endl;
    for (int i = 0; i < m_layers.size(); i++)
    {
        file << "Layer " << i << std::endl;
        for (Neuron *neuron : m_layers[i])
        {
            file << neuron->getRepresentation();
        }
    }
    file.close();
}

void NeuralNetwork::train(const std::vector<std::vector<std::vector<double>>> &inputVals, const std::vector<std::vector<std::vector<double>>> &targetVals)
{
    // Set batch size
    for (int layerNum = 0; layerNum < m_layers.size(); ++layerNum)
    {
        for (int neuronNum = 0; neuronNum < m_layers[layerNum].size(); ++neuronNum)
        {
            m_layers[layerNum][neuronNum]->changeBatchSize(inputVals[0].size());
        }
    }
    for (int i = 0; i < inputVals.size(); i++)
    {
        const std::vector<std::vector<double>> &input = inputVals[i];
        const std::vector<std::vector<double>> &output = targetVals[i];
        for (int j = 0; j < input.size(); j++)
        {
            if (input[j].size() != m_layers[0].size())
            {
                std::cerr << "Error: Input size does not match network input size i:" << i << " j:" << j << std::endl;
                return;
            }
        }
        feedForward(input);
        backProp(output);
    }
}

double NeuralNetwork::test(std::vector<std::vector<double>> *inputVals,
                           std::vector<std::vector<double>> *resultVals,
                           bool printResults)
{
    double avgError = 0.0;
    const size_t numSamples = inputVals->size();
    const size_t outputSize = m_layers.back().size();

    for (auto &layer : m_layers)
    {
        for (Neuron *neuron : layer)
        {
            neuron->changeBatchSize(1);
        }
    }

    for (size_t i = 0; i < numSamples; i++)
    {
        std::vector<double> input = (*inputVals)[i];
        std::vector<double> output = (*resultVals)[i];

        feedForward({input});
        std::vector<double> result = getTestResults();

        double error = 0.0;
        for (size_t j = 0; j < outputSize; j++)
        {
            error += std::abs(output[j] - result[j]);
        }
        error /= outputSize;

        avgError += error;
        if (printResults)
        {
            std::cout << "Expected:Actual - ";
            std::cout << std::setprecision(3) << std::fixed;
            for (size_t j = 0; j < outputSize; j++)
            {
                if (output[j] < 0.001)
                {
                    std::cout << "0.000:";
                }
                else
                {
                    std::cout << output[j] << ":";
                }
                if (result[j] < 0.001)
                {
                    std::cout << "0.000  ";
                }
                else
                {
                    std::cout << result[j] << "  ";
                }
            }
            std::cout << std::endl;
        }
    }
    avgError /= numSamples;
    return avgError;
}

void NeuralNetwork::randomizeWeightsAndBias()
{
    for (int i = 1; i < m_layers.size(); i++)
    {
        for (Neuron *neuron : m_layers[i])
        {
            neuron->randomizeWeightsAndBias();
        }
    }
}

unsigned long NeuralNetwork::pruneNetwork(double proportion)
{
    unsigned long pruned = 0;
    for (unsigned int i = 1; i < m_layers.size(); i++)
    {
        int numConnections = 0;
        for (Neuron *neuron : m_layers[i])
        {
            numConnections += neuron->inputConnections.size();
        }
        int numPruned = numConnections * proportion;
        for (int j = 0; j < numPruned; j++)
        {
            // Find the connection with lowest absolute weight
            double minWeight = 1000000;
            Neuron *minNeuron = nullptr;
            int minConnection = -1;
            for (Neuron *neuron : m_layers[i])
            {
                for (int k = 0; k < neuron->inputConnections.size(); k++)
                {
                    Connection connection = neuron->inputConnections[k];
                    if (abs(connection.weight) < minWeight)
                    {
                        minWeight = abs(connection.weight);
                        minNeuron = neuron;
                        minConnection = k;
                    }
                }
            }
            if (minConnection != -1)
            {
                minNeuron->inputConnections.erase(minNeuron->inputConnections.begin() + minConnection);
                pruned++;
            }
        }
    }
    return pruned;
}

void NeuralNetwork::feedForward(std::vector<std::vector<double>> inputVals)
{

    for (int i = 0; i < inputVals.size(); i++)
    {
        std::vector<double> input = inputVals[i];
        for (int j = 0; j < input.size(); j++)
        {
            if (i >= m_layers[0][j]->getBatchSize())
            {
                std::cerr << "Error: Input size does not match network input size i:" << i << " batchSize:" << m_layers[0][j]->getBatchSize() << std::endl;
                return;
            }
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
    std::vector<double> resultVals;
    resultVals.reserve(m_layers.back().size());
    for (Neuron *neuron : m_layers.back())
    {
        resultVals.push_back(neuron->outputs[0]);
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

unsigned long NeuralNetwork::getComputations()
{
    unsigned long computations = 0;
    for (int i = 1; i < m_layers.size(); i++)
    {
        for (Neuron *neuron : m_layers[i])
        {
            computations += neuron->inputConnections.size() + 1;
        }
    }
    return computations;
}