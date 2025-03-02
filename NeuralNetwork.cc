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
    std::unordered_map<std::string, Neuron *> neuronLookup; // Fast lookup for neurons

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        if (line.find("Layer") != std::string::npos)
        {
            // Store the previous layer before starting a new one
            if (!layer.empty())
            {
                m_layers.push_back(layer);
                layer.clear();
            }
            continue;
        }

        // Neuron Definition: "NeuronName:Bias"
        size_t colonPos = line.find(":");
        if (colonPos != std::string::npos)
        {
            std::string name = line.substr(0, colonPos);
            std::string biasStr = line.substr(colonPos + 1);
            double bias = 0.0;

            try
            {
                bias = std::stod(biasStr);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error: Invalid bias value for neuron " << name << std::endl;
                continue;
            }

            std::vector<Connection> inputConnections; // Initially empty
            Neuron *newNeuron = new Neuron(1, name, inputConnections, bias);
            layer.push_back(newNeuron);
            neuronLookup[name] = newNeuron; // Store for fast lookup
            continue;
        }

        // Connection Definition: "InputNeuron->OutputNeuron:Weight"
        size_t arrowPos = line.find("->");
        size_t weightPos = line.find(":", arrowPos);
        if (arrowPos != std::string::npos && weightPos != std::string::npos)
        {
            std::string inputName = line.substr(0, arrowPos);
            std::string outputName = line.substr(arrowPos + 2, weightPos - (arrowPos + 2));
            std::string weightStr = line.substr(weightPos + 1);
            double weight = 0.0;

            try
            {
                weight = std::stod(weightStr);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error: Invalid weight value -> " << line << std::endl;
                continue;
            }

            // Create connection
            if (neuronLookup.find(inputName) != neuronLookup.end() &&
                neuronLookup.find(outputName) != neuronLookup.end())
            {
                Connection connection;
                connection.weight = weight;
                connection.input = neuronLookup[inputName];

                neuronLookup[outputName]->inputConnections.push_back(connection);
            }
            else
            {
                std::cerr << "Error: Neuron(s) not found for connection -> " << line << std::endl;
            }
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
void NeuralNetwork::save(const std::string &filename) const
{
    std::ofstream file(filename);
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