#include "NeuralNetwork.h"
#include <iostream>
#include <vector>

using namespace std;
double getFRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void train(vector<vector<double>> inputs, vector<vector<double>> outputs, vector<vector<Neuron *>> &layers, double learningRate)
{
    for (int i = 0; i < inputs.size(); i++)
    {
        vector<double> input = inputs[i];
        vector<double> output = outputs[i];

        for (int j = 0; j < input.size(); j++)
        {
            layers[0][j]->outputs[0] = input[j];
        }

        for (int j = 1; j < layers.size(); j++)
        {
            for (Neuron *neuron : layers[j])
            {
                neuron->feedForward();
            }
        }

        for (int j = 0; j < output.size(); j++)
        {
            layers[layers.size() - 1][j]->addError(output[j] - layers[layers.size() - 1][j]->outputs[0]);
        }

        for (int j = layers.size() - 1; j > 0; j--)
        {
            for (Neuron *neuron : layers[j])
            {
                neuron->backProp(learningRate);
            }
        }
    }
}

void test(vector<vector<double>> inputs, vector<vector<double>> outputs, vector<vector<Neuron *>> &layers)
{
    for (int i = 0; i < inputs.size(); i++)
    {
        vector<double> input = inputs[i];
        vector<double> output = outputs[i];

        for (int j = 0; j < input.size(); j++)
        {
            layers[0][j]->outputs[0] = input[j];
        }

        for (int j = 1; j < layers.size(); j++)
        {
            for (Neuron *neuron : layers[j])
            {
                neuron->feedForward();
            }
        }

        for (int j = 0; j < output.size(); j++)
        {
            cout << "Expected: " << output[j] << " Actual: " << layers[layers.size() - 1][j]->outputs[0] << endl;
        }
    }
}

/*
void printNetwork(vector<vector<Neuron *>> &layers)
{
    for (int i = 0; i < layers.size(); i++)
    {
        cout << "Layer " << i << endl;
        for (Neuron *neuron : layers[i])
        {
            cout << neuron->getRepresentation();
        }
    }
} */

int main(int argc, char **argv)
{

    vector<int> *topology = new vector<int>();
    topology->push_back(2);
    topology->push_back(3);
    topology->push_back(6);
    topology->push_back(4);
    topology->push_back(1);

    NeuralNetwork *nn = new NeuralNetwork(topology, 0.01);

    int numTrainingSets = 10000;
    int numSilentTests = 100;
    int numPrintTests = 5;
    int numEpochs = 1000;
    int batchSize = 3;
    double learningRate = 0.1;
    vector<int> printResults = {0, 5, 10};
    nn->printNetwork();
    for (int epoch = 0; epoch < numEpochs; epoch++)
    {
        cout << "------Epoch " << epoch << "-----" << endl;
        vector<vector<vector<double>>> inputs;
        vector<vector<vector<double>>> outputs;

        for (int i = 0; i < numTrainingSets; i++)
        {
            vector<vector<double>> inputBatch = {};
            vector<vector<double>> outputBatch = {};
            for (int b = 0; b < batchSize; b++)
            {
                vector<double> input = {getFRand(0.5, 1), getFRand(0, 0.5)};
                vector<double> output = {input[0] - input[1]};
                inputBatch.push_back(input);
                outputBatch.push_back(output);
            }
            inputs.push_back(inputBatch);
            outputs.push_back(outputBatch);
        }

        nn->train(inputs, outputs);
        int numTestSets = numSilentTests;
        bool print = false;
        for (int i : printResults)
        {
            if (epoch == i)
            {
                print = true;
                numTestSets = numPrintTests;
            }
        }
        vector<vector<double>> inputBatch = {};
        vector<vector<double>> outputBatch = {};
        for (int t = 0; t < numTestSets; t++)
        {

            vector<double> input = {getFRand(0.5, 1), getFRand(0, 0.5)};
            vector<double> output = {input[0] - input[1]};
            inputBatch.push_back(input);
            outputBatch.push_back(output);
        }

        double averageError = nn->test(&inputBatch, &outputBatch, print);
        cout << "Average error: " << averageError << endl;
    }

    return 0;
}