#include "NeuralNetwork.h"
#include "mnistReader.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

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

void runSubtraction(NeuralNetwork *nn)
{
    int numTrainingSets = 100;
    int numSilentTests = 10;
    int numPrintTests = 1;
    int numEpochs = 10;
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
}

void runUserModeratedTraining(NeuralNetwork *nn)
{
    cout << "Enter number of training sets: ";
    int numTrainingSets;
    cin >> numTrainingSets;
    cout << "Enter number of silent tests: ";
    int numSilentTests;
    cin >> numSilentTests;
    cout << "Enter number of print tests: ";
    int numPrintTests;
    cin >> numPrintTests;
    cout << "Enter number of epochs: ";
    int numEpochs;
    cin >> numEpochs;
    cout << "Enter batch size: ";
    int batchSize;
    cin >> batchSize;
    cout << "Enter learning rate: ";
    double learningRate;
    cin >> learningRate;
    vector<int> printResults;
    cout << "Enter print results (enter -1 to stop): ";
    int printResult;
    cin >> printResult;
    while (printResult != -1)
    {
        printResults.push_back(printResult);
        cin >> printResult;
    }
    nn->printNetwork();
    // TODO implement this
}

vector<double> getOutputForLabel(int label)
{
    vector<double> output = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    output[label] = 1;
    return output;
}

void trainMnist(MNISTReader *reader, NeuralNetwork *nn, int numBatches, int batchSize, double learningRate)
{
    vector<vector<vector<double>>> inputs;
    vector<vector<vector<double>>> outputs;
    for (int i = 0; i < numBatches; i++)
    {
        vector<vector<double>> inputBatch = {};
        vector<vector<double>> outputBatch = {};
        for (int b = 0; b < batchSize; b++)
        {
            Image_Data *image1 = reader->getNextTrainImage();

            vector<double> input = image1->pixels;
            vector<double> output = getOutputForLabel(image1->label);

            inputBatch.push_back(input);
            outputBatch.push_back(output);
            delete image1;
        }
        inputs.push_back(inputBatch);
        outputs.push_back(outputBatch);
    }
    nn->train(inputs, outputs);
}

double testMnist(MNISTReader *reader, NeuralNetwork *nn, int numTests)
{
    vector<vector<double>> inputBatch = {};
    vector<vector<double>> outputBatch = {};
    bool numbersPresent[] = {false, false, false, false, false, false, false, false, false, false};
    for (int i = 0; i < numTests; i++)
    {
        Image_Data *image = reader->getNextTestImage();
        if (i % 10 == 0)
        {
            for (int j = 0; j < 10; j++)
            {
                numbersPresent[j] = false;
            }
        }
        if (!numbersPresent[image->label])
        {
            vector<double> input = image->pixels;
            vector<double> output = getOutputForLabel(image->label);
            // reader->printImage(image, true);
            inputBatch.push_back(input);
            outputBatch.push_back(output);
            numbersPresent[image->label] = true;
        }
        else
        {
            i--;
        }

        delete image;
    }
    return nn->test(&inputBatch, &outputBatch, true);
}

bool saveNetwork(NeuralNetwork *nn)
{
    cout << "Enter filename to save to: ";
    string filename;
    cin >> filename;

    // Check if file exists
    ifstream file(filename);
    if (file.good())
    {
        cout << "File already exists. Overwrite? (y/n): ";
        char response;
        cin >> response;
        if (response != 'y')
        {
            return false;
        }
    }
    nn->save(filename);
    return true;
}

bool saveNetwork(NeuralNetwork *nn, vector<double> averageErrors)
{
    cout << "Enter filename to save to: ";
    string filename;
    cin >> filename;

    // Check if file exists
    ifstream file(filename);
    if (file.good())
    {
        cout << "File already exists. Overwrite? (y/n): ";
        char response;
        cin >> response;
        if (response != 'y')
        {
            return false;
        }
    }
    nn->save(filename);
    ofstream errorFile(filename + ".error");
    unsigned long computations = nn->getComputations();
    errorFile << "Model Cost: " << computations << endl;
    for (double error : averageErrors)
    {
        errorFile << error << endl;
    }
    errorFile.close();
    return true;
}

int main(int argc, char **argv)
{

    srand(time(NULL));
    MNISTReader reader("data");

    Image_Data *image1 = reader.getNextTrainImage();
    Image_Data *image2 = reader.getNextTrainImage();

    cout << "Image 1: " << image1->pixels[(24 * 28) + 24] << endl;
    reader.printImage(image1, true);
    reader.printImage(image2, true);

    delete image1;
    delete image2;
    vector<int> *topology = new vector<int>();
    topology->push_back(28 * 28);
    topology->push_back(28 * 2);
    topology->push_back(10);

    NeuralNetwork *nn = new NeuralNetwork(topology, 0.05);

    // Need to fix constructor
    // trainMnist(&reader, nn, 10000, 1, 0.1);
    // double avgError = testMnist(&reader, nn, 10);
    // cout << "Average error: " << avgError << endl;
    //  runSubtraction(nn);
    vector<double> averageErrors;
    int numEpochs = 10;
    for (int i = 0; i < numEpochs; i++)
    {
        trainMnist(&reader, nn, 10000, 1, 0.1);
        double avgError = testMnist(&reader, nn, 40);
        cout << "Average error: " << avgError << endl;
        averageErrors.push_back(avgError);
    }
    while (!saveNetwork(nn, averageErrors))
    {
    };
    return 0;
}