#include "NeuralNetwork.h"
#include "mnistReader.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

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
    nn->setLearningRate(learningRate);
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
    return nn->test(&inputBatch, &outputBatch, false);
}

vector<double> bigMnistTest(MNISTReader *reader, NeuralNetwork *nn, int numTests)
{

    // bool numbersPresent[] = {false, false, false, false, false, false, false, false, false, false};
    double avgPerDigitError[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int numPerDigitTests[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < numTests; i++)
    {
        Image_Data *image = reader->getNextTestImage();
        vector<double> input = image->pixels;
        vector<double> output = getOutputForLabel(image->label);
        vector<vector<double>> inputBatch = {input};
        vector<vector<double>> outputBatch = {output};
        double error = nn->test(&inputBatch, &outputBatch, false);
        avgPerDigitError[image->label] += error;
        numPerDigitTests[image->label]++;

        delete image;
    }
    vector<double> avgErrors;
    for (int i = 0; i < 10; i++)
    {
        avgErrors.push_back(avgPerDigitError[i] / (double)numPerDigitTests[i]);
    }
    return avgErrors;
}

bool saveNetwork(NeuralNetwork *nn)
{
    cout << "Enter filename to save to: ";
    string filename;
    cin >> filename;
    string dirfilename = "models/" + filename;

    // Check if file exists
    ifstream file(dirfilename);
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

    nn->save(dirfilename);
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

void generateDeepErrorTable(string fileName, NeuralNetwork *nn, MNISTReader *reader, int numModels, int numEpochs, int batchSize)
{
    double totalErrorRates[numEpochs][11];

    double minError[10];
    for (int i = 0; i < 10; i++)
    {
        minError[i] = 1;
    }
    for (int i = 0; i < numEpochs; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            totalErrorRates[i][j] = 0;
        }
    }

    for (int m = 0; m < numModels; m++)
    {
        cout << "Model " << (m + 1) << endl;
        if (numModels > 1)
        {
            nn->randomizeWeightsAndBias();
        }
        // nn->randomizeWeightsAndBias();
        cout << "Completed Epochs: ... " << endl;
        double errorRates[numEpochs][11];

        for (int i = 0; i < numEpochs; i++)
        {
            for (int j = 0; j < 11; j++)
            {
                errorRates[i][j] = 0;
            }
            auto start = chrono::high_resolution_clock::now();
            int numBatches = 10000 / batchSize;
            trainMnist(reader, nn, numBatches, batchSize, 0.1);
            vector<double> avgErrors = bigMnistTest(reader, nn, 1000);
            double avgError = 0;
            for (int j = 0; j < 10; j++)
            {
                errorRates[i][j] += avgErrors[j];
                avgError += avgErrors[j];
            }

            avgError /= 10;
            errorRates[i][10] += avgError;
            // cout << "Average error: " << avgError << endl;
            cout << "" << (i + 1);
            if (i < numEpochs - 1)
            {
                // cout << " ... ";
            }
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(end - start);
            auto time = duration.count();
            auto ETA = ((numEpochs - i - 1) + (numEpochs * (numModels - m - 1))) * time;
            if (ETA > 60)
            {
                cout << " ETA: " << ETA / 60 << " minutes" << endl;
            }
            else
            {
                cout << " ETA: " << ETA << " seconds" << endl;
            }
            // averageErrors.push_back(avgError);
            for (int e = 0; e < 11; e++)
            {
                totalErrorRates[i][e] += errorRates[i][e];
            }
        }
        cout << endl;
        double totalDiff = 0;
        for (int i = 0; i < 11; i++)
        {
            totalDiff += errorRates[numEpochs - 1][i] - minError[i];
        }
        if (totalDiff < 0)
        {
            for (int i = 0; i < 11; i++)
            {
                minError[i] = errorRates[numEpochs - 1][i];
            }
            nn->save(fileName + ".nn");
        }
    }

    for (int i = 0; i < numEpochs; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            totalErrorRates[i][j] /= numModels;
        }
    }

    ofstream errorFile(fileName + ".csv");
    errorFile << "Epoch,0,1,2,3,4,5,6,7,8,9,Average" << endl;
    for (int i = 0; i < numEpochs; i++)
    {
        errorFile << (i + 1) << ",";
        for (int j = 0; j < 10; j++)
        {
            if (totalErrorRates[i][j] < 0.0001)
            {
                errorFile << "0,";
            }
            else
            {
                errorFile << totalErrorRates[i][j] << ",";
            }
        }
        if (totalErrorRates[i][10] < 0.0001)
        {
            errorFile << "0";
        }
        else
        {
            errorFile << totalErrorRates[i][10];
        }

        errorFile << endl;
    }

    errorFile.close();
}

int main(int argc, char **argv)
{

    srand(time(NULL));
    MNISTReader reader("data");

    Image_Data *image1 = reader.getNextTrainImage();
    // Image_Data *image2 = reader.getNextTrainImage();

    cout << "Image 1: " << image1->pixels[(24 * 28) + 24] << endl;
    reader.printImage(image1, true);
    // reader.printImage(image2, true);

    delete image1;
    // delete image2;
    int modify = 28;
    vector<int> *topology = new vector<int>();
    topology->push_back(28 * 28);
    // topology->push_back((28 * 14));
    topology->push_back(10);

    NeuralNetwork *nn = new NeuralNetwork(topology, 0.1);
    nn->setLearningRate(0.1);
    nn->randomizeWeightsAndBias();

    int numModels = 10;
    int numEpochs = 10;

    generateDeepErrorTable("quickTest1", nn, &reader, numModels, numEpochs, 1);
    generateDeepErrorTable("quickTest2", nn, &reader, numModels, numEpochs, 2);
    // int connectionsRemoved = nn->pruneNetwork(0.1);
    // cout << "Connections removed: " << connectionsRemoved << endl;
    //  generateDeepErrorTable("postPrune", nn, &reader, numModels, numEpochs);
    delete nn;
    /*delete nn;
    topology->clear();
    topology->push_back(28 * 28);
    topology->push_back((28 * 4) / modify);
    topology->push_back((28 * 4) / modify);
    topology->push_back((28 * 4) / modify);
    topology->push_back((28 * 4) / modify);
    topology->push_back(10);
    nn = new NeuralNetwork(topology, 0.1);
    nn->setLearningRate(0.1);
    nn->randomizeWeightsAndBias();
    generateDeepErrorTable("deep4short", nn, &reader, numModels, numEpochs);
    delete nn;
    topology->clear();
    topology->push_back(28 * 28);
    for (int i = 0; i < 20; i++)
    {
        topology->push_back((28 * 4) / modify);
    }
    topology->push_back(10);
    nn = new NeuralNetwork(topology, 0.1);
    nn->setLearningRate(0.1);
    nn->randomizeWeightsAndBias();
    generateDeepErrorTable("deep4long", nn, &reader, numModels, numEpochs);
    delete nn;
    topology->clear();
    topology->push_back(28 * 28);
    topology->push_back((28 * 8) / modify);
    topology->push_back((28 * 8) / modify);
    topology->push_back(10);
    nn = new NeuralNetwork(topology, 0.1);
    nn->setLearningRate(0.1);
    nn->randomizeWeightsAndBias();
    generateDeepErrorTable("deep8short", nn, &reader, numModels, numEpochs);
    delete nn;
    topology->clear();
    topology->push_back(28 * 28);
    for (int i = 0; i < 4; i++)
    {
        topology->push_back((28 * 8) / modify);
    }
    topology->push_back(10);
    nn = new NeuralNetwork(topology, 0.1);
    nn->setLearningRate(0.1);
    nn->randomizeWeightsAndBias();
    generateDeepErrorTable("deep8long", nn, &reader, numModels, numEpochs); //*/
    delete nn;
    delete topology;

    return 0;
}