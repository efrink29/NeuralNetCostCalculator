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
vector<double> getSevenSergmentOutputForLabel(int label)
{

    //  1 1 1
    // 2     3
    // 2     3
    //  4 4 4
    // 5     6
    // 5     6
    //  7 7 7
    vector<double> output = {0, 0, 0, 0, 0, 0, 0};
    switch (label)
    {
    case 0:
        output = {1, 1, 1, 0, 1, 1, 1};
        break;
    case 1:
        output = {0, 0, 1, 0, 0, 1, 0};
        break;
    case 2:
        output = {1, 0, 1, 1, 1, 0, 1};
        break;
    case 3:
        output = {1, 0, 1, 1, 0, 1, 1};
        break;
    case 4:
        output = {0, 1, 1, 1, 0, 1, 0};
        break;
    case 5:
        output = {1, 1, 0, 1, 0, 1, 1};
        break;
    case 6:
        output = {1, 1, 0, 1, 1, 1, 1};
        break;
    case 7:
        output = {1, 0, 1, 0, 0, 1, 0};
        break;
    case 8:
        output = {1, 1, 1, 1, 1, 1, 1};
        break;
    case 9:
        output = {1, 1, 1, 1, 0, 1, 1};
        break;
    default:
        cout << "Error: Invalid label " << label << endl;
        break;
    }
    return output;
}

vector<double> getOutputForLabel(int label)
{
    vector<double> output = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    output[label] = 1;
    return getSevenSergmentOutputForLabel(label);
}

void trainMnist(MNISTReader *reader, NeuralNetwork *nn, int numBatches, int batchSize, double learningRate)
{
    vector<vector<vector<double>>> inputs;
    vector<vector<vector<double>>> outputs;
    for (int i = 0; i < numBatches; i++)
    {

        for (uint8_t n = 0; n < 10; n++)
        {
            vector<vector<double>> inputBatch = {};
            vector<vector<double>> outputBatch = {};
            for (int b = 0; b < batchSize; b++)
            {
                Image_Data *image1 = reader->getNextLabelTrainImage(n);

                vector<double> input = image1->pixels;
                if (input.size() != 784)
                {
                    cout << "Error: input size is " << input.size() << endl;
                    exit(1);
                    return;
                }
                vector<double> output = getOutputForLabel(image1->label);

                inputBatch.push_back(input);
                outputBatch.push_back(output);
                delete image1;
            }
            inputs.push_back(inputBatch);
            outputs.push_back(outputBatch);
        }
    }
    for (vector<vector<double>> inputBatch : inputs)
    {
        for (vector<double> input : inputBatch)
        {
            if (input.size() != 784)
            {
                cout << "Error: input size is " << input.size() << endl;
                exit(1);
                return;
            }
        }
    }
    nn->setLearningRate(learningRate);
    nn->train(inputs, outputs);
}

double testMnist(MNISTReader *reader, NeuralNetwork *nn, int numTests)
{
    vector<vector<double>> inputBatch = {};
    vector<vector<double>> outputBatch = {};

    for (int i = 0; i < numTests; i++)
    {
        for (uint8_t n = 0; n < 10; n++)
        {
            Image_Data *image1 = reader->getNextLabelTestImage(n);
            vector<double> input = image1->pixels;
            vector<double> output = getOutputForLabel(image1->label);
            inputBatch.push_back(input);
            outputBatch.push_back(output);
            delete image1;
        }
    }
    return nn->test(&inputBatch, &outputBatch, false);
}

vector<double> bigMnistTest(MNISTReader *reader, NeuralNetwork *nn, int numTests)
{

    // bool numbersPresent[] = {false, false, false, false, false, false, false, false, false, false};
    double avgPerDigitError[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int numPerDigitTests[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < numTests; i++)
    {
        for (uint8_t n = 0; n < 10; n++)
        {
            Image_Data *image1 = reader->getNextLabelTestImage(n);
            vector<double> input = image1->pixels;
            vector<double> output = getOutputForLabel(image1->label);
            vector<vector<double>> inputBatch = {input};
            vector<vector<double>> outputBatch = {output};
            double error = nn->test(&inputBatch, &outputBatch, false);
            avgPerDigitError[n] += error;
            numPerDigitTests[n]++;
            delete image1;
        }
    }
    vector<double> avgErrors;
    for (int i = 0; i < 10; i++)
    {
        double avgError = avgPerDigitError[i] / (double)numPerDigitTests[i];
        // cout << "Average error for " << i << ": " << avgError << endl;
        avgErrors.push_back(avgError);
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
    int modelCost = nn->getComputations();
    cout << "Model Cost: " << modelCost << endl;
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
            int numBatches = 5000 / (batchSize * 10);
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
            for (int j = 0; j < 10; j++)
            {
                vector<double> output = getOutputForLabel(j);
                vector<vector<double>> inputBatch = {};
                vector<vector<double>> outputBatch = {};

                Image_Data *image1 = reader->getNextLabelTestImage(j);
                vector<double> input = image1->pixels;
                inputBatch.push_back(input);
                outputBatch.push_back(output);
                delete image1;
                nn->test(&inputBatch, &outputBatch, true);
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
            nn->save("models/" + fileName + ".nn");
        }
    }

    for (int i = 0; i < numEpochs; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            totalErrorRates[i][j] /= numModels;
        }
    }

    ofstream errorFile("results/" + fileName + ".csv");
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

    vector<int> *topology1 = new vector<int>();
    topology1->push_back(28 * 28);
    topology1->push_back(10);
    topology1->push_back(100);
    topology1->push_back(7);

    NeuralNetwork *nn1 = new NeuralNetwork(topology1, 0.1);
    nn1->setLearningRate(0.1);
    nn1->randomizeWeightsAndBias();
    // cout << "Cost of model: " << nn1->getComputations() << endl;
    int numModels = 1;
    int numEpochs = 10;

    generateDeepErrorTable("topologyTest5", nn1, &reader, numModels, numEpochs, 1);

    delete nn1;

    return 0;
}