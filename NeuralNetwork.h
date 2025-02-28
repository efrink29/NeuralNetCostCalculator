#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cstdlib>
#include <vector>

class NeuralNetwork
{
public:
    NeuralNetwork(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    double getRecentAverageError() const { return m_recentAverageError; }

private:
    struct Neuron;
    typedef std::vector<Neuron> Layer;

    static double eta;   // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight() { return rand() / double(RAND_MAX); }
    static double randomWeight(double min, double max) { return min + (max - min) * rand() / double(RAND_MAX); }

    void feedForward(const Layer &prevLayer, Layer &nextLayer);
    void backProp(const Layer &prevLayer, Layer &nextLayer, const std::vector<double> &targetVals);
    void calcOutputGradients(const std::vector<double> &targetVals);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateWeights(Layer &prevLayer, Layer &nextLayer);

    std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

#endif // NEURALNETWORK_H