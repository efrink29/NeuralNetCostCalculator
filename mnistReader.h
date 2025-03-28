#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <ctime>

using namespace std;

struct Image_Data
{
    vector<double> pixels;
    uint8_t label;
};

class MNISTReader
{
public:
    MNISTReader(const string &dataDirectory);

    void printImage(Image_Data *image, bool printPixels = true);

    Image_Data *getNextTrainImage();
    Image_Data *getNextTestImage();

    Image_Data *getNextLabelTrainImage(uint8_t label);
    Image_Data *getNextLabelTestImage(uint8_t label);

private:
    int train_index = 0;
    int test_index = 0;
    vector<uint8_t> train_labels;
    vector<vector<uint8_t>> train_images;
    vector<uint8_t> test_labels;
    vector<vector<uint8_t>> test_images;

    std::unordered_map<uint8_t, std::vector<int>> train_label_indices;
    std::unordered_map<uint8_t, std::vector<int>> test_label_indices;
    std::unordered_map<uint8_t, int> train_label_pointer;
    std::unordered_map<uint8_t, int> test_label_pointer;

    std::default_random_engine rng;

    // Add private method to build index maps
    void buildLabelIndices();

    uint32_t numTrainImages, numTestImages, rows, cols;

    void printImageInternal(Image_Data *image, uint32_t rows, uint32_t cols);
    uint32_t readBigEndianInt(ifstream &file);
    vector<uint8_t> loadLabels(const string &filename);
    vector<vector<uint8_t>> loadImages(const string &filename, uint32_t &numImages, uint32_t &rows, uint32_t &cols);
    void shuffleTrainImages();
    void shuffleTestImages();
};

#endif // MNIST_READER_H
