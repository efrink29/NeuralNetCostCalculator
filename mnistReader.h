#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

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

private:
    int train_index = 0;
    int test_index = 0;
    vector<uint8_t> train_labels;
    vector<vector<uint8_t>> train_images;
    vector<uint8_t> test_labels;
    vector<vector<uint8_t>> test_images;

    uint32_t numTrainImages, numTestImages, rows, cols;

    void printImageInternal(Image_Data *image, uint32_t rows, uint32_t cols);
    uint32_t readBigEndianInt(ifstream &file);
    vector<uint8_t> loadLabels(const string &filename);
    vector<vector<uint8_t>> loadImages(const string &filename, uint32_t &numImages, uint32_t &rows, uint32_t &cols);
    void shuffleTrainImages();
    void shuffleTestImages();
};

#endif // MNIST_READER_H
