#include "MNISTReader.h"

MNISTReader::MNISTReader(const string &dataDirectory)
{
    srand(time(NULL));
    string trainImageFile = dataDirectory + "\\train-images.idx3-ubyte";
    string trainLabelFile = dataDirectory + "\\train-labels.idx1-ubyte";
    string testImageFile = dataDirectory + "/t10k-images.idx3-ubyte";
    string testLabelFile = dataDirectory + "/t10k-labels.idx1-ubyte";

    rows = 28;
    cols = 28;
    numTrainImages = 60000;
    numTestImages = 10000;

    train_labels = loadLabels(trainLabelFile);
    train_images = loadImages(trainImageFile, numTrainImages, rows, cols);
    test_labels = loadLabels(testLabelFile);
    test_images = loadImages(testImageFile, numTestImages, rows, cols);
    rng.seed(time(NULL)); // seed once
    buildLabelIndices();
}

void MNISTReader::buildLabelIndices()
{
    for (int i = 0; i < static_cast<int>(train_labels.size()); ++i)
    {
        train_label_indices[train_labels[i]].push_back(i);
    }

    for (int i = 0; i < static_cast<int>(test_labels.size()); ++i)
    {
        test_label_indices[test_labels[i]].push_back(i);
    }

    // Shuffle each label group
    for (auto it = train_label_indices.begin(); it != train_label_indices.end(); ++it)
    {
        std::shuffle(it->second.begin(), it->second.end(), rng);
    }

    for (auto it = test_label_indices.begin(); it != test_label_indices.end(); ++it)
    {
        std::shuffle(it->second.begin(), it->second.end(), rng);
    }

    // Initialize label pointers
    for (uint8_t i = 0; i < 10; ++i)
    {
        train_label_pointer[i] = 0;
        test_label_pointer[i] = 0;
    }
}

void MNISTReader::printImage(Image_Data *image, bool printPixels)
{
    cout << "Label: " << static_cast<int>(image->label) << endl;
    if (printPixels)
    {
        printImageInternal(image, rows, cols);
    }
}
Image_Data *MNISTReader::getNextTrainImage()
{
    vector<double> image_pixels;
    for (uint32_t i = 0; i < rows; i++)
    {
        for (uint32_t j = 0; j < cols; j++)
        {
            double pixel = (double)train_images[train_index][i * cols + j] / 255.0;
            image_pixels.push_back(pixel);
        }
    }
    Image_Data *image = new Image_Data();
    image->pixels = image_pixels;
    image->label = train_labels[train_index];
    train_index++;
    if (train_index >= numTrainImages)
    {
        train_index = 0;
        shuffleTrainImages();
    }
    return image;
}

Image_Data *MNISTReader::getNextTestImage()
{
    vector<double> image_pixels;
    for (uint32_t i = 0; i < rows; i++)
    {
        for (uint32_t j = 0; j < cols; j++)
        {
            image_pixels.push_back(test_images[test_index][i * cols + j] / 255.0);
        }
    }
    Image_Data *image = new Image_Data();
    image->pixels = image_pixels;
    image->label = test_labels[test_index];
    test_index++;
    if (test_index >= numTestImages)
    {
        test_index = 0;
        shuffleTestImages();
    }
    return image;
}
Image_Data *MNISTReader::getNextLabelTrainImage(uint8_t label)
{
    auto &indices = train_label_indices[label];
    int &pointer = train_label_pointer[label];

    if (indices.empty())
        return nullptr;

    if (pointer >= indices.size())
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        pointer = 0;
    }

    int real_index = indices[pointer++];

    vector<double> image_pixels;
    image_pixels.reserve(rows * cols);
    for (int i = 0; i < rows * cols; ++i)
    {
        image_pixels.push_back(train_images[real_index][i] / 255.0);
    }

    Image_Data *image = new Image_Data();
    image->pixels = image_pixels;
    image->label = train_labels[real_index];
    return image;
}

Image_Data *MNISTReader::getNextLabelTestImage(uint8_t label)
{
    auto &indices = test_label_indices[label];
    int &pointer = test_label_pointer[label];

    if (indices.empty())
        return nullptr;

    if (pointer >= indices.size())
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        pointer = 0;
    }

    int real_index = indices[pointer++];

    vector<double> image_pixels;
    image_pixels.reserve(rows * cols);
    for (int i = 0; i < rows * cols; ++i)
    {
        image_pixels.push_back(test_images[real_index][i] / 255.0);
    }

    Image_Data *image = new Image_Data();
    image->pixels = image_pixels;
    image->label = test_labels[real_index];
    return image;
}

void MNISTReader::printImageInternal(Image_Data *image, uint32_t rows, uint32_t cols)
{
    for (int i = 0; i < (28 * 2) + 2; i++)
    {
        cout << "-";
    }
    cout << endl;
    for (uint32_t i = 0; i < rows; i++)
    {
        cout << "|";
        for (uint32_t j = 0; j < cols; j++)
        {
            if (image->pixels[i * cols + j] > 0.88)
            {
                cout << "##";
            }
            else if (image->pixels[i * cols + j] > 0.66)
            {
                cout << "++";
            }
            else if (image->pixels[i * cols + j] > 0.33)
            {
                cout << "--";
            }
            else if (image->pixels[i * cols + j] > 0.11)
            {
                cout << "..";
            }
            else
            {
                cout << "  ";
            }
        }
        cout << "|" << endl;
    }
    for (int i = 0; i < (28 * 2) + 2; i++)
    {
        cout << "-";
    }
    cout << endl;
}

uint32_t MNISTReader::readBigEndianInt(ifstream &file)
{
    uint32_t value = 0;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    return (value >> 24) |
           ((value >> 8) & 0x0000FF00) |
           ((value << 8) & 0x00FF0000) |
           (value << 24);
}

vector<uint8_t> MNISTReader::loadLabels(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening label file: " << filename << endl;
        exit(1);
    }

    uint32_t magic = readBigEndianInt(file);
    uint32_t numLabels = readBigEndianInt(file);

    if (magic != 2049)
    {
        cerr << "Invalid magic number for label file: " << magic << endl;
        exit(1);
    }

    vector<uint8_t> labels(numLabels);
    file.read(reinterpret_cast<char *>(labels.data()), numLabels);

    file.close();
    return labels;
}

vector<vector<uint8_t>> MNISTReader::loadImages(const string &filename, uint32_t &numImages, uint32_t &rows, uint32_t &cols)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening image file: " << filename << endl;
        exit(1);
    }

    uint32_t magic = readBigEndianInt(file);
    numImages = readBigEndianInt(file);
    rows = readBigEndianInt(file);
    cols = readBigEndianInt(file);

    if (magic != 2051)
    {
        cerr << "Invalid magic number for image file: " << magic << endl;
        exit(1);
    }

    vector<vector<uint8_t>> images(numImages, vector<uint8_t>(rows * cols));
    for (uint32_t i = 0; i < numImages; i++)
    {
        file.read(reinterpret_cast<char *>(images[i].data()), rows * cols);
    }

    file.close();
    return images;
}

void MNISTReader::shuffleTestImages()
{
    cout << "Shuffling Test Set ..." << endl;
    for (int i = 0; i < numTestImages; i++)
    {
        int index = rand() % test_images.size();
        uint8_t temp_label = test_labels[index];
        vector<uint8_t> temp_image = test_images[index];
        test_labels[index] = test_labels[i];
        test_images[index] = test_images[i];
        test_labels[i] = temp_label;
        test_images[i] = temp_image;
        if ((i % (numTestImages / 10)) == 0)
        {
            // cout << "Test images left: " << (numTestImages - i) << endl;
        }
    }
    test_index = 0;
    // cout << " Done ... ";
    //  cout << "Test images: " << test_images.size() << endl;
}

void MNISTReader::shuffleTrainImages()
{

    cout << "Shuffling Training Set ..." << endl;
    for (int i = 0; i < numTrainImages; i++)
    {
        int index = rand() % train_images.size();
        uint8_t temp_label = train_labels[index];
        vector<uint8_t> temp_image = train_images[index];
        train_labels[index] = train_labels[i];
        train_images[index] = train_images[i];
        train_labels[i] = temp_label;
        train_images[i] = temp_image;
        if ((i % (numTrainImages / 10)) == 0)
        {
            // cout << "Train images left: " << (numTrainImages - i) << endl;
        }
    }
    train_index = 0;

    // cout << "Done ... ";
}
