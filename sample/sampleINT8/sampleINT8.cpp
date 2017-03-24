#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unordered_map>

#include "NvCaffeParser.h"
#include "NvInfer.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)                                                          \
    {                                                                          \
        if (status != 0)                                                       \
        {                                                                      \
            std::cout << "Cuda failure: " << status;                           \
            abort();                                                           \
        }                                                                      \
    }

// stuff we know about the network and the caffe input/output blobs

const char *INPUT_BLOB_NAME  = "data";
const char *OUTPUT_BLOB_NAME = "prob";
const char *gNetworkName;
// const int gBatchSize = 10;
// const int gStandardC = 3;
// const int gStandardH = 224;
// const int gStandardW = 224;

// Logger for GIE info/warning/errors
class Logger : public ILogger
{
    void log(Severity severity, const char *msg) override
    {
        if (severity != Severity::kINFO) std::cout << msg << std::endl;
    }
} gLogger;

std::string locateDataDir()
{
    std::string file = std::string("data/int8/") + gNetworkName;
    struct stat info;
    int i, MAX_DEPTH = 10;
    for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
        file = "../" + file;

    assert(i != MAX_DEPTH);

    return file;
}

std::string locateFile(const std::string &input)
{
    return locateDataDir() + "/" + input;
}

void caffeToGIEModel(const std::string &deployFile, // name for caffe prototxt
                     const std::string &modelFile,  // name for model
                     const std::vector<std::string> &outputs, // network outputs
                     unsigned int maxBatchSize, // batch size - NB must be at
                                                // least as large as the batch
                                                // we want to run with)
                     IInt8Calibrator *calibrator,
                     std::ostream &gieModelStream)

{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser        = createCaffeParser();
    std::cout << locateFile(deployFile).c_str() << std::endl;
    const IBlobNameToTensor *blobNameToTensor =
    parser->parse(locateFile(deployFile).c_str(), locateFile(modelFile).c_str(),
                  *network, DataType::kFLOAT);

    std::cout << "caffeToGIEModel 1!" << std::endl;
    // specify which tensors are outputs
    for (auto &s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 28);
    builder->setAverageFindIterations(1);
    builder->setMinFindIterations(1);
    builder->setDebugSync(true);
    std::cout << "caffeToGIEModel 2!" << std::endl;

    builder->setInt8Mode(calibrator != nullptr);
    builder->setInt8Calibrator(calibrator);

    std::cout << "caffeToGIEModel 3!" << std::endl;

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    std::cout << "caffeToGIEModel 4!" << std::endl;
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    engine->serialize(gieModelStream);
    engine->destroy();
    builder->destroy();
}

float doInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine
    // requires exactly IEngine::getNbBindings(), of these, but in this case we
    // know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];
    float ms{ 0.0f };

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. note that indices are guaranteed to be less than
    // IEngine::getNbBindings()
    int inputIndex  = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    Dims3 inputDims = context.getEngine().getBindingDimensions(
    context.getEngine().getBindingIndex(INPUT_BLOB_NAME));
    Dims3 outputDims = context.getEngine().getBindingDimensions(
    context.getEngine().getBindingIndex(OUTPUT_BLOB_NAME));

    size_t inputSize = batchSize * inputDims.c * inputDims.h * inputDims.w * sizeof(float),
           outputSize =
           batchSize * outputDims.c * outputDims.h * outputDims.w * sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

    CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    cudaEventRecord(start, stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    CHECK(cudaMemcpy(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaStreamDestroy(stream));
    return ms;
}

int calculateScore(float *batchProb, float *labels, int batchSize, int outputSize, int threshold)
{
    int success = 0;
    for (int i = 0; i < batchSize; i++)
    {
        float *prob = batchProb + outputSize * i, correct = prob[(int)labels[i]];
        printf("%d:%d, %f; ", i, (int)labels[i], correct);

        int better = 0;
        for (int j = 0; j < outputSize; j++)
            if (prob[j] >= correct) better++;
        if (better <= threshold) success++;
    }
    printf("\n");
    return success;
}

class BatchStream
{
    public:
    BatchStream(int batchSize, int maxBatches)
    : mBatchSize(batchSize), mMaxBatches(maxBatches)
    {
        std::string strfn = locateFile(std::string("batches/batch0"));
        std::cout << "BatchStream " + strfn + "\n";
        FILE *file = fopen(strfn.c_str(), "rb");
        fread(&mDims, sizeof(int), 4, file);
        fclose(file);
        std::cout << mDims.n << ", " << mDims.c << ", " << mDims.h << ", "
                  << mDims.w << std::endl;
        //		mDims.c = gStandardC;
        //		mDims.h = gStandardH;
        //		mDims.w = gStandardW;
        //		mDims.n = gBatchSize;
        mImageSize = mDims.c * mDims.h * mDims.w;
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.n * mImageSize, 0);
        mFileLabels.resize(mDims.n, 0);
        reset(0);
    }

    void reset(int firstBatch)
    {
        std::cout << "BatchStream reset\n";
        mBatchCount   = 0;
        mFileCount    = 0;
        mFileBatchPos = mDims.n;
        skip(firstBatch);
    }

    bool next()
    {
        //    std::cout << "BatchStream next\n";
        if (mBatchCount == mMaxBatches) return false;

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize;
             batchPos += csize, mFileBatchPos += csize)
        {
            //      std::cout << "csize:" << csize << ", batchPos:" << batchPos
            //      << ", mBatchSize:" << mBatchSize << ", mFileBatchPos:" <<
            //      mFileBatchPos << std::endl;
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.n);
            if (mFileBatchPos == mDims.n && !update()) return false;

            // copy the smaller of: elements left to fulfill the request, or
            // elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.n - mFileBatchPos);
            std::copy_n(getFileBatch() + mFileBatchPos * mImageSize,
                        csize * mImageSize, getBatch() + batchPos * mImageSize);
            std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
        }
        mBatchCount++;
        return true;
    }

    void skip(int skipCount)
    {
        std::cout << "BatchStream skip\n";
        if (mBatchSize >= mDims.n && mBatchSize % mDims.n == 0 &&
            mFileBatchPos == mDims.n)
        {
            mFileCount += skipCount * mBatchSize / mDims.n;
            return;
        }

        int x = mBatchCount;
        for (int i  = 0; i < skipCount; i++) next();
        mBatchCount = x;
    }

    float *getBatch() { return &mBatch[0]; }
    float *getLabels() { return &mLabels[0]; }
    int getBatchesRead() const { return mBatchCount; }
    int getBatchSize() const { return mBatchSize; }
    Dims4 getDims() const { return mDims; }

    private:
    float *getFileBatch() { return &mFileBatch[0]; }
    float *getFileLabels() { return &mFileLabels[0]; }

    bool update()
    {
        // batch format: n, c, h, w, data0(n*c*h*w), label0(n), data1, label1,
        // ...
        //    std::cout << "BatchStream update\n";
        std::string inputFileName =
        locateFile(std::string("batches/batch") + std::to_string(mFileCount++));
        //    std::cout << "BatchStream update:" + inputFileName + "\n";
        FILE *file = fopen(inputFileName.c_str(), "rb");
        if (!file) return false;

        Dims4 dims;
        fread(&dims, sizeof(int), 4, file);
        assert(dims.n == mDims.n && dims.c == mDims.c && dims.h == mDims.h &&
               dims.w == mDims.w);

        size_t readInputCount =
        fread(getFileBatch(), sizeof(float), dims.n * mImageSize, file);
        size_t readLabelCount = fread(getFileLabels(), sizeof(float), dims.n, file);
        ;
        float *pfdata = getFileBatch();
        float *pflbl  = getFileLabels();
        for (int i = 0; i < dims.n; i++)
            printf("%.3f,%.1f;", pfdata[i], pflbl[i]);
        printf("\n");
        assert(readInputCount == size_t(dims.n * mImageSize) &&
               readLabelCount == size_t(dims.n));

        fclose(file);
        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{ 0 };
    int mMaxBatches{ 0 };
    int mBatchCount{ 0 };

    int mFileCount{ 0 }, mFileBatchPos{ 0 };
    int mImageSize{ 0 };

    Dims4 mDims;
    std::vector<float> mBatch;
    std::vector<float> mLabels;
    std::vector<float> mFileBatch;
    std::vector<float> mFileLabels;
};

class Int8Calibrator : public IInt8Calibrator
{
    public:
    Int8Calibrator(BatchStream &stream, int firstBatch, double cutoff, double quantile, bool readCache = true)
    : mStream(stream), mFirstBatch(firstBatch), mReadCache(readCache)
    {
        Dims4 dims  = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.c * dims.h * dims.w;
        std::cout << dims.c << "  " << dims.h << "  " << dims.w << "  "
                  << dims.n << std::endl;
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        reset(cutoff, quantile);
    }

    ~Int8Calibrator() { CHECK(cudaFree(mDeviceInput)); }

    int getBatchSize() const override { return mStream.getBatchSize(); }
    double getQuantile() const override { return mQuantile; }
    double getRegressionCutoff() const override { return mCutoff; }

    bool getBatch(void *bindings[], const char *names[], int nbBindings) override
    {
        std::cout << "Int8Calibrator getBatch\n";
        if (!mStream.next()) return false;

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(),
                         mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], INPUT_BLOB_NAME));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void *readCalibrationCache(size_t &length) override
    {
        std::cout << "Int8Calibrator readCalibrationCache\n";
        mCalibrationCache.clear();
        std::ifstream input(locateFile("CalibrationTable"), std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(mCalibrationCache));

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void *cache, size_t length) override
    {
        std::cout << "Int8Calibrator writeCalibrationCache\n";
        std::ofstream output(locateFile("CalibrationTable"), std::ios::binary);
        output.write(reinterpret_cast<const char *>(cache), length);
    }

    const void *readHistogramCache(size_t &length) override
    {
        std::cout << "Int8Calibrator readHistogramCache\n";
        length = mHistogramCache.size();
        return length ? &mHistogramCache[0] : nullptr;
    }

    void writeHistogramCache(const void *cache, size_t length) override
    {
        std::cout << "Int8Calibrator writeHistogramCache\n";
        mHistogramCache.clear();
        std::copy_n(reinterpret_cast<const char *>(cache), length,
                    std::back_inserter(mHistogramCache));
    }

    void reset(float cutoff, float quantile)
    {
        std::cout << "Int8Calibrator reset\n";
        mCutoff   = cutoff;
        mQuantile = quantile;
        mStream.reset(mFirstBatch);
    }

    private:
    BatchStream mStream;
    int mFirstBatch;
    double mCutoff, mQuantile;
    bool mReadCache{ true };

    size_t mInputCount;
    void *mDeviceInput{ nullptr };
    std::vector<char> mCalibrationCache, mHistogramCache;
};

std::pair<float, float> scoreModel(int batchSize,
                                   int firstBatch,
                                   int nbScoreBatches,
                                   Int8Calibrator *calibrator,
                                   bool quiet = false)
{
    std::stringstream gieModelStream;
    std::cout << gNetworkName << std::endl;
    std::string strNetName(gNetworkName);
    // std::cout << "scoreModel 1!" << std::endl;
    caffeToGIEModel("deploy.prototxt", strNetName + ".caffemodel",
                    std::vector<std::string>{ OUTPUT_BLOB_NAME }, batchSize,
                    calibrator, gieModelStream);
    // std::cout << "scoreModel 2!" << std::endl;
    // Create engine and deserialize model.
    IRuntime *infer = createInferRuntime(gLogger);
    gieModelStream.seekg(0, gieModelStream.beg);
    ICudaEngine *engine        = infer->deserializeCudaEngine(gieModelStream);
    IExecutionContext *context = engine->createExecutionContext();

    BatchStream stream(batchSize, nbScoreBatches);
    stream.skip(firstBatch);

    Dims3 outputDims = context->getEngine().getBindingDimensions(
    context->getEngine().getBindingIndex(OUTPUT_BLOB_NAME));
    int outputSize = outputDims.c * outputDims.h * outputDims.w;
    int top1{ 0 }, top5{ 0 };
    float totalTime{ 0.0f };
    std::vector<float> prob(batchSize * outputSize, 0);

    while (stream.next())
    {
        totalTime += doInference(*context, stream.getBatch(), &prob[0], batchSize);

        top1 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 1);
        top5 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 5);

        std::cout << (!quiet && stream.getBatchesRead() % 10 == 0 ? "." : "")
                  << (!quiet && stream.getBatchesRead() % 800 == 0 ? "\n" : "")
                  << std::flush;
    }
    int imagesRead = stream.getBatchesRead() * batchSize;
    float t1 = float(top1) / float(imagesRead), t5 = float(top5) / float(imagesRead);

    if (!quiet)
    {
        std::cout << "\nTop1: " << t1 << ", Top5: " << t5 << std::endl;
        std::cout << "Processing " << imagesRead << " images averaged "
                  << totalTime / imagesRead << " ms/image and "
                  << totalTime / stream.getBatchesRead() << " ms/batch." << std::endl;
    }

    context->destroy();
    engine->destroy();
    infer->destroy();
    return std::make_pair(t1, t5);
}

struct CalibrationParameters
{
    const char *networkName;
    double cutoff;
    double quantileIndex;
};

CalibrationParameters gCalibrationTable[] = { { "alexnet", 0.6, 7.0 },
                                              { "vgg19", 0.5, 5 },
                                              { "googlenet", 1, 8.0 },
                                              { "resnet-50", 0.61, 2.0 },
                                              { "resnet-101", 0.51, 2.5 },
                                              { "resnet-152", 0.4, 5.0 } };

static const int gCalibrationTableSize =
sizeof(gCalibrationTable) / sizeof(CalibrationParameters);

static const int CAL_BATCH_SIZE        = 32;
static const int FIRST_CAL_BATCH       = 0,
                 NB_CAL_BATCHES        = 10; // calibrate over images 0-500
static const int FIRST_CAL_SCORE_BATCH = 100,
                 NB_CAL_SCORE_BATCHES  = 100; // score over images 5000-10000

double quantileFromIndex(double quantileIndex)
{
    return 1 - pow(10, -quantileIndex);
}

void searchCalibrations(double firstCutoff,
                        double cutoffIncrement,
                        int nbCutoffs,
                        double firstQuantileIndex,
                        double quantileIndexIncrement,
                        int nbQuantiles,
                        float &bestScore,
                        double &bestCutoff,
                        double &bestQuantileIndex,
                        Int8Calibrator &calibrator)
{
    for (int i = 0; i < nbCutoffs; i++)
    {
        for (int j = 0; j < nbQuantiles; j++)
        {
            double cutoff = firstCutoff + double(i) * cutoffIncrement,
                   quantileIndex = firstQuantileIndex + double(j) * quantileIndexIncrement;
            calibrator.reset(cutoff, quantileFromIndex(quantileIndex));
            float score = scoreModel(CAL_BATCH_SIZE, FIRST_CAL_SCORE_BATCH,
                                     NB_CAL_SCORE_BATCHES, &calibrator, true)
                          .first; // score the model in quiet mode

            std::cout << "Score: " << score << " (cutoff = " << cutoff
                      << ", quantileIndex = " << quantileIndex << ")" << std::endl;
            if (score > bestScore)
                bestScore = score, bestCutoff = cutoff, bestQuantileIndex = quantileIndex;
        }
    }
}

void searchCalibrations(double &bestCutoff, double &bestQuantileIndex)
{
    float bestScore   = std::numeric_limits<float>::lowest();
    bestCutoff        = 0;
    bestQuantileIndex = 0;

    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
    Int8Calibrator calibrator(calibrationStream, 0, quantileFromIndex(0), false); // force calibration by ignoring region cache

    searchCalibrations(1, 0, 1, 2, 1, 7,
                       bestScore, bestCutoff, bestQuantileIndex, calibrator); // search the space with cutoff = 1 (i.e. max'ing over the histogram)
    searchCalibrations(0.4, 0.05, 7, 2, 1, 7,
                       bestScore, bestCutoff, bestQuantileIndex, calibrator); // search the space with cutoff = 0.4 to 0.7 (inclusive)

    // narrow in: if our best score is at cutoff 1 then search over quantiles,
    // else over both dimensions
    if (bestScore == 1)
        searchCalibrations(1, 0, 1, bestQuantileIndex - 0.5, 0.1, 11, bestScore,
                           bestCutoff, bestQuantileIndex, calibrator);
    else
        searchCalibrations(bestCutoff - 0.04, 0.01, 9, bestQuantileIndex - 0.5, 0.1,
                           11, bestScore, bestCutoff, bestQuantileIndex, calibrator);
    std::cout << "\n\nBest score: " << bestScore << " (cutoff = " << bestCutoff
              << ", quantileIndex = " << bestQuantileIndex << ")" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Please provide the network as the first argument." << std::endl;
        exit(0);
    }
    gNetworkName = argv[1];

    int batchSize = 32, firstScoreBatch = 10,
        nbScoreBatches = 40; // by default we score over 40K images starting at
                             // 10000, so we don't score those used to search
                             // calibration
    bool search = false;

    for (int i = 2; i < argc; i++)
    {
        if (!strncmp(argv[i], "batch=", 6))
            batchSize = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "start=", 6))
            firstScoreBatch = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "score=", 6))
            nbScoreBatches = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "search", 6))
            search = true;
        else
        {
            std::cout << "Unrecognized argument " << argv[i] << std::endl;
            exit(0);
        }
    }

    if (batchSize > 128)
    {
        std::cout << "Please provide batch size <= 128" << std::endl;
        exit(0);
    }

    if ((firstScoreBatch + nbScoreBatches) * batchSize > 500000)
    {
        std::cout << "Only 50000 images available" << std::endl;
        exit(0);
    }

    std::cout.precision(6);

    {
        double cutoff = 1, quantileIndex = 6;
        if (search)
            searchCalibrations(cutoff, quantileIndex);
        else
        {
            for (int i = 0; i < gCalibrationTableSize; i++)
            {
                if (!strcmp(gCalibrationTable[i].networkName, gNetworkName))
                    cutoff        = gCalibrationTable[i].cutoff,
                    quantileIndex = gCalibrationTable[i].quantileIndex;
            }
        }

        std::cout << "\nINT8 run:" << nbScoreBatches << " batches of size " << batchSize
                  << " starting at " << firstScoreBatch << " with cutoff " << cutoff
                  << " and quantile index " << quantileIndex << std::endl;
        BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
        Int8Calibrator calibrator(calibrationStream, FIRST_CAL_BATCH, cutoff,
                                  quantileFromIndex(quantileIndex));
        std::cout << "after calibrator!---------------------" << std::endl;
        scoreModel(batchSize, firstScoreBatch, nbScoreBatches, &calibrator);
    }
    if (1)
    {
        std::cout << "\nFP32 run:" << nbScoreBatches << " batches of size "
                  << batchSize << " starting at " << firstScoreBatch << std::endl;
        scoreModel(batchSize, firstScoreBatch, nbScoreBatches, nullptr);
    }

    shutdownProtobufLibrary();
    return 0;
}
