#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "opencv/cv.h"
#include "opencv/highgui.h"

using namespace std;
#define BATCHSIZE 64
int readyLabel(string labelfn, unordered_map<string, int> &labelmap);
int makeBatchs(string imglistfn, unordered_map<string, int> &labelmap, int dims[4], string tgtfolder);
int test();
void split(string &s, string &delim, vector<string> &ret);

int main()
{
    test();
    return 0;
}

int test()
{
    string labelfn = "../ILSVR2012.label.txt";
    // string imglistfn = "/mnt/disk_4T/imgnet/train.txt";
    string imglistfn = "/mnt/disk_4T/imgnet/val.txt";
    string tgtfolder = "/mnt/disk_4T/lbin_work/batches";

    unordered_map<string, int> labelmap;
    readyLabel(labelfn, labelmap);
    int dims[4] = { BATCHSIZE, 3, 224, 224 };
    makeBatchs(imglistfn, labelmap, dims, tgtfolder);
    return 0;
}

int readyLabel(string labelfn, unordered_map<string, int> &labelmap)
{
    string line;
    int label = 0;

    ifstream myfile(labelfn);
    if (!myfile.is_open()) return -1;
    labelmap.clear();
    while (getline(myfile, line))
    {
        labelmap.insert({ line, label++ });
    }
    myfile.close();

    return 0;
}

int makeBatchs(string imglistfn, unordered_map<string, int> &labelmap, int dims[4], string tgtfolder)
{
    string line;
    int label = 0;
    int bsz = dims[0], cn = dims[1], std_height = dims[2], std_width = dims[3];
    printf("nchw: %d, %d, %d, %d\n", dims[0], dims[1], dims[2], dims[3]);
    int totalsize     = bsz * cn * std_height * std_width + bsz;
    float *pfOneBatch = new float[totalsize];
    float *pfLabels   = pfOneBatch + bsz * cn * std_height * std_width;
    int imgsize       = std_height * std_width;

    ifstream myfile(imglistfn);
    if (!myfile.is_open()) return -1;

    vector<string> parts, parts2;
    string delim;
    cv::Mat stdimg(std_height, std_width, CV_8UC3);
    cv::Mat stdimgf(std_height, std_width, CV_32FC3);

    float *pfDataArea = pfOneBatch;
    int nowbi         = 0;
    int nowbnum       = 0;
    // string imgFolder  = "/mnt/disk_4T/imgnet/train/";
    string imgFolder = "/mnt/disk_4T/val/";
    while (getline(myfile, line))
    {
        //! TRAIN
        // delim = "/";
        // split(line, delim, parts);
        // string imgfn = *(parts.end() - 1);
        // delim        = "_";
        // split(imgfn, delim, parts);
        // string labelpre = parts[0];
        // delim           = " ";
        // split(line, delim, parts);
        // line = imgFolder + parts[0];

        //! VAL
        delim = " ";
        split(line, delim, parts);
        string imgfn    = imgFolder + parts[0];
        string labelpre = parts[1];

        // std::cout << imgfn << " " << labelpre << std::endl;
        cv::Mat img = cv::imread(imgfn);
        cv::resize(img, stdimg, cv::Size(std_width, std_height), 0, 0, CV_INTER_LINEAR);
        stdimg.convertTo(stdimgf, CV_32FC3, 1.0f, 0);
        float *pftmpdata = (float *)stdimgf.data;
        // cv::imshow("img", stdimg);
        // cv::waitKey(1);
        float *pfNow = pfDataArea + nowbi * cn * imgsize;
        for (int ri = 0; ri < std_height; ri++)
        {
            for (int ci = 0; ci < std_width; ci++)
            {
                int oft                  = ri * std_width + ci;
                pfNow[oft]               = pftmpdata[oft * 3 + 0] - 128;
                pfNow[oft + imgsize]     = pftmpdata[oft * 3 + 1] - 128;
                pfNow[oft + imgsize * 2] = pftmpdata[oft * 3 + 2] - 128;
            }
        }
        pfLabels[nowbi] = atoi(labelpre.c_str());
        nowbi++;
        if (nowbi % bsz == 0)
        {
            stringstream ss;
            // cout << imgfn << ", " << labelpre << ", " << labelmap[labelpre]
            // << endl;
            ss << tgtfolder << "/batch" << nowbnum;
            string dstbatchfn;
            ss >> dstbatchfn;
            cout << dstbatchfn << endl;
            ofstream mybin(dstbatchfn, ios::out | ios::binary);
            mybin.write((char *)dims, 4 * sizeof(int));
            // for (int ii = 0; ii < 10; ii++) printf("%.3f,", pfOneBatch[ii]);
            // printf("\n");
            mybin.write((char *)pfOneBatch, totalsize * sizeof(float));
            mybin.close();

            nowbi = 0;
            nowbnum++;
        }
    }
    myfile.close();

    delete[] pfOneBatch;

    return 0;
}

void split(string &s, string &delim, vector<string> &ret)
{
    size_t last  = 0;
    size_t index = s.find_first_of(delim, last);
    ret.clear();
    while (index != std::string::npos)
    {
        ret.push_back(s.substr(last, index - last));
        last  = index + 1;
        index = s.find_first_of(delim, last);
    }
    if (index - last > 0)
    {
        ret.push_back(s.substr(last, index - last));
    }
}
