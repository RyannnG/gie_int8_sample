# gie_int8_sample

# Benchmark
TitanX (Pascal) val_dataset 49920 images [batch_size=64]

|                     |   Top1   |   Top5   |                 Time                   |
| --------------------|:--------:|:--------:| :-------------------------------------:|
| GoogLeNet(FP32)     | 0.676863 | 0.882933 | 0.395122 ms/image and 25.2878 ms/batch |
| GoogLeNet(INT8)     | 0.650641 | 0.869251 | 0.147287 ms/image and 9.42637 ms/batch |
| VGGNet(FP32)        | 0.654888 | 0.863882 | 2.41093 ms/image and 154.3 ms/batch    |
| VGGNet(INT8)        | 0.568329 | 0.851202 | 0.903336 ms/image and 57.8135 ms/batch |
| GoogLeNet_half(FP32)| 0.604667 | 0.832632 | 0.155812 ms/image and 9.97194 ms/batch |
| GoogLeNet_half(INT8)| 0.567728 | 0.809615 | 0.0690427 ms/image and 4.41873 ms/batch|

## FP32
### GoogLeNet:
```
Top1: 0.676863, Top5: 0.882933
Processing 49920 images averaged 0.395122 ms/image and 25.2878 ms/batch.
```
### VGGNet
```
Top1: 0.654888, Top5: 0.863882
Processing 49920 images averaged 2.41093 ms/image and 154.3 ms/batch.
```
### GoogLeNet_half:
```
Top1: 0.604667, Top5: 0.832632
Processing 49920 images averaged 0.155812 ms/image and 9.97194 ms/batch.
```


## INT8
### GoogLeNet:
```
Top1: 0.650641, Top5: 0.869251
Processing 49920 images averaged 0.147287 ms/image and 9.42637 ms/batch.
```
### VGGNet:
```
Top1: 0.568329, Top5: 0.851202
Processing 49920 images averaged 0.903336 ms/image and 57.8135 ms/batch.
```
### GoogLeNet_half:
```
Top1: 0.567728, Top5: 0.809615
Processing 49920 images averaged 0.0690427 ms/image and 4.41873 ms/batch.
```
