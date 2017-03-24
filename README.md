# gie_int8_sample

# Benchmark
TitanX (Pascal) val_dataset 49920 images [batch_size=64]

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
