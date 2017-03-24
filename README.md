# gie_int8_sample

# Benchmark
TitanX (Pascal) val_dataset 44800 images [batch_size=64]

## FP32
### GoogLeNet:
```
Top1: 0.676406, Top5: 0.8825
Processing 44800 images averaged 0.678648 ms/image and 43.4334 ms/batch.
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
Top1: 0.649732, Top5: 0.86875
Processing 44800 images averaged 0.243285 ms/image and 15.5702 ms/batch.
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
