# gie_int8_sample

# Benchmark
TitanX (Pascal) val_dataset 44800 images [batch_size=64]

## FP32
### GoogLeNet:
```
Top1: 0.676406, Top5: 0.8825
Processing 44800 images averaged 0.678648 ms/image and 43.4334 ms/batch.
```
### VGGNET
```
Top1: 0.654018, Top5: 0.863237
Processing 44800 images averaged 3.97572 ms/image and 254.446 ms/batch
```

## INT8
### GoogLeNet:
```
Top1: 0.649732, Top5: 0.86875
Processing 44800 images averaged 0.243285 ms/image and 15.5702 ms/batch.
```
### VGGNet:
```
Top1: 0.56683, Top5: 0.850424
Processing 44800 images averaged 1.23129 ms/image and 78.8029 ms/batch
```
