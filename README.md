# AGMC
AGMC: Auto Graph encoder-decoder for Model Compression and Network Acceleration

## Dependencies

Current code base is tested under following environment:

1. Python   3.7.9 
2. PyTorch  1.6.0
3. torchvision 0.7.0
4. torch-geometric 1.6.1
5. NumPy 1.19.2 

## Training AGMC

### CIFAR-10
  To search the strategy on ResNet-56 with channel pruning, and prunes 50% FLOPs reduction, by running:   
  ```
python agmc_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.5 --pruning_method cp --train_episode 300 --train_size 5000 --val_size 1000 --output ./logs
   ```
  To search the strategy on ResNet-20 with channel pruning, and prunes 50% FLOPs reduction, by running:   
```
python agmc_network_pruning.py --dataset cifar10 --model resnet20 --compression_ratio 0.5 --pruning_method cp --train_episode 300 --train_size 5000 --val_size 1000 --output ./logs
```
   To search the strategy on ResNet-56 with fine-grained pruning,by running:
   ```
python agmc_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.5 --pruning_method fg --train_episode 50 --train_size 5000 --val_size 1000 --output ./logs
   ```


### ILSVRC-2012
To evaluate the AGMC on the ILSVRC-2012 dataset, you need to first download the dataset from [ImageNet](http://www.image-net.org/download-images) and export the data.

To search the strategy on VGG-16 with channel pruning on convolutional layers and fine-grained pruning on dense layers, and prunes 80% FLOPs reduction on convolutional layers, by running:

   ```
python agmc_network_pruning.py --dataset ILSVRC --model vgg16 --compression_ratio 0.8 --pruning_method cpfg --data_root data/datasets/dat1  --train_size 50000 --val_size 10000 --output ./logs
   ```
To search the strategy on MobileNet-V2 with channel pruning on convolutional layers and fine-grained pruning on dense layers, and prunes 30% FLOPs reduction on convolutional layers, by running:
   ```
python agmc_network_pruning.py --dataset ILSVRC --model mobilenetv2 --compression_ratio 0.3 --pruning_method cpfg --data_root [data_dir]  --train_size 50000 --val_size 10000 --output ./logs
   ```
## Evaluate the compressed Model
When searching, we evaluate the compressed Models with part of the validation set to speed up searching. And when we finished searching, we can evaluate the compressed Model on the whole validation set, which is saved on the default directory ```./logs```. 
For example, if we want to evaluate the performance of the compressed ResNet56 on CIFAR-10 py running:
   ```
python eval_compressed_model.py --dataset cifar10 --model resnet56 --pruning_method cp --data_root ./data --model_root ./logs/ResNet56.pkl
   ```
To evaluate the compressed VGG-16 on ILSVRC-2012, by running:
```

python eval_compressed_model.py --dataset ILSVRC --model vgg16 --pruning_method cpfg --data_root [data_dir] --model_root ./logs/vgg16.pkl
```
To evaluate the compressed MobileNet-v2 on ILSVRC-2012, by running:

```
python eval_compressed_model.py --dataset ILSVRC --model mobilenetv2 --pruning_method cpfg --data_root [data_dir] --model_root ./logs/mobilenetv2.pkl
```
## Results
| Models                   | Compressed ratio | Top1 Acc (%) |
| ------------------------ | ------------     | ------------ |
| ResNet-20                | 50% FLOPs        | 88.42        |
| ResNet-56                | 50% FLOPs        | 92.00         |
| ResNet-56                | 50% Params       | 95.64        |

| Models                   | Compressed ratio | Top5 Acc (%) |
| ------------------------ | ------------     | ------------ |
| VGG16                    | 80% FLOPs        | 80.73         |
| MobileNetV2              | 30% FLOPs        | 89.64       |


