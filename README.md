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
   To search the strategy on ResNet-56 with channel pruning, and prunes 50% FLOPs reduction ,by running:
   ```
python agmc_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.5 --pruning_method cp --data_root ./data --output ./logs
   ```
   To search the strategy on ResNet-56 with fine-grained pruning, and prunes 50% params,by running:
   ```
python agmc_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.5 --pruning_method fg --data_root ./data --output ./logs
   ```


### ILSVRC-2012
To evaluates the AGMC on the ILSVRC-2012 dataset, you need first download the dataset from [ImageNet](http://www.image-net.org/download-images), and export the data.

To search the strategy on VGG-16 with channel pruning on convolutional layers and fine-grained pruning on dense layers, and prunes 50% FLOPs reduction on convolutional layers,by running:

   ```
python agmc_network_pruning.py --dataset ILSVRC --model vgg16 --compression_ratio 0.5 --pruning_method cpfg --data_root [ILSVRC_dir] --output ./logs
   ```

## Evaluate the compressed Model
After searching we can evaluate the compressed Model, which are saved on defualt directory ```./logs``` . 
For example, if we want to evaluate the performance of compressed Models py running:
   ```
python eval_compressed_model.py --dataset cifar10 --model vgg16 --compression_ratio 0.5 --pruning_method cpfg --data_root [ILSVRC_dir] --output ./logs
   ```

## Results
| Models                   | Compressed ratio | Top5 Acc (%) |
| ------------------------ | ------------     | ------------ |
| ResNet-20                | 50% FLOPs        | 88.42         |
| ResNet-56                | 50% FLOPs        | **92.0**   |
| ResNet-56                | 50% Params       | **95.64**   |
| VGG16                    | 80% FLOPs        | 89.6         |
| MobileNetV2              | 30% FLOPs        | **89.914**   |
