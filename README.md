```
python convert_dataset_tocsv.py
```

```
brainome -headerless -nosplit cifar_train.csv -o cifar_out.py
```

## How to run our ResNet18 model on CIFAR-10
```
python train.py --final_dim 100
python train2.py --final_dim 100
```

Ref: https://github.com/kuangliu/pytorch-cifar
