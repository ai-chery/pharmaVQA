# pharmaVQA

# Installation
You can use the 

# Dataset Download
We have placed the dataset in the \URL, you can download it and put it in the ```\dataset``` folder.

# How to run
## Run for 8 classification datasets and 3 regression dataset
If you run the classification dataset, please specify the ```dataset_type``` to ```classification``` and the ```metric``` is ```rocauc```. Given one sample which is bace dataset: 

```
python fintune_pharVQA.py --dataset bace --dataset_type classification --metric rocauc --batch_size 8 --lr 0.00003 --n_epochs 50 --split scaffold-0 --device cuda:0
```

If you run the regression dataset, please specify the ```dataset_type``` to ```regression``` and the ```metric``` is ```rmse```. Given one sample which is lipo dataset: 

```
python fintune_pharVQA.py --dataset lipo --dataset_type regression --metric rmse --batch_size 8 --lr 0.00003 --n_epochs 50 --split scaffold-0 --device cuda:0
```


## Run for MoleculeACE dataset
If you run for the MoleculeACE benchmark dataset, you can run the following command:

``` python fintune_pharVQA_ace.py ```
## Run your own dataset
### prepare dataset

### finetune dataset
