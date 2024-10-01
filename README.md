# pharmaVQA

# Package Installation
You can utilize the environment.yaml to install the conda enviorment.

# Dataset Download
We have placed the dataset in the \URL, you can download it and put it in the ```\dataset``` folder.

# How to run
## Run for Li's 8 classification and 3 regression dataset
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

```
python fintune_pharVQA_ace.py --dataset CHEMBL204_Ki --batch_size 8 --lr 0.00003 --n_epochs 50 --device cuda:0 --alpha 0.5 --beta 0.1 --num_runs 3 --seed 66
```

## Run your own dataset
If you want to run your own dataset, you can run the following command to preprocess it firstly.
### prepare dataset
```
cd script

# construct the  graph feature
python preprocess_downstream_dataset.py --data_path ../datasets/your_dataset_folder --dataset your_dataset_name

# extract the pharmacophores feature
python extract_downstream_phar_features.py --data_path ../datasets/your_dataset_folder --dataset your_dataset_name
```
### finetune dataset
After preprocessing step, you can finetune your own dataset. the running command is the same as the ```How to run``` section.

