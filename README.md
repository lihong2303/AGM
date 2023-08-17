# Boosting Multi-modal Model Performance with Adaptive Gradient Modulation


## Dataset
### 1. AV-MNIST

This dataset can be downloaded from [here](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing).

### 2. CREMA-D

This dataset can be downloaded from [here](https://github.com/CheyneyComputerScience/CREMA-D). Data preprocessing can refer to [here](https://github.com/GeWu-Lab/OGM-GE_CVPR2022/tree/main/data/CREMAD).

### 3. UR-Funny

This raw datasets can be downloaded from [here](https://github.com/ROC-HCI/UR-FUNNY). Also, the processed data can be obtained from [here](https://github.com/ROC-HCI/UR-FUNNY).

### 4. AVE
This dataset can be downloaded from [here](https://sites.google.com/view/audiovisualresearch).

### 5. CMU-MOSEI

This dataset can be downloaded from [here](https://drive.google.com/file/d/1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P/view?usp=sharing).

## Training
To train the model using the following command:
```python 
python main.py --data_root '' --device cuda:0 --methods Normal --modality Multimodal --fusion_type late_fusion --random_seed 999 --expt_dir checkpoint --expt_name test --batch_size 64 --EPOCHS 100 --learning_rate 0.0001 --dataset AV-MNIST --alpha 2.5 --SHAPE_contribution False