# Boosting Multi-modal Model Performance with Adaptive Gradient Modulation
Here is the official Pytorch implementation of AGM proposed in "Boosting Multi-modal Model Performance with Adaptive Gradient Modulation".

**Paper Title: Boosting Multi-modal Model Performance with Adaptive Gradient Modulation**

**Authors: Hong Li<sup> * </sup>, Xingyu Li<sup> * </sup>, Pengbo Hu, Yinuo Lei, Chunxiao Li, Yi Zhou**

**Accepted by: ICCV 2023**

[[arXiv](https://arxiv.org/abs/2308.07686)] [[ICCV Proceedings](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Boosting_Multi-modal_Model_Performance_with_Adaptive_Gradient_Modulation_ICCV_2023_paper.html)]

## Dataset
### 1. AV-MNIST

This dataset can be downloaded from [here](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing).

### 2. CREMA-D

This dataset can be downloaded from [here](https://github.com/CheyneyComputerScience/CREMA-D). Data preprocessing can refer to [here](https://github.com/GeWu-Lab/OGM-GE_CVPR2022/tree/main/data/CREMAD).

### 3. UR-Funny

This raw dataset can be downloaded from [here](https://github.com/ROC-HCI/UR-FUNNY). Also, the processed data can be obtained from [here](https://github.com/ROC-HCI/UR-FUNNY).

### 4. AVE
This dataset can be downloaded from [here](https://sites.google.com/view/audiovisualresearch).

### 5. CMU-MOSEI

This dataset can be downloaded from [here](https://drive.google.com/file/d/1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P/view?usp=sharing).

## Training

### Environment config
1. Python: 3.9.13
2. CUDA Version: 11.3
3. Pytorch: 1.12.1
4. Torchvision: 0.13.1
### Train
To train the model using the following command:
```python 
python main.py --data_root '' --device cuda:0 --methods Normal --modality Multimodal --fusion_type late_fusion --random_seed 999 --expt_dir checkpoint --expt_name test --batch_size 64 --EPOCHS 100 --learning_rate 0.0001 --dataset AV-MNIST --alpha 2.5 --SHAPE_contribution False
```

## Citation
```
@inproceedings{li2023boosting,
  title={Boosting Multi-modal Model Performance with Adaptive Gradient Modulation},
  author={Li, Hong and Li, Xingyu and Hu, Pengbo and Lei, Yinuo and Li, Chunxiao and Zhou, Yi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22214--22224},
  year={2023}
}
```
