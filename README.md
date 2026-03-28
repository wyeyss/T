# Cross-city Few-shot Traffic Forecasting via Traffic Pattern Bank

[CIKM 2023] In this repository, we presents the code of "Cross-city Few-shot Traffic Forecasting via Traffic Pattern Bank" (TPB).

![TPB](./fig/mainfig.png)


## Data

The data is in https://drive.google.com/drive/folders/1UrKTgR27YmP9PjJ-FWv4SCDH3zUxtc5R?usp=share_link.
Please download it and save them in `./data`

## Environment
The code is implemented in pytorch 1.10.0, CUDA version 11.3, python 3.7.0.

```bash
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## Reproducibility
The default configs of the four datasets are set in `./config`.
To reproduce the result, please run following command:
```bash
bash train.sh
```
or run the experiment on specific dataset (PEMS-BAY as an example):
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --config_file ./configs/config_pems.yaml > train_pems.out 2>&1 &
```

## Pre-trained stuff

The pre-trained patch encoder and traffic pattern bank is contained in this repository.
The pre-trained patch encoder is in `./save/pretrain_model` and the traffic pattern bank is in `./pattern`.

You can also pre-train and generate traffic pattern bank on your own by:
```bash
# Pre-train
python -u ./pretrain.py --test_dataset ${test_dataset} --data_list ${data_list}
wait
python -u ./patch_devide.py --test_dataset ${test_dataset} --data_list ${data_list}
wait
python -u ./pattern_clustering.py --data_list $data_list --test_dataset ${test_dataset} --sim ${sim} --K ${K}
```
`${data_list}` is the source data. For example, if you want to pre-train the encoder in `Chengdu`, `METR-LA` and `PEMS-BAY`, then `${data_list}` is `chengdu_metr_pems`.

`${test_dataset}` is the dataset you want to build target data on. If you want to build target data on `Shenzhen` then the `${test_dataset}` is `shenzhen`.

`${sim}` and `${K}` are the clustering hyper-parameter. You can set them by your own.


# 基于交通模式库的跨城市少样本交通预测

[CIKM 2023] 本仓库提供了《基于交通模式库的跨城市少样本交通预测》（TPB）的代码。

![TPB](./fig/mainfig.png)


## 数据

数据位于 https://drive.google.com/drive/folders/1UrKTgR27YmP9PjJ-FWv4SCDH3zUxtc5R?usp=share_link。
请下载数据并将其保存到 `./data` 目录下。

## 环境
代码基于 pytorch 1.10.0、CUDA 11.3 版本和 python 3.7.0 实现。

```bash
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## 复现性
四个数据集的默认配置设置在 `./config` 目录下。
要复现结果，请运行以下命令：
```bash
bash train.sh
```
或者在特定数据集上运行实验（以 PEMS-BAY 为例）：
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --config_file ./configs/config_pems.yaml > train_pems.out 2>&1 &
```

## 预训练内容

预训练的补丁编码器和交通模式库包含在本仓库中。
预训练的补丁编码器位于 `./save/pretrain_model` 目录下，交通模式库位于 `./pattern` 目录下。

您也可以自行预训练并生成交通模式库，命令如下：
```bash
# 预训练
python -u ./pretrain.py --test_dataset ${test_dataset} --data_list ${data_list}
wait
python -u ./patch_devide.py --test_dataset ${test_dataset} --data_list ${data_list}
wait
python -u ./pattern_clustering.py --data_list $data_list --test_dataset ${test_dataset} --sim ${sim} --K ${K}
```
`${data_list}` 是源数据。例如，如果您想在 `成都`、`METR-LA` 和 `PEMS-BAY` 上预训练编码器，那么 `${data_list}` 就是 `chengdu_metr_pems`。

`${test_dataset}` 是您要构建目标数据所基于的数据集。如果您想在 `深圳` 上构建目标数据，那么 `${test_dataset}` 就是 `shenzhen`。

`${sim}` 和 `${K}` 是聚类超参数。您可以自行设置它们的值。