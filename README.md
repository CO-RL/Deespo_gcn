# Deespo_gcn

服务器：Linux server4 4.15.0-163-generic #171-Ubuntu SMP Fri Nov 5 11:55:11 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux
显卡：NVIDIA GeForce RTX 3090

## Install requirement：
### 安装pytorch（需要根据显卡选择对应的cuda版本）
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install scipy
pip install networkx
pip install tensorboardX
pip install tensorboard
pip install tensorflow
pip install tqdm

## Running the experiments:
### p-median
#### training:
python 02_train_gcn.py p-median -m gcn_3layers  
python 02_train_gcn.py p-median -m gcn_4layers  
python 02_train_gcn.py p-median  
python 02_train_gcn.py p-median -m gcn_6layers  
python 02_train_gcn.py p-median -m gcn_7layers  
#### testing:
python 03_test_gcn.py p-median  
#### evalute result:
python 04_evaluate_result.py p-median

### p-center
#### training:
python 02_train_gcn.py p-center-m gcn_3layers  
python 02_train_gcn.py p-center-m gcn_4layers  
python 02_train_gcn.py p-center  
python 02_train_gcn.py p-center-m gcn_6layers  
python 02_train_gcn.py p-center-m gcn_7layers  
#### testing:
python 03_test_gcn.py p-center
#### evalute result:
python 04_evaluate_result.py p-center

### MCLP
#### training:
python 02_train_gcn.py MCLP -m gcn_3layers  
python 02_train_gcn.py MCLP -m gcn_4layers  
python 02_train_gcn.py MCLP  
python 02_train_gcn.py MCLP -m gcn_6layers  
python 02_train_gcn.py MCLP -m gcn_7layers  
#### testing:
python 03_test_gcn.py MCLP 
#### evalute result:
python 04_evaluate_result.py MCLP

### LSCP
#### training:
python 02_train_gcn.py LSCP -m gcn_3layers  
python 02_train_gcn.py LSCP -m gcn_4layers  
python 02_train_gcn.py LSCP  
python 02_train_gcn.py LSCP -m gcn_6layers  
python 02_train_gcn.py LSCP -m gcn_7layers  
#### testing:
python 03_test_gcn.py LSCP
#### evalute result:
python 04_evaluate_result.py LSCP







