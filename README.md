# Deespo_gcn

服务器：Linux server4 4.15.0-163-generic #171-Ubuntu SMP Fri Nov 5 11:55:11 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux  
显卡：NVIDIA GeForce RTX 3090

## Install requirement：

### Install Gurobi
1.You can register and download ```Gurobi``` from here.  
2.解压文件到你的目录  
```
 tar -xvfz gurobi9.5.0_linux64.tar.gz
```
3.修改环境变量
```
vim ~/.bashrc
```
在文件最后添加
```
export GUROBI_HOME="/home/***/gurobi950/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```
保存后，输入
```
source ~/.bashrc #使环境变量生效
```
4.You need a liscense from gurobi. You can get one from here.  
5. 激活gurobi  
进入目录 gurobi9.5.0/linux64/bin 
run grbgetkey ******-f*e*-4f*8-2b*c-5*f8e*7*7*7f(申请的liscense)  

Installation instructions of other solvers will be given later.
### Install pytorch（Select the corresponding CUDA version）
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install scipy
pip install networkx
pip install tensorboardX
pip install tensorboard
pip install tensorflow
pip install tqdm
```

## Running the experiments:
### p-median
```
# Generate p-median samples
python 01_generate_samples.py p-median
# Training
python 02_train_gcn.py p-median -m gcn_3layers  
python 02_train_gcn.py p-median -m gcn_4layers  
python 02_train_gcn.py p-median  
python 02_train_gcn.py p-median -m gcn_6layers  
python 02_train_gcn.py p-median -m gcn_7layers  
# Test
python 03_test_gcn.py p-median  
# Evaluation
python 04_evaluate_result.py p-median
```

### p-center
```
# Generate p-center samples
python 01_generate_samples.py p-center
#### Training
python 02_train_gcn.py p-center-m gcn_3layers  
python 02_train_gcn.py p-center-m gcn_4layers  
python 02_train_gcn.py p-center  
python 02_train_gcn.py p-center-m gcn_6layers  
python 02_train_gcn.py p-center-m gcn_7layers  
# Test
python 03_test_gcn.py p-center
# Evaluation:
python 04_evaluate_result.py p-center
```

### MCLP
```
# Generate MCLP samples
python 01_generate_samples.py MCLP
# Training
python 02_train_gcn.py MCLP -m gcn_3layers  
python 02_train_gcn.py MCLP -m gcn_4layers  
python 02_train_gcn.py MCLP  
python 02_train_gcn.py MCLP -m gcn_6layers  
python 02_train_gcn.py MCLP -m gcn_7layers  
# Test
python 03_test_gcn.py MCLP 
# Evaluation
python 04_evaluate_result.py MCLP
```

### LSCP
```
# Generate LSCP samples
python 01_generate_samples.py LSCP
# Training:
python 02_train_gcn.py LSCP -m gcn_3layers  
python 02_train_gcn.py LSCP -m gcn_4layers  
python 02_train_gcn.py LSCP  
python 02_train_gcn.py LSCP -m gcn_6layers  
python 02_train_gcn.py LSCP -m gcn_7layers  
# Test
python 03_test_gcn.py LSCP
# Evaluation
python 04_evaluate_result.py LSCP
```
## Examples
Graph generation

## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.




