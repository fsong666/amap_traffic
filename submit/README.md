#模型预测
##1.运行
###1.1 切换目录到 ../amap_traffic_project/code   
###1.2 运行shell测试程序   
    ./run_test.sh 
##2.环境和依赖
|   name    | version |
| ---------- | --- |
| Operating System|  Ubuntu16.04 |
| GPU       |  NVIDIA-SMI 410.78 |
| CUDA      |  10.0 |
| cuDNN     |  7.6.2|
| Virtual environment     |  anaconda  |
| Python    | 3.5.6 |
| Pytorch   | 1.2.0|
##3.算法
第一阶段是ResNet152,加三个全连接层

第二阶段是三层的LSTM,加两个全连接层



