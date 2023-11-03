# 实验结果和复现

## STL10为预训练数据集和shadow dataset
- batch size:256, lr:1e-4


# Todolist

- train_224测试filter

- attacker只能控制data，只知道下游任务的范围，解决filter可迁移的问题
- ctrl 控制不能训练过程，☞只能posioning data，这种asr是多少， 希望不好

# Done

- badencoder filter效果很烂 找出原因 提升效果 Self-supervised learning 学习率调大(SIMCLR)

- 为什么在监督里面不好，Self-supervised learning里面效果不好(例如WA-Net，ISSBA，BP-attack)，从而改进trigger。

- ABS：提出了filter attack，ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation

- simCLR:自监督，对比学习

# 疑问
- badencoder使用了不同的trigger来对应target class，我们只用了一个filter，是否需要多设计几个filter
<!-- - Imagenet的没有给出训练代码 -->


# 汇报
- 去掉那个loss好像有时候也可以攻击成功，试了两个SSMI 98%的filter，ASR一次是gtsrb、svhn：100%, stl10：0%，另一次是gtsrb：0%，svhn、stl10：100%
    - 每100个epoch尝试一下
- imagenet的gtsrb patch攻击效果不好
- SSL-backdoor的数据集val说需要文件夹，但是是图片
- CTRL跑出来了
- 第二个任务具体的目标