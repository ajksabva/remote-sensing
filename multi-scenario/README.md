**实验环境**

+ 硬件配置

  + CPU: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
  + GPU: Tesla V100-SXM2-32gb

+ 软件配置

  + Ubuntu 16.04.6 LTS

  + Python=3.7.12 
  + CUDA=9.2 
  + Pytorch=1.7.1 
  + MMCV=1.6.1
  + MMDet=2.28.2



**环境配置**

```bash
git clone git@github.com:ajksabva/remote-sensing.git
cd remote-sensing
chmod +x ./prepare_data.sh
sh ./prepare_data,sh

# 多场景
conda create -n multi-scenario python==3.7.12 -y
conda activate multi-scenario

# pytorch 1.7.1 + cu92
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# mmcv 1.6.1 && mmdet 2.28.2
wget https://download.openmmlab.com/mmcv/dist/cu92/torch1.7.0/mmcv_full-1.6.1-cp37-cp37m-manylinux1_x86_64.whl
pip install filelock fsspec requests tqdm yapf==0.40.0 timm ./mmcv_full-1.6.1-cp37-cp37m-manylinux1_x86_64.whl mmdet==2.28.2 && rm ./mmcv_full-1.6.1-cp37-cp37m-manylinux1_x86_64.whl 


cd multi-scenario
pip install -e .
```



**训练**

```bash
python tools/train.py configs/rotated_imted_oriented_rcnn_vit_base_3x_masativ2_rr_le90_stdc_xyawh321v.py --work-dir ./val --gpus 1    
```



**测试**

```bash
python tools/test.py configs/rotated_imted_oriented_rcnn_vit_base_3x_masativ2_rr_le90_stdc_xyawh321v.py data/models/multi-scenario.pth --eval mAP --work-dir ./val --gpu-ids 0 --show-dir ./show 
```

