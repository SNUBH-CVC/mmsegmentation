## Installation
```
sudo chown -R whikwon:whikwon /opt/conda
pip install -r requirements/runtime.txt
pip install -r requirements_snubhcvc.txt
pip install mmcv==2.1.0 mmengine mmpretrain ftfy regex scipy prettytable # mmpretrain은 pretrained model 다운로드 받을 때 필요
python setup.py develop

# download pretrained vit
python tools/model_converters/vit2mmseg.py https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth pretrain/jx_vit_base_p16_224-80ecf9dd.pth 

# download pretrained DINOv2
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -P ./pretrain
```

## Train
```
# ViT-UperNet
CUDA_VISIBLE_DEVICES=1 PORT=29500 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_8xb2-80k_cag-512x512.py 1

# DINOv2-UperNet
CUDA_VISIBLE_DEVICES=1 PORT=29500 tools/dist_train.sh configs/vit/dinov2_vit-b14_mln_upernet_8xb2-80k_cag-518x518.py 1
```

## Tips 
아래 코드를 포함해야 registry가 정상적으로 동작한다. 
```
from mmengine.registry import init_default_scope
```

`SegDataPreProcessor`에서 normalization을 진행한다. normalize가 dataset에 없으니 헷갈리지 않길 

VQGAN 사용 시 DDP 사용하면 `find_unused_parameters` 관련 에러가 발생한다. config 파일 가장 위에 아래 내용을 추가해주자.
https://mmengine.readthedocs.io/en/v0.8.4/common_usage/debug_tricks.html
```
model_wrapper_cfg=dict(
    type='MMDistributedDataParallel', find_unused_parameters=True)
```

`UperNet` decoder를 거친 후에 크기가 1/4이 되어 있는 것을 확인할 수 있는데 이는 `EncoderDecoder`의 `postprocess_result`를 통과해서 최종적으로 resize 된다.