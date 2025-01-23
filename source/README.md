# ESPFire

# How to use
##  Dependent installation
```
pip install -r requirements.txt
```
## Show heatmap
```
python core/utils/visualize.py --path datasets/train
```
## Train 
```
python train.py --yaml configs/configs.yaml
```
## Evaluate on test set
```
python eval.py --yaml configs/configs.yaml --model models/fire_net.keras
```
## Test
```
python test.py  --yaml configs/configs.yaml 
                --model models/fire_net.keras
                --rgb_path datasets/train/nofire/1.jpg 
                --thermal_path datasets//train/nofire/1.txt           
```
<!-- * [FastestDet For object detection](https://github.com/dog-qiuqiu/FastestDet) -->