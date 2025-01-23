run_all: train test eval

train:
	python train.py --yaml configs/configs.yaml --model_type r2plus1d-18 --model_name r2plus1d
eval:
	python eval.py --yaml configs/configs.yaml --model models/r2plus1d.keras
test:
	python test.py 	--yaml configs/configs.yaml --model models/r2plus1d.keras -video_path  
show:
	python show.py --yaml configs/configs.yaml --path datasets/fire/train 

