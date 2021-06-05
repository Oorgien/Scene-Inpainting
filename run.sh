if [ $1 -eq 1 ]; then
	isort . 2>/dev/null
	autopep8 --in-place --ignore E501 --recursive . 2>/dev/null
elif [ $1 -eq 0 ]; then
	echo "Autopep8 and isort disabled"
fi
python main.py -cfg 'configs/configs_BestModel/config_init.yaml' -m 'train'
#python main.py -cfg 'configs/configs_EdgeGAN/config_bce_two_gens_imagenet.yaml' -m 'train'