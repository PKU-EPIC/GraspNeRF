cd src/nr
CUDA_VISIBLE_DEVICES=$1 python run_training.py --cfg configs/nrvgn_sdf.yaml
cd -