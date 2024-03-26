export CUDA_VISIBLE_DEVICES=2

timestamp=$(date +%Y-%m-%d-%H-%M)


# echo "Running fbf cifar10"  &
# python adversarial_training_FBF.py --epochs 30 --output_dir 'log_fbf_cifar10/' >> log/$timestamp.log 2>&1 &
echo "Running ba cifar10"  &
python adversarial_training_BA.py --epochs 30 --output_dir 'log_ba_cifar10/' >> log/$timestamp.log 2>&1 &
