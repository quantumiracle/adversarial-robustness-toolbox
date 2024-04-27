export CUDA_VISIBLE_DEVICES=2


declare -a coefs=(0.00001 0.0001 0.0002 0.0005 0.0008 0.001 0.01 0.1 1 5 10 50) 
timestamp=$(date +%Y-%m-%d-%H-%M)

for i in "${coefs[@]}"
do
   echo "Running with coef $i"  &
   python adversarial_training_BA.py --delta_coeff "$i" --epochs 30 --output_dir 'log_ba_cifar10/' >> log/$timestamp.log 2>&1 &
done
