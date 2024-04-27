export CUDA_VISIBLE_DEVICES=2

attacks=('pgd' 'carlini' 'wasserstein' 'auto_pgd' 'ba' 'ba_mean') 

timestamp=$(date +%Y-%m-%d-%H-%M)

# mnist
# model_dir='log_fbf2/'

# for at in "${attacks[@]}"
# do
#    echo "Running with attack $at"  &
#    python attack_mnist.py --attack_type $at --method fbf --model_dir $model_dir >> log/$timestamp.log 2>&1 &
# done


# cifar10
model_dir='log_ba_cifar10_16/'

for at in "${attacks[@]}"
do
   echo "Running with attack $at"  &
   python attack_cifar10.py --attack_type $at --method ba --model_dir $model_dir >> log/$timestamp.log 2>&1 &
done
