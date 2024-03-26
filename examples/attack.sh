export CUDA_VISIBLE_DEVICES=1

attacks=('pgd' 'carlini' 'wasserstein' 'auto_pgd' 'ba') 
timestamp=$(date +%Y-%m-%d-%H-%M)
model_dir='log_fbf2/'

for at in "${attacks[@]}"
do
   echo "Running with attack $at"  &
   python attack_mnist.py --attack_type $at --method ba --model_dir $model_dir >> log/$timestamp.log 2>&1 &
done

