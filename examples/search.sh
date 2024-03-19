export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# declare -a coefs=(0.01 0.05 0.1 0.2 0.5 1 1.25 2 3 5) 
# declare -a coefs=(7 9 10 15 20 50) 
# declare -a coefs=(0.01 0.05 0.1 0.2 0.5 1 1.25 2 3 5 7 9 10 15 20 50) 
# declare -a coefs=(0.001 0.01 0.1 1 5 10 50) 
declare -a coefs=(0.00001 0.0001 0.0002 0.0005 0.0008 0.001 0.01 0.1 1 5 10 50) 

for i in "${coefs[@]}"
do
   echo "Running with coef $i"  &
   # python adversarial_training_BA_mnist.py --delta_coeff "$i"   &
   python adversarial_training_BA_mnist.py --delta_coeff "$i" --epochs 30 --output_dir 'log30_iter_delta_eps0.3/'  &
done