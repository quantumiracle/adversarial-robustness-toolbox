export CUDA_VISIBLE_DEVICES=1

# declare -a coefs=(0.01 0.05 0.1 0.2 0.5 1 1.25 2 3 5) 
# declare -a coefs=(7 9 10 15 20 50) 
# declare -a coefs=(0.01 0.05 0.1 0.2 0.5 1 1.25 2 3 5 7 9 10 15 20 50) 
# declare -a coefs=(0.001 0.01 0.1 1 5 10 50) 
declare -a coefs=(0.00001 0.0001 0.0002 0.0005 0.0008 0.001 0.01 0.1 1 5 10 50) 
timestamp=$(date +%Y-%m-%d-%H-%M)

# for i in "${coefs[@]}"
# do
#    echo "Running with coef $i"  &
#    # python adversarial_training_BA_mnist.py --delta_coeff "$i"   &
#    python adversarial_training_BA_mnist.py --delta_coeff "$i" --epochs 30 --output_dir 'log30_iter_delta_eps0.2/' >> log/$timestamp.log 2>&1 &
# done

for i in {1..2}; do
   echo "Running BA mnist $i" &
   python adversarial_training_BA_mnist.py --delta_coeff 10 --epochs 30 --output_dir log_ba${i}/ >> log/$timestamp.log 2>&1 &
   echo "Running FBF mnist $i" &
   python adversarial_training_FBF_mnist.py --output_dir log_fbf${i}/ >> log/$timestamp.log 2>&1 &
done