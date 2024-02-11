declare -a coefs=(0.01 0.05 0.1 0.2 0.5 1 1.25 2 3 5) 

for i in "${coefs[@]}"
do
   echo "Running with coef $i"  &
   python adversarial_training_BA_mnist.py --delta_coeff "$i"   &
done