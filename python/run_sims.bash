# Set constant fast clock speed
sudo cpupower frequency-set -g performance

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run all sims with same execution settings
for sim in \
tests.cosserat.test_num_nodes \
tests.tendon_robot.test_nees \
tests.tendon_robot.test_tip_force \
tests.parallel_robot.test_tip_force 


do
    echo "Running $sim"
    nice -n 0 taskset -c 2 python -m $sim
done