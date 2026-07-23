# Set constant fast clock speed
sudo cpupower frequency-set -g performance

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run all sims with same execution settings
for sim in \
scripts/cosserat_rod/test_num_nodes.py \
scripts/tendon_robot/test_nees.py \
scripts/tendon_robot/test_tip_force.py \
scripts/parallel_robot/test_tip_force.py

do
    echo "Running $sim"
    nice -n 0 taskset -c 2 python $sim
done
