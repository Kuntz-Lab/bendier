for sim in \
"scripts/cosserat/spring_sim.py" \
"scripts/cosserat/test_priors_sim.py --tip-prior force" \
"scripts/cosserat/test_priors_sim.py --tip-prior moment" \
"scripts/cosserat/test_priors_sim.py --tip-prior pose" \
"scripts/tendon_robot/test_simple.py" \
"scripts/parallel_robot/test_simple.py"

do
    echo "Running $sim"
    python $sim
done
