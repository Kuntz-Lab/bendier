for sim in \
"scripts/cosserat/test_tip_priors.py --tip-prior force" \
"scripts/cosserat/test_tip_priors.py --tip-prior moment" \
"scripts/cosserat/test_tip_priors.py --tip-prior pose" \
"scripts/cosserat/test_spring_shape.py" \
"scripts/tendon_robot/test_simple.py" \
"scripts/tendon_robot/test_three_tendons.py" \
"scripts/parallel_robot/test_simple.py" \
"scripts/parallel_robot/test_three_rods.py"

do
    echo "Running $sim"
    python $sim
done
