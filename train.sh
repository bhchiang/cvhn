### REPLACE 
prefix=/media/bryan/easystore
###
phase_path="$prefix/holography_training_data/11planes_pairs_1700/phase/"
captured_path="$prefix/holography_training_data/11planes_pairs_1700/captured/"

echo "Phase path: $phase_path"
echo "Captured path: $captured_path"

# 1: CNNr with default parameters 
# python train.py --phase_path "$phase_path" --captured_path "$captured_path" --experiment test_shape

# 2: CNNr with no outer_skip
# python train.py --phase_path "$phase_path" --captured_path "$captured_path" --outer_skip false

# 3: CNNcStacked with default parameters
# python train.py --phase_path "$phase_path" --captured_path "$captured_path" --target_network "stackedcnnc"

# 4: Complex
# python train.py --phase_path "$phase_path" --captured_path "$captured_path" --target_network "complexcnnc" --experiment complex2


# Real Adam, complex cardiod
python train.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment exp5 \
    --activation complex_cardiod \
    --optimizer real_adam
    --lr_model 5e-5

# Complex Adam, real_relu
python train.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment exp6 \
    --activation real_relu \
    --optimizer complex_adam
    --lr_model 5e-5

# Complex Adam, complex relu
python train.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment exp7 \
    --activation complex_relu \
    --optimizer complex_adam
    --lr_model 5e-5

# Complex Adam, complex cardiod
python train.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment exp8 \
    --activation complex_cardiod \
    --optimizer complex_adam
    --lr_model 5e-5

# Complex Adam, fixed_mod_relu
python train.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment exp9 \
    --activation fixed_mod_relu \
    --optimizer complex_adam
    --lr_model 5e-5

# Complex Adam, learnable_mod_relu
python train.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment exp10 \
    --activation learnable_mod_relu \
    --optimizer complex_adam
    --lr_model 5e-5
