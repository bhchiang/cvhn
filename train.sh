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

# 5: Complex with lower learning rate
python train.py --phase_path "$phase_path" --captured_path "$captured_path" --target_network "complexcnnc" --experiment test_activation --activation mod_relu
