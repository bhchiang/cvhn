### REPLACE 
prefix=/media/bryan/easystore
###
phase_path="$prefix/holography_training_data/11planes_pairs_1700/phase/"
captured_path="$prefix/holography_training_data/11planes_pairs_1700/captured/"

echo "Phase path: $phase_path"
echo "Captured path: $captured_path"

# 1: CNNr with default parameters 
# python train.py --phase_path "$phase_path" --captured_path "$captured_path"

# 2: CNNr with no outer_skip
# python train.py --phase_path "$phase_path" --captured_path "$captured_path" --outer_skip false

# 3: CNNcStacked with default parameters
python train.py --phase_path "$phase_path" --captured_path "$captured_path" --target_network "stackedcnnc"
