# Toy example on the small dataset
phase_path="data/phase"
captured_path="data/captured"

# Best Stacked CNNc
pretrained_path="final_models/green__Targetstackedcnnc-Activationrelu-Norminstance_L1loss_lr0.0005_outerskipTrue_model_11epoch.pth"
python evaluate.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "stackedcnnc" \
    --experiment evaluate_best_stacked_cnnc \
    --pretrained_path "$pretrained_path" \

# Best real_relu for ComplexCNNc
pretrained_path="final_models/green_exp6_Targetcomplexcnnc-Activationreal_relu-Norminstance_LossL1_lr5e-05_Optimizercomplex_adam_model_6epoch.pth"
python evaluate.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment evaluate_real_relu \
    --activation real_relu \
    --pretrained_path "$pretrained_path" \
    --optimizer complex_adam