source common.sh

# mod_relu
# pretrained_path="final_models/green_exp10_Targetcomplexcnnc-Activationlearnable_mod_relu-Norminstance_LossL1_lr5e-05_Optimizercomplex_adam_model_4epoch.pth"
# python evaluate.py --phase_path "$phase_path" --captured_path "$captured_path" \
#     --target_network "complexcnnc" \
#     --experiment  evaluate_complex_relu \
#     --activation learnable_mod_relu \
#     --pretrained_path "$pretrained_path" \
#     --optimizer complex_adam

# complex_cardiod
# pretrained_path="final_models/green_exp8_Targetcomplexcnnc-Activationcomplex_cardiod-Norminstance_LossL1_lr5e-05_Optimizercomplex_adam_model_4epoch.pth"
# python evaluate.py --phase_path "$phase_path" --captured_path "$captured_path" \
#     --target_network "complexcnnc" \
#     --experiment  evaluate_complex_cardiod \
#     --activation complex_cardiod \
#     --pretrained_path "$pretrained_path" \
#     --optimizer complex_adam


# complex_relu
pretrained_path="final_models/green_exp7_Targetcomplexcnnc-Activationcomplex_relu-Norminstance_LossL1_lr5e-05_Optimizercomplex_adam_model_6epoch.pth"
python evaluate.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment  evaluate_complex_relu \
    --activation complex_relu \
    --pretrained_path "$pretrained_path" \
    --optimizer complex_adam


# real_relu
pretrained_path="final_models/green_exp6_Targetcomplexcnnc-Activationreal_relu-Norminstance_LossL1_lr5e-05_Optimizercomplex_adam_model_6epoch.pth"
python evaluate.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment  evaluate_real_relu \
    --activation real_relu \
    --pretrained_path "$pretrained_path" \
    --optimizer complex_adam