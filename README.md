# cvhn

Complex Valued Holography Networks

# About

This is the repository corresponding to our project titled "Learned Propagation Model with Complex Convolutions for Holographic Systems" for [EE 367](http://stanford.edu/class/ee367/) in Winter 2021.

The authors are Manu Gopakumar (manugopa@stanford.edu) and Bryan Chiang (bhchiang@stanford.edu).

## Dependencies

First, install JAX with GPU support enabled.

```sh
pip install --upgrade jax jaxlib==0.1.62+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Depending on your CUDA version, you might need to change to `+cuda111` or `+cuda112`. All versions are listed here (https://storage.googleapis.com/jax-releases/jax_releases.html), scroll down for the most recent CUDA releases.

Install Torch following their official instructions: https://pytorch.org/get-started/locally/. For example:

```sh
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install the rest of the requirements.

```sh
pip install -r requirements.txt
```

## Usage

Note: the dataset has not been published and as a result is not publicly available. You can find more details about the dataset inside our paper.

Find the root location of the dataset. It should contain a folder called `holography_training_data`.

```sh
readlink -f /path/to/data
```

In `common.sh`, set the location of `prefix` to the root location of your dataset.

### Training

Training happens with `train.py`. This creates a network of the specified type and trains is over the training set with periodic evaluations on the validation set.

```sh
source common.sh
python train.py --phase_path "$phase_path" --captured_path "$captured_path" \
    --target_network "complexcnnc" \
    --experiment exp6_trial2 \
    --activation real_relu \
    --optimizer complex_adam
    --lr_model 5e-5
```

A full list of options and their descriptions are available in `train_helper.py`.

You can see a list of different training commands in `train.sh`.

The three types of networks are:

- `cnnr`: Real network that operates on the amplitude of the phase.
- `stackedcnnc`: Convert complex phase to real, 2-channel representation.
- `complexcnnc`: Treat phase as complex, entire network has complex valued weights and biases.

### Evaluation

Evaluation happens with `evaluate.py`. This script runs a model (provided via `pretrained_path`) over the test set, providing L1 and MSE (-> PSNR) metrics.

We've provided sample commands for running evaluation over the actual test set in `evaluate.sh`.

We've also provided a small dataset in `data` containing several images from the actual test set for quick evaluation.

You can download our best Stacked CNNC and Complex CNNC models from [Google Drive](https://drive.google.com/drive/folders/1q5TsIo7rFdlCb0T4uP1wpJw1oMlpxdeN?usp=sharing), and put them into a folder called `final_models`.

You can then run `toy_evaluate.sh` to get the results.

```sh
chmod +x ./toy_evaluate.sh
./toy_evaluate.sh
```

```sh
# toy_evaluate.sh
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
```

These were our results on our setup:

```txt
# Stacked CNNC
avg_mse = 0.0002837978463503532, avg_objective = 0.010521531105041504

# Complex CNNc with real_relu
avg_mse = 0.00035572584602050483, avg_objective = 0.011939069256186485
```

### Artifacts

Models will be saved to the `models/` folder, and TensorBoard logging will be written to `JAX_runs/`. You can start TensorBoard with the following command;

```sh
make tb
```

### Code Structure

- `optimize.py` contains our custom complex optimizers.
- `phase_capture_loader.py` contains the Torch DataLoader for running the dataset.
- `complex_activations.py` contains the complex activations functions for ComplexCNNC.
- `asm.py` contains a JAX implementation of the angular spectrum method for free-space propagation.
- `train.py` contains our training script.
- `evaluate.py` contains our evaluation script.
