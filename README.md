# RV PINN

Physics-informed neural network (PINN) framework for inferring right ventricular myocardial material parameters and stress-stretch behavior from hemodynamic measurements and ex vivo biaxial mechanics.

## Structure of files

- `train.py`: main training and evaluation entry point.
- `src/config.py`: central configuration and hyperparameters.
- `src/data_loader.py`: Excel loading, sample extraction, and train/validation/test splitting.
- `src/model.py`: TensorFlow PINN model.
- `src/physics.py`: strain invariants and constitutive stress calculations.
- `src/losses.py`: training loss functions.
- `src/trainer.py`: optimizer, training loop, and weight saving.
- `src/evaluation.py`: test prediction, CSV export, and plotting.

- `train_singlefile.py`: single-file version of the full training and evaluation pipeline.
- `train_singlefile_SSBroyden.py`: single-file version using the SS Broyden optimizer during training.
- `_optimize.py`: optimizer implementation file used for the SS Broyden optimization workflow.

- `RV_in-vivo_ex-vivo.xlsx`: combined in-vivo hemodynamic and ex-vivo biaxial mechanics dataset used by the training scripts.