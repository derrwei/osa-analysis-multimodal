# PhD project tools

## Inherent from Ning
- deepsound/
- inference.py 

## New Functions
- lightning_modules/ --> training scripts to use lightning
    - the dataset was reduced size into float32 and saved in *.dat; which are ignored when pushing
- TO Training: using ./run_train_all_folds.sh
- TO Evaluate: ./eval_all_folds
    - TODO: add more arguments for seleting which model and exp setup to use for inference