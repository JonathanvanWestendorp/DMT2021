# DMT2021
### Assignment 1 - advanced
Dylan Prins, Naomi Tilon and Jonathan van Westendorp

## File explanations
- **Usedvariables.txt:** LSTM results with certain variable withheld
- **accandloss.png:** A snapshot of LSTM accuracy and loss
- **assignment_1_advanced.pdf:** the assignment
- **benchmark.py:** Script for generating benchmark accuracy on the data split in train and test sets
- **dataloader.py:** Cointains Dataset instance for pytorch training and a script for data processing
- **dataset_mood_smartphone.csv:** The used dataset
- **evaluatemodel.py:** Script for LSTM evaluation
- **lstm.py:** Contains LSTM model implementation in pytorch
- **svm.py:** Contains SVM training script
- **train.py:** Contains training script for LSTM
- **trained_model_seed_\*** Saved pre-trained LSTM models. Multiple seeds and history sizes have been tested.
  - W10: Window size of 10 -> a history of 10 days
  - seed_123 -> seed 123 used for random number generations in numpy and pytorch

Feel free to experiment with every model implementation. Code should run directly without error. See specific files for specific run instructions.
