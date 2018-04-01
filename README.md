# ANN-using-Back-Propagation

Pre-Processing :
----------------
python preprocessing.py <url> <output_file_path>

for instance : python preprocessing.py https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data C:/Users/pxg131330/Documents/

Run the code like :
-------------------

python ann.py <dataset> <training_percent> <no_of_perceptrons> <no_of_hidden_layers> <neurons_in_hidden_layer>

for instance : python ann.py C:/Users/kgasw/Grad/sem1/standardized_adult_onehot.csv 80 1000 2 10 20

utilities.py has functions needed by ann.py to train model.

Please modify learning_rate from within the code for better results, since the question didn't ask us to pass this as a parameter.
