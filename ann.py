import utilities
import pandas as pd
import sys
data_set_path = sys.argv[1]
training_percent = int(sys.argv[2])
iterations = int(sys.argv[3])
no_of_layers = int(sys.argv[4])
layer_perceptrons = []
learning_rate = 0.1
for i in range(0,no_of_layers) :
    layer_perceptrons.append(int(sys.argv[5+i]))
data = pd.read_csv(data_set_path)
column_list = data.columns.values
attr = column_list[:-1]
classname = column_list[-1]
no_of_attributes = len(attr)
no_of_bits = len(utilities.getBinaryArray(data[classname].max()))
array = data.as_matrix()
max_array = []
for k in range(0,len(array)):
    max_array.append(max(array[k]))
train_data,test_data = utilities.split(array,training_percent)
output_perceptrons = no_of_bits
layer_perceptrons.append(output_perceptrons)
layer_random_count = no_of_attributes
print('Getting random weights ...')
weights = utilities.getRandomWeights(layer_perceptrons,layer_random_count)
print('Populated random weights ...')
print('Building model ...')
for i in range(0,len(train_data)):
    expected_output = []
    expected_output = utilities.modifyBinaryArray(int(train_data[i][-1:]),no_of_bits)
    utilities.back_propogation(weights, train_data[i][:-1], layer_perceptrons, 0, expected_output, learning_rate, iterations)
print('ANN created ...')
print('')
utilities.printWeights(weights,layer_perceptrons)
print('------------------------------------------')
print('Predicting train and test classification : ')
print('------------------------------------------')
squared_error,accuracy = utilities.getAccuarcy(train_data,weights,layer_perceptrons,no_of_bits)
print('')
print('Training data : ')
print('---------------')
print('Accuracy : ',accuracy)
print('Squared_error : ',squared_error)
squared_error,accuracy = utilities.getAccuarcy(test_data,weights,layer_perceptrons,no_of_bits)
print('')
print('Testing data : ')
print('--------------')
print('Accuracy : ',accuracy)
print('Squared_error : ',squared_error)
