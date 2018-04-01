import math
import random

def sigmoidFn(value):
    return (1/(1+math.exp(-1*value)))

def getRandomWeights(layer_perceptrons,layer_random_count):
    weights = []
    for i in range(0,len(layer_perceptrons)):
        for j in range(0,layer_perceptrons[i]):
            list = []
            for k in range(0,layer_random_count+1):
                list.append(random.uniform(-1, 1))
            weights.append(list)
        layer_random_count = layer_perceptrons[i]
    return weights

def split(array,training_percent):
    train_data = []
    test_data = []
    total_instances = len(array)
    total_train_instances = int((training_percent/100)*total_instances)
    total_test_instances = total_instances - total_train_instances
    test_step = int(total_instances/total_test_instances)
    for i in range(0, total_instances):
        if i%test_step == 0 and total_test_instances>0 :
            test_data.append(array[i])
            total_test_instances-=1
        else:
            train_data.append(array[i])
    return train_data,test_data

def modifyBinaryArray(value,no_of_bits):
    binary = []
    while value>0 :
        binary.append(value%2)
        value = round(value/2)
    binary.reverse()
    bits_binary = []
    for i in range (0, (no_of_bits-(len(binary)))):
        bits_binary.append(0)
    bits_binary.extend(binary)
    return bits_binary

def getBinaryArray(value):
    binary = []
    while value>0 :
        binary.append(value%2)
        value = round(value/2)
    binary.reverse()
    return binary

def printWeights(weights,layer_perceptrons):
    print('---------------')
    print('Updated Weights')
    print('---------------')
    neuron_index=0
    print('')
    for i in range(0,len(layer_perceptrons)):
        print('Layer ',i)
        print('--------')
        for j in range(0,layer_perceptrons[i]):
            print('Neuron',j+1,' Weights : ',weights[neuron_index])
            neuron_index +=1
            print('')

def perceptron(weights,input,index):
    bias = weights[index][0]
    net = bias
    for i in range(0,len(input)):
        net += input[i]*weights[index][i+1]
    return sigmoidFn(net)

def isLastValue(inputList):
    return (len(inputList) == 1)

def hasValue(inputList):
    return (len(inputList) > 0)

def layer_recurse(weights,input,layer_perceptrons,perceptron_no,expected_output,learning_rate):
    first_perceptron_layer = perceptron_no
    perceptron_output = []
    for i in range(0,layer_perceptrons[0]):
        perceptron_output.append(perceptron(weights, input, perceptron_no))
        perceptron_no+=1
    delta = []
    if(isLastValue(layer_perceptrons)):
        flag = 1
        for j in range(0,len(expected_output)):
            delta_value = perceptron_output[j] * (1 - perceptron_output[j]) * (expected_output[j] - perceptron_output[j])
            delta.append(delta_value)
            weights[first_perceptron_layer+j][0] = weights[first_perceptron_layer+j][0]+(learning_rate*delta_value)
            if perceptron_output[j] != expected_output[j] :
                flag = 0
        if flag == 0 :
            return delta
        else:
            return []
    else:
        prev_delta = layer_recurse(weights, perceptron_output, layer_perceptrons[1:], perceptron_no,expected_output,learning_rate)
        if hasValue(prev_delta):
            for j in range(0, len(perceptron_output)):
                weight_delta_sum = 0
                for k in range(0,len(prev_delta)):
                    weight_delta_sum += weights[perceptron_no+k][j+1]*prev_delta[k]
                    weights[perceptron_no + k][j + 1] += learning_rate*prev_delta[k]*perceptron_output[j]
                delta_value = perceptron_output[j] * (1 - perceptron_output[j]) * weight_delta_sum
                weights[first_perceptron_layer + j][0] += (learning_rate * delta_value)
                delta.append(delta_value)
            return delta
        else:
            return []

def back_propogation(weights,input,layer_perceptrons,perceptron_no,expected_output,learning_rate,iterations):
    for i in range(0,iterations):
        delta = layer_recurse(weights, input, layer_perceptrons, perceptron_no, expected_output, learning_rate)
        if hasValue(delta):
            for j in range(0,len(input)):
                for k in range(0,len(delta)):
                    weights[perceptron_no+k][j+1] += learning_rate*delta[k]*input[j]
        else:
            break

def predict(weights,input,layer_perceptrons,perceptron_no):
    perceptron_output = []
    for i in range(0, layer_perceptrons[0]):
        perceptron_output.append(perceptron(weights, input, perceptron_no))
        perceptron_no += 1
    if (isLastValue(layer_perceptrons)):
        return perceptron_output
    else:
        return (predict(weights, perceptron_output, layer_perceptrons[1:], perceptron_no))

def softmax(output):
    sum = 0
    exp_array = []
    softmax_array = []
    for i in range(0,len(output)):
        temp = math.exp(output[i])
        exp_array.append(temp)
        sum += temp
    for i in range(0, len(exp_array)):
        exp_array[i] = exp_array[i]/sum
    maxIndex = exp_array.index(max(exp_array))
    for i in range(0,len(output)):
        if maxIndex == i :
            softmax_array.append(1)
        else:
            softmax_array.append(0)
    return softmax_array

def getAccuarcy(data,weights,layer_perceptrons,no_of_bits):
    positive = 0
    sum_squared_error = 0
    for i in range(0, len(data)):
        output = predict(weights, data[i][:-1], layer_perceptrons, 0)
        expected_output = modifyBinaryArray(int(data[i][-1:]), no_of_bits)
        softmax_output = softmax(output)
        flag = True
        for j in range(0, len(softmax_output)):
            if softmax_output[j] != expected_output[j]:
                flag = False
            sum_squared_error += squared_diff(expected_output[j], output[j])
        if flag:
            positive += 1
    squared_error = (1/(2*len(data)))*sum_squared_error
    return (squared_error,round(((positive / len(data)) * 100), 2))

def squared_diff(expected_value,computed_value):
    return (math.pow(expected_value,2)-math.pow(computed_value,2))