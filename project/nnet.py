import numpy
from keras.layers import Dense
from keras.models import Sequential
import csv

"""
Params that can change:

loss function
    number of hidden layers
    number of units in each layer
    activation functions in each layer

different activation functions within each layer?
    don't think I'll get to this
    
optimizer to use - nesterov/adam
number of epochs to run
batch size in each epoch
        
Observations:

Anything more than 300 epochs with a batch size of 10 doesn't change the
error.

Loss: 1.1229e-07, 0.68 accuracy on test set
300, relu
200, relu
100, relu
50, relu
1, sigmoid

0.626
Loss: 1.1244e-07
300, relu
300, relu
300, relu
300, relu
1, sigmoid

0.683
Loss: 1.1229e-07
300, relu
150, relu
75, relu
37, relu
1, sigmoid

0.675
Loss:  1.1975e-07
100, relu
50, relu
25, relu
12, relu
1, sigmoid

0.669
Loss: 2.9767e-07
50, relu
25, relu
12, relu
6, relu
1, sigmoid

 
"""

numpy.random.seed(0)

train_data = numpy.loadtxt("train.csv", delimiter=",", skiprows=1)
test_data = numpy.loadtxt("test.csv", delimiter=",", skiprows=1)

num_features = len(train_data[0]) - 2

train_instances = train_data[:, 2:num_features + 2]
test_instances = test_data[:, 1:num_features + 1]
train_labels = train_data[:, 1]

neural_net_model = Sequential()

# neural_net_model.add(Dense(100, input_dim=num_features, init='uniform', activation='relu'))
# neural_net_model.add(Dense(50, init='uniform', activation='relu'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(300, input_dim=num_features, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(200, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(100, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(50, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(300, input_dim=num_features, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(300, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(300, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(300, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(300, input_dim=num_features, init='uniform', activation='relu'))
# neural_net_model.add(Dense(150, init='uniform', activation='relu'))
# neural_net_model.add(Dense(75, init='uniform', activation='relu'))
# neural_net_model.add(Dense(37, init='uniform', activation='relu'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(100, input_dim=num_features, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(50, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(25, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(12, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(50, input_dim=num_features, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(25, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(12, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(6, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(150, input_dim=num_features, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(75, input_dim=num_features, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(40, input_dim=num_features, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(20, input_dim=num_features, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(10, input_dim=num_features, init='uniform', activation='sigmoid'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(150, input_dim=num_features, init='uniform', activation='relu'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(75, input_dim=num_features, init='uniform', activation='relu'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(40, input_dim=num_features, init='uniform', activation='relu'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))
#
neural_net_model.add(Dense(20, input_dim=num_features, init='uniform', activation='relu'))
neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))

# neural_net_model.add(Dense(10, input_dim=num_features, init='uniform', activation='relu'))
# neural_net_model.add(Dense(1, init='uniform', activation='sigmoid'))


neural_net_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
neural_net_model.fit(train_instances, train_labels, epochs=2, batch_size=1, verbose=2)

predictions = neural_net_model.predict(test_instances)

rounded = [int(round(x[0])) for x in predictions]

rows = []
start_id = 250
for prediction in rounded:
    rows.append([start_id, prediction])
    start_id += 1

with open("submission.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "target"])
    writer.writerows(rows)
