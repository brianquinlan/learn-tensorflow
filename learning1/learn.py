import pandas
import numpy
from IPython import display
import tensorflow
from tensorflow.contrib import learn
import numpy

def get_features(dataframe):
    return dataframe[['a', 'b', 'c']]

def get_targets(dataframe):
    return dataframe[['d']]

data = pandas.read_csv(open('linear.csv'), sep=',').astype(numpy.float32)
randomized_data = data.reindex(numpy.random.permutation(data.index))

training_examples = get_features(randomized_data.head(900000))
training_targets = get_targets(randomized_data.head(900000))
validation_examples = get_features(randomized_data.head(100000))
validation_targets = get_targets(randomized_data.head(100000))

STEPS = 500
BATCH_SIZE = 5
periods = 200
steps_per_period = STEPS / periods

feature_columns = learn.infer_real_valued_columns_from_input(
    training_examples)

linear_regressor = learn.LinearRegressor(
    feature_columns=feature_columns)

for period in range(periods):
    linear_regressor.fit(training_examples,
                         training_targets,
                         steps=steps_per_period)
    e = linear_regressor.evaluate(validation_examples, validation_targets)
    print('Error:', e)

a = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=int) # 0, 12

e = linear_regressor.evaluate(validation_examples, validation_targets)
x = linear_regressor.predict(a)
print(x)
print(e)