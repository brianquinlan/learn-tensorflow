import pandas
import numpy
from IPython import display
import tensorflow
from tensorflow.contrib import learn
# from tensorflow import train
import numpy

data = pandas.read_csv(open('smallnumbers.csv'), sep=',').astype(numpy.float32)
randomized_data = data.reindex(numpy.random.permutation(data.index))
training_examples = randomized_data.head(90000)['a']
training_targets = randomized_data.head(90000)['b']

validation_examples = randomized_data.tail(10000)['a']
validation_targets = randomized_data.tail(10000)['b']

display.display(training_examples.describe())
display.display(training_targets.describe())

display.display(validation_examples.describe())
display.display(validation_targets.describe())

LEARNING_RATE = 0.01
STEPS = 500
BATCH_SIZE = 5
periods = 200
steps_per_period = STEPS / periods

feature_columns = learn.infer_real_valued_columns_from_input(
    training_examples)
print('feature_columns:', feature_columns)
linear_regressor = learn.LinearRegressor(
    feature_columns=feature_columns)

for period in range(periods):
    linear_regressor.fit(training_examples, training_targets,
        steps=steps_per_period, batch_size=BATCH_SIZE)
    e = linear_regressor.evaluate(validation_examples, validation_targets)
    print('Error:', e)

a = numpy.array([[-1000], [-5], [-3], [0], [1], [2], [500], [100000], [1000000]], dtype=int)

e = linear_regressor.evaluate(validation_examples, validation_targets)
x = linear_regressor.predict(a)
print(x)
print(e)