import pandas
import numpy
from IPython import display
import tensorflow
from tensorflow.contrib import learn
import numpy
import pprint

def choose_features_from_correlations(correlations, target_name, num_features):
    sorted_by_correlation = [name for (name, value) in sorted(
        correlations[target_name].items(),
        key=lambda l: abs(l[1]))]
    sorted_by_correlation.remove(target_name)

    selected_features = []
    for _ in range(num_features):
        selected_feature = sorted_by_correlation.pop()
        selected_features.append(selected_feature)
        for name, correlation in correlations[selected_feature].items():
            if abs(correlation) > 0.8 and name in sorted_by_correlation:
                print('Removing: {0} (too similar to {1}: {2})'.format(
                    name, selected_feature, correlation))
                sorted_by_correlation.remove(name)

    return selected_features

columns = set(open('communities.data').readline().strip().split(','))
columns.discard('communityname')

data = pandas.read_csv(
    open('communities.data'),
    sep=',',
    na_values="?",
    usecols=list(columns),
    dtype=numpy.float32)
randomized_data = data.reindex(numpy.random.permutation(data.index))
correlations = data.corr()

features = choose_features_from_correlations(
    correlations, 'ViolentCrimesPerPop', 5)

training_examples = randomized_data.head(1596-399)[features]
training_targets = randomized_data.head(1596-399)[['ViolentCrimesPerPop']]
validation_examples = randomized_data.tail(399)[features]
validation_targets = randomized_data.tail(399)[['ViolentCrimesPerPop']]

STEPS = 5000
periods = 100
steps_per_period = STEPS / periods

feature_columns = learn.infer_real_valued_columns_from_input(
    training_examples)

pprint.pprint(training_examples)
linear_regressor = learn.LinearRegressor(
    feature_columns=feature_columns)

for period in range(periods):
    linear_regressor.fit(training_examples,
                         training_targets,
                         steps=steps_per_period)
    e = linear_regressor.evaluate(validation_examples, validation_targets)
    print('Error:', e)

a = numpy.array([[0.83, 0.86, 0.03, 0.67, 0.12],  # Pleasanton
                 [0.23, 0.0, 0.86, 0.36, 0.88]],  # Oakland
                 dtype=numpy.float32)

e = linear_regressor.evaluate(validation_examples, validation_targets)
x = linear_regressor.predict(a)
print(x)
print(e)

