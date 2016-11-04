import pandas
import numpy
from IPython import display
import tensorflow
from tensorflow.contrib import learn, layers
import numpy
import pprint


CATEGORICAL_FEATURES = ['sex']
REAL_VALUED_FEATURES = ['length', 'diameter', 'height', 'wholeweight', 'shuckedweight', 'visceraweight', 'shellweight']
FEATURES = CATEGORICAL_FEATURES + REAL_VALUED_FEATURES
TARGET = 'rings'

data = pandas.read_csv(
    open('abalone.data'),
    header=None,
    names=FEATURES + [TARGET],
    dtype={'sex': str, 'length': numpy.float32, 'diameter': numpy.float32,
    'height': numpy.float32, 'wholeweight': numpy.float32, 'shuckedweight':numpy.float32, 'visceraweight':numpy.float32, 'shellweight' :numpy.float32, 'rings': int})

training_size = int(len(data) * 0.8)
verification_size = len(data) - training_size

randomized_data = data.sample(frac=1)
training_examples = randomized_data.head(training_size)[FEATURES]
training_targets = randomized_data.head(training_size)[[TARGET]]
validation_examples = randomized_data.tail(verification_size)[FEATURES]
validation_targets = randomized_data.tail(verification_size)[[TARGET]]


STEPS = 5000
BATCH_SIZE = 5
periods = 1

feature_columns = [
    layers.sparse_column_with_keys(
        column_name="sex", keys=["M", "F", "I"])] + (
    [layers.real_valued_column(name) for name in REAL_VALUED_FEATURES])


linear_regressor = learn.LinearRegressor(
    optimizer=tensorflow.train.GradientDescentOptimizer(0.05),
    feature_columns=feature_columns)

def input_fn(features, target=None):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tensorflow.constant(features[k].values) for k in REAL_VALUED_FEATURES}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tensorflow.SparseTensor(
    indices=[[i, 0] for i in range(features[k].size)],
    values=features[k].values,
    shape=[features[k].size, 1])
                      for k in CATEGORICAL_FEATURES}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    if target is not None:
        label = tensorflow.constant(target.values)
          # Returns the feature columns and the label.
        return feature_cols, label
    else:
        return feature_cols

for period in range(periods):
    linear_regressor.fit(input_fn=lambda : input_fn(training_examples, training_targets), steps=STEPS)
    e = linear_regressor.evaluate(input_fn=lambda : input_fn(validation_examples, validation_targets), steps=1)

print('Print final error:', e)
x = linear_regressor.predict(input_fn=lambda : input_fn(validation_examples))

print('Correlations to rings in verification data:\n', randomized_data.tail(verification_size).corr()[TARGET])

results = pandas.DataFrame.from_dict({'actual': validation_targets[TARGET].values, 'estimate': list(x)},
    dtype=numpy.float32)
print('Correlation between predicted values and actual values in verification data:\n',results.corr())


"""
Print final error: {'global_step': 250000, 'loss': 5.2495279}
Correlations to rings in verification data:
 length           0.545720
diameter         0.572259
height           0.593282
wholeweight      0.548274
shuckedweight    0.425626
visceraweight    0.514003
shellweight      0.638602
rings            1.000000
Name: rings, dtype: float64
Correlation between predicted values and actual values in verification data:
             actual  estimate
actual    1.000000  0.734059
estimate  0.734059  1.000000


    Sex     nominal         M, F, and I (infant)
    Length      continuous  mm  Longest shell measurement
    Diameter    continuous  mm  perpendicular to length
    Height      continuous  mm  with meat in shell
    Whole weight    continuous  grams   whole abalone
    Shucked weight  continuous  grams   weight of meat
    Viscera weight  continuous  grams   gut weight (after bleeding)
    Shell weight    continuous  grams   after being dried
    Rings       integer         +1.5 gives the age in years
"""