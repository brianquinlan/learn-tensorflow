#!/usr/bin/env python3

"""Classify the sum of two numbers as positive or not.

Run generatedata.py to generate the required data file. 
"""

import re

import numpy
import pandas
import pprint
from tensorflow.contrib import learn

STEPS = 10


def get_features(dataframe):
  return dataframe[['value1', 'value2']]


def get_targets(dataframe):
  return dataframe[['positive']]


data = pandas.read_csv(
    open('data.csv'), dtype={'value1': numpy.float32,
                             'value2': numpy.float32,
                             'positive': bool}, sep=',')
randomized_data = data.reindex(numpy.random.permutation(data.index))

training_examples = get_features(randomized_data.head(900000))
training_targets = get_targets(randomized_data.head(900000))
validation_examples = get_features(randomized_data.head(100000))
validation_targets = get_targets(randomized_data.head(100000))

feature_columns = learn.infer_real_valued_columns_from_input(training_examples)

linear_classifier = learn.LinearClassifier(feature_columns=feature_columns)

for step in range(STEPS):
    linear_classifier.fit(training_examples, training_targets, steps=1)
    e = linear_classifier.evaluate(validation_examples, validation_targets)
    print()
    print('Evaluation Results [step: %d]' % step)
    print('----------------------------')
    print()
    pprint.pprint(e)
    print()

while True:
    values = input('Enter two numbers: ')
    value1, value2 = [
      float(v) for v in re.findall('[-+]?[0-9]*\.?[0-9]+', values)
    ]
    prediction = linear_classifier.predict(
        numpy.array(
              [[value1, value2]], dtype=numpy.float32))
    if prediction[0]:
        print('%s + %s is positive' % (value1, value2))
    else:
        print('%s + %s is not positive' % (value1, value2))
