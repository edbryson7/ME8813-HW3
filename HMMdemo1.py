from pomegranate import *
import numpy as np
from matplotlib import pyplot

# emission/observation probabilities
d1 = DiscreteDistribution({0: 0.95, 1: 0.05})
d2 = DiscreteDistribution({0: 0.05, 1: 0.95})

# define states
s1 = State(d1, name='idle')
s2 = State(d2, name='work')

# create model and add states
model = HiddenMarkovModel('machine')
model.add_states(s1, s2)

# specify initial probabilities
model.add_transition(model.start, s1, 0.5)
model.add_transition(model.start, s2, 0.5)

# specify transition probabilities
model.add_transition(s1, s1, 0.7)
model.add_transition(s1, s2, 0.3)
model.add_transition(s2, s1, 0.3)
model.add_transition(s2, s2, 0.7)

model.add_transition(s1, model.end, 0.1)

model.bake()

## the orginal model
print("=========original model========== ")
print(model)

## plot the model with networkx
model.plot()
pyplot.show()

## fit/train the parameters of HMM with a sequence of observations
datasequence1 = [0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1, \
            1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0]
datasequence2 = [0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1, \
            1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0]
model.fit( [datasequence1, datasequence2], \
            max_iterations=5, transition_pseudocount=2, use_pseudocount=True )

## updated model after training
print("==========updated model after training==========")
print(model)

## generate some sample emissions/observations
numSeq = 10
data = model.sample(numSeq)
print("==========generated samples==========")
print(data)
