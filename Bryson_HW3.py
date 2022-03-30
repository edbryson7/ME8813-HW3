from pomegranate import *
import numpy as np
from matplotlib import pyplot

# emission/observation probabilities

dA = NormalDistribution(.01,.0001)
# Given that the state is A, the expression is .01 with a variance of .0001
dS = NormalDistribution(-.01,.0004)
dB = NormalDistribution(.01,.0001)
dM = NormalDistribution(.005,.0001)
dD = NormalDistribution(-.01,.0004)

# define states
s1 = State(dA, name='A')
s2 = State(dS, name='S')
s3 = State(dB, name='B')
s4 = State(dM, name='M')
s5 = State(dD, name='D')
sR = State(dD, name='R')

# create model and add states
model = HiddenMarkovModel('machine')
model.add_states(s1, s2, s3, s4, s5)

# specify initial probabilities
model.add_transition(model.start, s1, 0.2)
model.add_transition(model.start, s2, 0.2)
model.add_transition(model.start, s3, 0.2)
model.add_transition(model.start, s4, 0.2)
model.add_transition(model.start, s5, 0.2)

# specify transition probabilities

# From State A
model.add_transition(s1, s1, 0.2)
model.add_transition(s1, s2, 0.2)
model.add_transition(s1, s3, 0.2)
model.add_transition(s1, s4, 0.2)
model.add_transition(s1, s5, 0.2)

# From State S
model.add_transition(s2, s1, 0.2)
model.add_transition(s2, s2, 0.2)
model.add_transition(s2, s3, 0.2)
model.add_transition(s2, s4, 0.2)
model.add_transition(s2, s5, 0.2)

# From State B
model.add_transition(s3, s1, 0.2)
model.add_transition(s3, s2, 0.2)
model.add_transition(s3, s3, 0.2)
model.add_transition(s3, s4, 0.2)
model.add_transition(s3, s5, 0.2)

# From State M
model.add_transition(s4, s1, 0.2)
model.add_transition(s4, s2, 0.2)
model.add_transition(s4, s3, 0.2)
model.add_transition(s4, s3, 0.2)
model.add_transition(s4, s5, 0.2)

# From State D
model.add_transition(s5, s1, 0.2)
model.add_transition(s5, s2, 0.2)
model.add_transition(s5, s3, 0.2)
model.add_transition(s5, s4, 0.2)
model.add_transition(s5, s5, 0.2)
model.bake()

model.add_transition(s5, model.end, 0.1)

## the orginal model
print("=========original model========== ")
print(model)

## plot the model with networkx
model.plot()
pyplot.show()
exit()

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
