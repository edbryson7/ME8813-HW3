#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pomegranate import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def main():
    companies = ['.//HistoricalData_CISCO.csv', './/HistoricalData_AMD.csv', './/HistoricalData_QUALCOMM.csv', './/HistoricalData_APPLE.csv', './/HistoricalData_MICROSOFT.csv']

    model = initModel()
    # showModel(model)

    # model.freeze_distributions()
    model = fit(model, companies[:3])
    showModel(model)

def fit(model, companies):
    data = [getStockData(comp) for comp in companies]

    print(f'Fitting with {[comp[18:-4] for comp in companies]}')

    model.fit(data, transition_pseudocount=5, use_pseudocount=True)

    return model

def getStockData(fpath):
    df = pd.read_csv(fpath, usecols = ['Close/Last']).replace('[\$,]', '', regex=True).astype(float)
    dfDiff = df.diff()/df

    # return dfDiff.iloc[200:1200].to_numpy()

    data = dfDiff.iloc[1:].to_numpy()
    return data[::-1]

def showModel(model):
    for state in model.states[:-2]:
        print('State {:s}: Epression: {:.5f}'.format(state.name, state.distribution.parameters[0]))

    # plot the model with network
    model.plot()
    plt.show()

def initModel():
    # emission/observation probabilities
    # Given that the state is A, the expression is .01 with a variance of .0001

    dA = NormalDistribution(.001,.0001)
    dS = NormalDistribution(-.001,.0004)
    dB = NormalDistribution(.001,.0001)
    dM = NormalDistribution(.005,.0001)
    dD = NormalDistribution(-.001,.0004)

    # define states
    sA = State(dA, name='Adoption')
    sS = State(dS, name='Stagnant')
    sB = State(dB, name='Breakthrough')
    sM = State(dM, name='Mature')
    sD = State(dD, name='Decline')

    # create model and add states
    model = HiddenMarkovModel('machine')
    model.add_states(sA, sS, sB, sM, sD)

    # specify initial probabilities
    model.add_transition(model.start, sA, 1)
    model.add_transition(model.start, sS, 1)
    model.add_transition(model.start, sB, 1)
    model.add_transition(model.start, sM, 1)
    model.add_transition(model.start, sD, 1)

    # specify transition probabilities

    # From State A
    model.add_transition(sA, sA, 0.9)
    model.add_transition(sA, sS, 0.1)
    model.add_transition(sA, sB, 0.1)
    model.add_transition(sA, sM, 0.1)

    # From State S
    model.add_transition(sS, sS, 0.9)
    model.add_transition(sS, sB, 0.1)
    model.add_transition(sS, sD, 0.1)

    # From State B
    model.add_transition(sB, sA, 0.1)
    model.add_transition(sB, sS, 0.1)
    model.add_transition(sB, sB, 0.9)
    model.add_transition(sB, sM, 0.1)

    # From State M
    model.add_transition(sM, sA, 0.1)
    model.add_transition(sM, sS, 0.1)
    model.add_transition(sM, sB, 0.1)
    model.add_transition(sM, sM, 0.9)

    # From State D
    model.add_transition(sD, sS, 0.1)
    model.add_transition(sD, sD, 0.9)

    # Not sure what the effect of this is
    # model.add_transition(sA, model.end, 0.1)
    # model.add_transition(sS, model.end, 0.1)
    # model.add_transition(sB, model.end, 0.1)
    # model.add_transition(sM, model.end, 0.1)
    # model.add_transition(sD, model.end, 0.1)

    model.bake()
    return model

if __name__ == "__main__":
    main()
