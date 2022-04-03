#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pomegranate import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def main():
    companies = ['.//HistoricalData_AMD.csv', './/HistoricalData_APPLE.csv', './/HistoricalData_CISCO.csv', './/HistoricalData_MICROSOFT.csv', './/HistoricalData_QUALCOMM.csv']

    model = initModel()
    # showModel(model)

    for comp in companies[:3]:
        print(f'Fitting with {comp[18:-4]}')
        data = getStockData(comp)
        # model.fit(data, transition_pseudocount=5, use_pseudocount=True)
        model.fit(data, max_iterations=200, transition_pseudocount=5, use_pseudocount=True)

        # print(model)
    showModel(model)
    return

def getStockData(fpath):
    df = pd.read_csv(fpath, usecols = ['Close/Last']).replace('[\$,]', '', regex=True).astype(float)
    dfDiff = df.diff()/df
    return dfDiff.iloc[200:1200].to_numpy()
    # return dfDiff.to_numpy()


def showModel(model):
    # print("=========original model========== ")
    # print(model)

    ## plot the model with network
    model.plot()
    plt.show()


def initModel():
    # emission/observation probabilities
    # Given that the state is A, the expression is .01 with a variance of .0001
    dA = NormalDistribution(.01,.0001)
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
    model.add_transition(s1, s1, 0.6)
    model.add_transition(s1, s2, 0.1)
    model.add_transition(s1, s3, 0.1)
    model.add_transition(s1, s4, 0.1)
    model.add_transition(s1, s5, 0.1)

    # From State S
    model.add_transition(s2, s1, 0.1)
    model.add_transition(s2, s2, 0.6)
    model.add_transition(s2, s3, 0.1)
    model.add_transition(s2, s4, 0.1)
    model.add_transition(s2, s5, 0.1)

    # From State B
    model.add_transition(s3, s1, 0.1)
    model.add_transition(s3, s2, 0.1)
    model.add_transition(s3, s3, 0.6)
    model.add_transition(s3, s4, 0.1)
    model.add_transition(s3, s5, 0.1)

    # From State M
    model.add_transition(s4, s1, 0.1)
    model.add_transition(s4, s2, 0.1)
    model.add_transition(s4, s3, 0.1)
    model.add_transition(s4, s4, 0.6)
    model.add_transition(s4, s5, 0.1)

    # From State D
    model.add_transition(s5, s1, 0.1)
    model.add_transition(s5, s2, 0.1)
    model.add_transition(s5, s3, 0.1)
    model.add_transition(s5, s4, 0.1)
    model.add_transition(s5, s5, 0.6)

    # Not sure what the effect of this is
    # model.add_transition(s1, model.end, 0.00001)
    # model.add_transition(s2, model.end, 0.00001)
    # model.add_transition(s3, model.end, 0.00001)
    # model.add_transition(s4, model.end, 0.00001)
    # model.add_transition(s5, model.end, 0.00001)

    model.bake()
    return model

if __name__ == "__main__":
    main()
