# impara probabilità a posteriori da accidents_ny.csv
# data l'ora e la strada, calcola la probabilità di blocco stradale
#

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


class BeliefNetwork:

    def __init__(self, df):
        self.model = None
        self.data = df
        # elimina le righe la cui strada compare meno di 10 volte
        self.data = self.data.groupby('Street').filter(lambda x: len(x) > 10)


    def train_model(self):
        # crea il modello
        self.model = BayesianNetwork([('Time', 'road_closure'), ('Street', 'road_closure')])
        self.model.fit(self.data, estimator=MaximumLikelihoodEstimator)

    def get_road_closure_probability(self,time, street):

        # se la strada non è presente nel dataset, ritorna errore
        if street not in self.data['Street'].values:
            return 'Street not found'
        #crea l'oggetto per l'inferenza
        inference = VariableElimination(self.model)

        #calcola la probabilità di blocco stradale
        q = inference.query(variables=['road_closure'], evidence={'Time': time, 'Street': street})
        return q







