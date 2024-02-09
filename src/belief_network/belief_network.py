# impara probabilità a posteriori da accidents_ny.csv
# data l'ora e la strada, calcola la probabilità di blocco stradale
#

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def get_road_closure_probability(time, street):
    #carica i dati
    data = pd.read_csv('../../dataset/accidents_ny.csv')

    #elimina le righe la cui strada compare meno di 10 volte
    data = data.groupby('Street').filter(lambda x: len(x) > 10)

    # se la strada non è presente nel dataset, ritorna errore
    if street not in data['Street'].values:
        return 'Street not found'

    #crea il modello
    model = BayesianNetwork([('Time', 'road_closure'), ('Street', 'road_closure')])
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    #crea l'oggetto per l'inferenza
    inference = VariableElimination(model)

    #calcola la probabilità di blocco stradale
    q = inference.query(variables=['road_closure'], evidence={'Time': time, 'Street': street})
    return q



#test
print(get_road_closure_probability('17:00:00', 'George Washington Bridge'))
