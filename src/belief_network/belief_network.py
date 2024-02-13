# impara probabilità a posteriori da accidents_ny.csv
# data l'ora e la strada, calcola la probabilità di blocco stradale
#

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from src.find_path.utils import find_street_names
from fuzzywuzzy import fuzz



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

    """
    def get_road_closure_probability(self,x_time, street):
        # se la strada non è presente nel dataset, ritorna errore
        if street not in self.data['Street'].values:
            return 'Street not found'
        #crea l'oggetto per l'inferenza
        inference = VariableElimination(self.model)

        #calcola la probabilità di blocco stradale
        print(x_time, street)
        print(self.data.columns)
        q = inference.query(variables=['road_closure'], evidence={'Time': x_time, 'Street': street})
        return q
    """
    def get_road_closure_probability(self, x_time, street):
        x_time = str(x_time)
        # Controllo della similarità del 90% con i nomi delle strade nel dataset
        similar_streets = [street_name for street_name in self.data['Street'].unique() if fuzz.ratio(street, street_name) >= 85]
        if len(similar_streets) == 0 or similar_streets is None:
            print('Street not found')
            return 0

        try:
            inference = VariableElimination(self.model)

            q = inference.query(variables=['road_closure'], evidence={'Time': x_time, 'Street': similar_streets[0]})
            q.normalize(inplace=True)
            return q.values[1]
        except KeyError:
            return 0

    def predict_road_closure_probability(self, G, x_time, path):
        street_names = find_street_names(G, path)
        max_prob = 0
        for i in range(len(street_names) - 1):
            prob = self.get_road_closure_probability(str(x_time), street_names[i])
            print(prob)
            if prob > max_prob:
                max_prob = prob

        return max_prob


# test class

if __name__ == '__main__':

    # read data
    data = pd.read_csv('../../dataset/street_status_one_week.csv')
    # create belief network
    BN = BeliefNetwork(df=data)
    # train model
    BN.train_model()
    # predict road closure probability
    print(BN.get_road_closure_probability('10:00', 'street1'))
    print(BN.get_road_closure_probability('10:00', 'street2'))
    print(BN.get_road_closure_probability('10:00', 'street3'))
    print(BN.get_road_closure_probability('10:00', 'street4'))
    print(BN.get_road_closure_probability('10:00', 'street5'))






