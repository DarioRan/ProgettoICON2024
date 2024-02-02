import itertools
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

data = pd.read_csv('../dataset/accidents_ny.csv')

# Calcolo della densità degli incidenti per ogni combinazione unica delle variabili genitori
# Raggruppa per le variabili genitori e calcola la frequenza relativa degli incidenti
incident_density = data.groupby(['Visibility(km)', 'Wind_Speed(km/h)', 'Precipitation(mm)', 'Astronomical_Twilight']).size().reset_index(name='incident_count')

# Calcola il numero totale di incidenti
total_incidents = data.shape[0]

# Calcola la densità degli incidenti (probabilità)
incident_density['accident_probability'] = incident_density['incident_count'] / 100

incident_density.drop(columns=['incident_count'], inplace=True)
incident_density.to_csv('../dataset/incident_density.csv', index=False)

model = BayesianNetwork([
    ('Visibility(km)', 'accident_probability'),
    ('Wind_Speed(km/h)', 'accident_probability'),
    ('Precipitation(mm)', 'accident_probability'),
    ('Astronomical_Twilight', 'accident_probability')
])

model.fit(incident_density, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(model)

query_result = infer.query(variables=['accident_probability'], evidence={
    'Visibility(km)': 2.0,
    'Wind_Speed(km/h)': 3.0,
    'Precipitation(mm)': 0.06,
    'Astronomical_Twilight': 'Night'
})

print(query_result)
