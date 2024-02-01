import itertools
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

data = pd.read_csv('../dataset/accidents_nyc.csv')

# Calcolo della densità degli incidenti per ogni combinazione unica delle variabili genitori
# Raggruppa per le variabili genitori e calcola la frequenza relativa degli incidenti
incident_density = data.groupby(['Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Astronomical_Twilight']).size().reset_index(name='incident_count')

# Calcola il numero totale di incidenti
total_incidents = data.shape[0]

# Calcola la densità degli incidenti (probabilità)
incident_density['accident_probability'] = incident_density['incident_count'] / 100

incident_density.drop(columns=['incident_count'], inplace=True)
incident_density.to_csv('../dataset/incident_density.csv', index=False)

model = BayesianNetwork([
    ('Visibility(mi)', 'accident_probability'),
    ('Wind_Speed(mph)', 'accident_probability'),
    ('Precipitation(in)', 'accident_probability'),
    ('Weather_Condition', 'accident_probability'),
    ('Astronomical_Twilight', 'accident_probability')
])

model.fit(incident_density, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(model)

query_result = infer.query(variables=['accident_probability'], evidence={
    'Visibility(mi)': 2.0,
    'Wind_Speed(mph)': 3.0,
    'Precipitation(in)': 0.06,
    'Weather_Condition': 'Light Rain',
    'Astronomical_Twilight': 'Night'
})

print(query_result)
