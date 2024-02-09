import pandas as pd

# carica il dataset accidents.csv
data = pd.read_csv('../old/accidents_ny.csv')

# aggiungi colonna booleana blocco stradale road_closure in maniera casuale

import random

#la road closure è in funzione dell'ora, più è tardi, maggiore è la probabilità di blocco stradale
#per questo motivo, si può usare la stessa probabilità di blocco stradale per tutte le strade

data['road_closure'] = data['Time'].apply(lambda x: random.choice([True, False]) if x > '12:00:00' else random.choice([
    True
    for i in range(0, 20)
] + [False
    for i in range(0, 80)
]))

#dividi Time in date e time, il campo Time è in formato 2016-12-01 08:21:11


data['Date'] = pd.to_datetime(data['Time'],format='mixed').dt.date
data['Time'] = pd.to_datetime(data['Time'],format='mixed').dt.time

#round time to the nearest half hour and remove seconds
data['Time'] = data['Time'].apply(lambda x: x.replace(second=0))
data['Time'] = data['Time'].apply(lambda x: x.replace(minute=30) if x.minute >= 30 else x.replace(minute=0))







# salva il dataset modificato
data.to_csv('../accidents_ny.csv', index=False)





