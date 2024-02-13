import pandas as pd

# carica il dataset accidents.csv
data = pd.read_csv('../old/accidents_ny.csv')

# aggiungi colonna booleana blocco stradale road_closure in maniera casuale

import random
"""
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
"""
#crea un csv dove per ogni mezzora genera lo stato, in maniera casuale, di ogni strada di new york
#crea un dataset con tutte le strade di new york
streets = data['Street'].unique()
time = data['Time'].unique()
#crea un dataset con tutte le combinazioni di strade e date
import itertools
import numpy as np
import random
import datetime

#per ogni mezzora e per ogni strada genera un valore casuale che corrsiponde allo stato della strada per 7 giorni
#per ogni strada e per ogni mezzora
#per ogni giorno della settimana
#per ogni mezzora
#per ogni strada
#genera un valore casuale che corrisponde allo stato della strada per 7 giorni rispetto ad una poissoin distribution

#crea un dataset con tutte le combinazioni di strade e date
streets = data['Street'].unique()
time = ["00:00","00:30","01:00","01:30","02:00","02:30","03:00","03:30","04:00"
    ,"04:30","05:00","05:30","06:00","06:30","07:00","07:30","08:00","08:30",
        "09:00","09:30","10:00","10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:30","16:00","16:30","17:00",
        "17:30","18:00","18:30","19:00","19:30","20:00","20:30","21:00","21:30","22:00","22:30","23:00","23:30"]
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

#crea un dataset con tutte le combinazioni di strade e date
import numpy as np
import random
import datetime

df = pd.DataFrame(columns=['Date', 'Time', 'Street', 'road_closure'])
for day in days:
    for street in streets:
        for t in time:
                #genera un valore casuale che corrisponde allo stato della strada per 7 giorni rispetto ad una distribution poissoniana
                #la probabilità  di un road_closure=true è maggiore durante i giorni lavorativi [Monday, Tuesday, Wednesday, Thursday, Friday]
                #la probabilità di un road_closure=true è maggiore durante le ore notturne

                #se il giorno è lavorativo e l'ora è notturna

                if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] and datetime.datetime.strptime(t, '%H:%M').time() > datetime.datetime.strptime('18:00', '%H:%M').time() and datetime.datetime.strptime(t, '%H:%M').time() < datetime.datetime.strptime('06:00', '%H:%M').time():
                    #probabilità di blocco stradale
                    p = 0.5
                else:
                    p = 0.1
                #genera un valore casuale che corrisponde allo stato della strada per 7 giorni rispetto ad una distribution poissoniana
                #il valore è 0 se non c'è blocco stradale, 1 se c'è blocco stradale
                road_closure = np.random.poisson(p)
                #se il valore è maggiore di 0, allora c'è blocco stradale
                road_closure = road_closure > 0
                #aggiungi il valore al dataset
                df = df._append({'Date': day, 'Time': t, 'Street': street, 'road_closure': road_closure}, ignore_index=True)

#salva il dataset modificato
df.to_csv('../street_status_one_week.csv', index=False)

#




















# salva il dataset modificato
data.to_csv('../accidents_ny.csv', index=False)





