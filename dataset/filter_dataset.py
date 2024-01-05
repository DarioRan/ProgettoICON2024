import pandas as pd

# Leggi il CSV in un DataFrame
dataframe = pd.read_csv('food_order_new.csv')

numero_righe = dataframe.shape[0]
print(f"Il CSV new ha {numero_righe} righe.")

# Definisci i limiti geografici più ampi per New York City
latitudine_minima = 40.4
latitudine_massima = 40.9
longitudine_minima = -74.5
longitudine_massima = -73.7

# Converti la colonna restaurant_location da stringa a tupla di float
dataframe['restaurant_location'] = dataframe['restaurant_location'].apply(eval)

# Filtra il DataFrame per coordinate all'interno dei limiti di New York City
dataframe_nyc = dataframe[dataframe['restaurant_location'].apply(lambda x: latitudine_minima <= x[0] <= latitudine_massima and longitudine_minima <= x[1] <= longitudine_massima)]

numero_righe = dataframe_nyc.shape[0]

# Stampa il numero di righe
print(f"Il CSV filtrato ha {numero_righe} righe.")

# Salva il DataFrame filtrato in un nuovo file CSV
dataframe_nyc.to_csv('food_order_nyc.csv', index=False)
