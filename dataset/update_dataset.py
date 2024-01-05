import pandas as pd
import random
from geopy.distance import geodesic
import requests

# Leggi il CSV in un DataFrame
dataframe = pd.read_csv('food_order.csv')

# Funzione per ottenere le coordinate geografiche come tupla
def get_coordinates(restaurant_name):
    query = f'{restaurant_name} New York'
    url = 'https://nominatim.openstreetmap.org/search'
    params = {
        'q': query,
        'format': 'json',
    }
    print(f"Elaborazione del ristorante: {restaurant_name}")
    response = requests.get(url, params=params)
    data = response.json()
    if data:
        location = data[0]
        latitude = float(location['lat'])
        longitude = float(location['lon'])
        print(f"Coordinate ottenute: Latitudine={latitude}, Longitudine={longitude}")
        return (latitude, longitude)  # Restituisce le coordinate come tupla
    else:
        print("Coordinate non disponibili per questo ristorante.")
        return None

# Applica la funzione a ogni riga del DataFrame
dataframe['restaurant_location'] = dataframe['restaurant_name'].apply(get_coordinates)

# Elimina le righe in cui le coordinate non sono disponibili
dataframe = dataframe.dropna(subset=['restaurant_location'])

# Funzione per calcolare la customer_location
def calculate_customer_location(row):
    # Tempo totale di consegna in minuti
    total_delivery_time = row['delivery_time']

    # Numero di slot da 5 minuti
    num_slots = total_delivery_time // 5

    # Inizializza la distanza totale e la direzione a zero
    total_distance_km = 0

    # Loop attraverso ogni slot
    for slot in range(num_slots):
        # Genera una velocità casuale tra 10 e 15 km/h (non si considera il traffico quindi si abbassa la velocità media di almeno la metà)
        speed_kmh = random.uniform(10, 15)

        # Calcola la distanza percorsa in questo slot utilizzando la velocità
        distance_km = (speed_kmh / 60) * 5

        # Aggiungi la distanza al totale
        total_distance_km += distance_km

    # Genera un numero casuale tra 0 e 359 gradi per la direzione
    random_direction = random.uniform(0, 359)

    # Estrai le coordinate del ristorante
    restaurant_location = row['restaurant_location']

    # Calcola le nuove coordinate del cliente con la direzione casuale
    destination = geodesic(kilometers=total_distance_km).destination(point=restaurant_location, bearing=random_direction)

    return (destination.latitude, destination.longitude)

# Applica la funzione di calcolo della customer_location a ogni riga del DataFrame
dataframe['customer_location'] = dataframe.apply(calculate_customer_location, axis=1)

# Salva il DataFrame aggiornato in un nuovo file CSV
dataframe.to_csv('food_order_new_1.csv', index=False)
