import pandas as pd
import random
from geopy.distance import geodesic
import requests

file_path = '../old/food_order.csv'
# Leggi il CSV in un DataFrame
dataframe = pd.read_csv(file_path)

# Lista di ristoranti per ogni tipo di cucina
ristoranti_per_cucina = {
    'Spanish': ['Boqueria W40th', 'Salinas Restaurant', 'El Castillo De Madison'],
    'Korean': ['miss KOREA BBQ'],
    'Thai': ['Thai Villa', 'Sabai Thai'],
    'French': ['Le District', 'Le Coucou', 'Le Baratin', 'Gaby'],
    'Vietnamese': ['Saigon Vietnamese Sandwich Deli', 'Hanoi House', 'Madame Vo'],
    'Southern': ['Spaha Soul', 'Jacob Soul Food Restaurant', "Melba's Restaurant", 'SoCo']
}


# Genera dati casuali per le nuove righe
def generate_random_data():
    """
    Generate random data for the new rows

    :return: a dictionary with the random data

    """
    return {
        'order_id': random.randint(10000, 99999),
        'customer_id': random.randint(1000, 9999),
        'rating': round(random.uniform(3, 5)),
        'food_preparation_time': random.randint(20, 30),
        'delivery_time': random.randint(20, 30)
    }


# Aggiungi le nuove righe al dataframe esistente
def add_rows_to_dataframe(df, cuisine, restaurants):
    new_rows = []
    for restaurant in restaurants:
        new_data = generate_random_data()
        new_data.update({
            'restaurant_name': restaurant,
            'cuisine_type': cuisine,
            'cost_of_the_order': round(random.uniform(10, 50), 2),
            'day_of_the_week': random.choice(['Weekday', 'Weekend'])
        })
        new_rows.append(new_data)
    return df._append(new_rows, ignore_index=True)


# Aggiungi le nuove righe per ogni tipo di cucina
for cuisine, restaurants in ristoranti_per_cucina.items():
    dataframe = add_rows_to_dataframe(dataframe, cuisine, restaurants)

# Salva il dataframe aggiornato nel file CSV
dataframe.to_csv(file_path, index=False)


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
        return latitude, longitude  # Restituisce le coordinate come tupla
    else:
        print("Coordinate non disponibili per questo ristorante.")
        return None


# Applica la funzione a ogni riga del DataFrame
dataframe['restaurant_location'] = dataframe['restaurant_name'].apply(get_coordinates)

# Elimina le righe in cui le coordinate non sono disponibili
dataframe = dataframe.dropna(subset=['restaurant_location'])

def generate_random_row(restaurant_name, restaurant_location, cuisine, df):
    day_of_the_week = random.choice(['Weekday', 'Weekend'])
    delivery_time = random.randint(15, 30) if day_of_the_week == 'Weekday' else random.randint(20, 40)

    # Calcola la media dei food_preparation_time per lo stesso restaurant_location
    avg_preparation_time = df[df['restaurant_location'] == restaurant_location]['food_preparation_time'].mean()

    # Genera il food_preparation_time basato sulla media con un intervallo di +/- 7 minuti
    food_preparation_time = round(random.uniform(avg_preparation_time - 7, avg_preparation_time + 7))

    return {
        'order_id': random.randint(10000, 99999),
        'customer_id': random.randint(1000, 9999),
        'restaurant_name': restaurant_name,
        'restaurant_location': restaurant_location,
        'cuisine_type': cuisine,
        'cost_of_the_order': round(random.uniform(15, 50), 2),
        'day_of_the_week': day_of_the_week,
        'rating': round(random.uniform(3, 5)),
        'food_preparation_time': food_preparation_time,
        'delivery_time': delivery_time
    }


for cuisine in dataframe['cuisine_type'].unique():
    rows_count = len(dataframe[dataframe['cuisine_type'] == cuisine])
    while rows_count < 100:
        unique_restaurants = dataframe[dataframe['cuisine_type'] == cuisine].groupby(
            ['restaurant_name', 'restaurant_location']).size().reset_index(name='counts')
        min_row_restaurant = unique_restaurants.loc[unique_restaurants['counts'].idxmin()]

        restaurant_to_update = min_row_restaurant['restaurant_name']
        location_to_update = min_row_restaurant['restaurant_location']

        new_row = generate_random_row(restaurant_to_update, location_to_update, cuisine, dataframe)
        dataframe = pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)
        rows_count += 1  # Aggiorna il conteggio delle righe


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
        # Genera una velocità casuale tra 10 e 15 km/h (non si considera il traffico quindi si abbassa la velocità
        # media di almeno la metà)
        speed_kmh = random.uniform(10, 15)

        # Calcola la distanza percorsa in questo slot utilizzando la velocità
        distance_km = (speed_kmh / 60) * 5

        # Aggiungi la distanza al totale
        total_distance_km += distance_km

    # Genera un numero casuale tra 0 e 359 gradi per la direzione
    random_direction = random.uniform(0, 359)

    # Estrai le coordinate del ristorante
    restaurant_location_str = row['restaurant_location']
    restaurant_location_tuple = tuple(map(float, restaurant_location_str.strip('()').split(',')))

    # Calcola le nuove coordinate del cliente con la direzione casuale
    destination = geodesic(kilometers=total_distance_km).destination(point=restaurant_location_tuple,
                                                                     bearing=random_direction)

    return destination.latitude, destination.longitude


# Applica la funzione di calcolo della customer_location a ogni riga del DataFrame
dataframe['customer_location'] = dataframe.apply(calculate_customer_location, axis=1)

# Salva il DataFrame aggiornato in un nuovo file CSV
dataframe.to_csv('food_order_new.csv', index=False)
