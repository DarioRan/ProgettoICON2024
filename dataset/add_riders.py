import pandas as pd
import random

# Leggi il CSV in un DataFrame
dataset = pd.read_csv('food_order_updated.csv')


# Lista di nomi e cognomi
names = ["Noah", "Olivia", "Liam", "Emma", "Jackson", "Ava", "Lucas", "Sophia", "Aiden", "Isabella",
           "Caden", "Mia", "Grayson", "Amelia", "Mason", "Harper", "Elijah", "Evelyn", "Logan", "Abigail",
           "Oliver", "Ella", "Carter", "Scarlett", "Ethan", "Aria", "Landon", "Riley", "Luke", "Zoe"]

surnames = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
             "Hernandez", "Jackson", "Thompson", "White", "Harris", "Martin", "Hall", "Adams", "Allen", "King",
             "Scott", "Green", "Baker", "Lee", "Turner", "Taylor", "Moore", "Parker", "Collins", "Bennett"]


# Lista di tipi di veicoli disponibili
vehicles = ["bike", "car", "scooter", "motorcycle"]

# Inizializzazione del DataFrame
dataset['rider'] = None

# Definisci la funzione per generare i rider
def generate_riders(location_group):
    riders = []
    for _ in range(5):
        rider_id = random.randint(0, 100000)
        name = random.choice(names) + " " + random.choice(surnames)
        vehicle = random.choice(["bike", "car", "scooter", "motorcycle"])
        rider = {'id': rider_id, 'name': name, 'vehicle': vehicle}
        riders.append(rider)
    return riders

# Aggiorna il DataFrame con i rider
unique_locations = dataset['restaurant_location'].unique()
riders_by_location = {location: generate_riders(location) for location in unique_locations}

dataset['rider'] = dataset['restaurant_location'].apply(lambda location: str(random.choice(riders_by_location[location])))

dataset.to_csv('food_order_final.csv')


