import pandas as pd
import random
from collections import defaultdict

# Carica il dataset dal tuo file CSV
dataset = pd.read_csv('food_order_nyc.csv')

# Calcola la media e la deviazione standard per ciascun ristorante, giorno della settimana e posizione
preparation_time_distributions = dataset.groupby(['restaurant_name', 'day_of_the_week', 'restaurant_location'])[
    'food_preparation_time'].agg(['mean', 'std']).reset_index()

# Converti il DataFrame risultante in un dizionario di dizionari
preparation_time_distributions_dict = preparation_time_distributions.groupby(
    ['restaurant_name', 'day_of_the_week', 'restaurant_location']).apply(
    lambda x: x[['mean', 'std']].to_dict(orient='index')).to_dict()


menu_by_cuisine = {
    'Korean': {
        'Bibimbap': 15, 'Kimchi': 12, 'Bulgogi': 18, 'Japchae': 20, 'Samgyeopsal': 17,
        'Doenjang Jjigae': 13, 'Tteokbokki': 16, 'Gimbap': 14, 'Sundubu Jjigae': 19, 'Galbi': 15
    },
    'Mexican': {
        'Carnitas': 16, 'Tacos': 14, 'Enchiladas': 18, 'Chiles Rellenos': 20, 'Tamales': 15,
        'Pozole': 17, 'Chiles en Nogada': 19, 'Guacamole': 12, 'Quesadillas': 13, 'Salsa Verde': 14
    },
    'American': {
        'Hot Dogs': 12, 'French Fries': 15, 'Chicken Tenders': 18, 'Pizza': 20, 'Burgers': 16,
        'Buffalo Chicken Wings': 14, 'Tater Tots': 13, 'Apple Pie': 15, 'Barbecue Ribs': 19, 'Reuben Sandwich': 17
    },
    'Indian': {
        'Biryani': 17, 'Butter Chicken': 15, 'Chaat': 14, 'Korma': 18, 'Lamb Saag': 16,
        'Dal': 13, 'Dosa': 20, 'Samosa': 12, 'Tandoori Chicken': 19, 'Rogan Josh': 16
    },
    'Italian': {
        'Pasta Carbonara': 14, 'Pizza': 20, 'Spaghetti con Salsa di Pomodoro': 16,
        'Pasta alla Gricia': 15, 'Rigatoni all\'Amatriciana': 18, 'Ossobuco': 19, 'Arancini': 13,
        'Ragù Bolognese': 17, 'Risotto': 18, 'Lasagna': 20
    },
    'Mediterranean': {
        'Greek Salad': 13, 'Armenian Losh Kebab': 18, 'Mediterranean White Bean Soup': 15,
        'Garlicky Sautéed Shrimp with Creamy White Beans and Blistered Tomatoes': 19,
        'Mediterranean Falafel Bowls': 14, 'Mediterranean Diet Chicken': 16,
        'Mediterranean Baked Cod Recipe': 17, 'Mediterranean Grilled Chicken + Dill Greek Yogurt Sauce': 20,
        'Mediterranean Pasta Salad': 16, 'Mediterranean Quinoa Salad': 15
    },
    'Chinese': {
        'Peking Roasted Duck': 20, 'Kung Pao Chicken': 17, 'Sweet and Sour Pork': 14,
        'Hot Pot': 19, 'Dumplings': 16, 'Chow Mein': 15, 'Yangzhou Fried Rice': 18,
        'Fish-Flavored Shredded Pork': 13, 'Sweet and Sour Pork Fillet': 12, 'Congee': 20
    },
    'Japanese': {
        'Sushi': 18, 'Tempura': 15, 'Yakitori': 13, 'Tsukemono (sottaceti)': 16,
        'Yakisoba (noodles saltati)': 17, 'Kaiseki': 20, 'Ichiju-ju': 14, 'Sashimi': 19,
        'Nigiri': 12, 'Miso Soup': 15
    },
    'Middle Eastern': {
        'Jewelled rice': 19, 'Cauliflower and chickpea tagine': 17,
        'Beef Dolmas with Apricots and Tamarind': 14, 'Cauliflower Shawarma Berber': 16,
        'Roast Chicken with Sumac Flatbread (M’sakhan)': 20, 'Persian Kuku Sabzi': 15,
        'Hummus': 13, 'Baba Ghannouj': 18, 'Falafel': 14, 'Shawarma': 17
    },
    'Thai': {
        'Som Tum': 14, 'Pad Thai': 19, 'Tom Yum Goong': 15, 'Khao Pad': 16, 'Guay Tiew Reua': 17,
        'Tom Kha Kai': 18, 'Massaman Curry': 20, 'Green Curry': 13, 'Thai Fried Rice': 12, 'Nam Tok Mu': 14
    },
    'Spanish': {
        'Paella': 20, 'Tortilla de Patatas': 18, 'Patatas Bravas': 17, 'Gazpacho': 15,
        'Churros': 14, 'Jamón Serrano': 16, 'Pisto': 19, 'Fabada Asturiana': 13,
        'Pulpo a la Gallega': 12, 'Tarta de Santiago': 20
    },
    'Southern': {
        'Buttermilk Biscuits': 16, 'Baked Macaroni Cheese': 14, 'Chicken and Dumplings': 18,
        'Fried Green Tomatoes': 19, 'Pan-cooked Cornbread': 15, 'Hoppin’ John': 17, 'Gumbo': 20,
        'Mississippi Mud Pie': 13, 'Red Rice': 12, 'Tomato, Cheddar, and Bacon Cake': 14
    },
    'French': {
        'Soupe à l’oignon': 18, 'Coq au vin': 15, 'Cassoulet': 20, 'Bœuf bourguignon': 17,
        'Chocolate soufflé': 13, 'Flamiche': 14, 'Confit de canard': 16, 'Salade Niçoise': 19,
        'Ratatouille': 12, 'Tarte Tatin': 15
    },
    'Vietnamese': {
        'Phở': 19, 'Bánh Mì': 17, 'Cơm Tấm': 14, 'Xôi': 16, 'Bánh cuốn': 20,
        'Gỏi cuốn': 15, 'Bun cha': 18, 'Chả giò': 13, 'Cà phê sữa đá': 12, 'Bánh xèo': 14
    }
}

# Aggiungi le pietanze al dataset
for index, row in dataset.iterrows():
    restaurant_name = row['restaurant_name']
    day_of_the_week = row['day_of_the_week']
    restaurant_location = row['restaurant_location']
    cuisine_type = row['cuisine_type']

    remaining_food_preparation_time = row['food_preparation_time']
    total_preparation_time = 0
    dishes = []

    available_dishes = menu_by_cuisine.get(cuisine_type, {}).items()

    # Scegli casualmente una pietanza tra quelle disponibili
    dish_name, preparation_time = random.choice(list(available_dishes))

    distribution_key = (restaurant_name, day_of_the_week, restaurant_location)
    distribution = preparation_time_distributions_dict.get(distribution_key, {'mean': 25, 'std': 2})

    # Campiona il tempo di preparazione per la pietanza dalla distribuzione
    sampled_preparation_time = round(random.gauss(preparation_time, distribution.get('std', 2)))
    sampled_preparation_time = max(10, min(20, sampled_preparation_time))

    # Verifica se l'aggiunta della pietanza non supera il tempo rimanente
    if total_preparation_time + sampled_preparation_time <= remaining_food_preparation_time:
        dishes.append({'dish_name': dish_name, 'preparation_time': sampled_preparation_time})
        total_preparation_time += sampled_preparation_time

    # Se c'è spazio per aggiungere ulteriori pietanze, cerca quella più vicina al tempo rimanente
    while total_preparation_time < remaining_food_preparation_time:
        closest_dish = None
        closest_distance = float('inf')

        for dish_name, preparation_time in menu_by_cuisine.get(cuisine_type, {}).items():
            distribution_key = (restaurant_name, day_of_the_week, restaurant_location)
            distribution = preparation_time_distributions_dict.get(distribution_key, {'mean': 15, 'std': 2})

            # Calcola la distanza tra il tempo rimanente e il tempo di preparazione della pietanza
            distance = abs(remaining_food_preparation_time - total_preparation_time - preparation_time)

            # Verifica se la pietanza è più vicina rispetto alle precedenti
            if distance < closest_distance:
                closest_dish = {'dish_name': dish_name, 'preparation_time': preparation_time}
                closest_distance = distance

        # Aggiungi la pietanza più vicina al tempo rimanente
        dishes.append(closest_dish)
        total_preparation_time += closest_dish['preparation_time']

    if total_preparation_time > row['food_preparation_time']:
        dataset.at[index, 'food_preparation_time'] = total_preparation_time

    dataset.at[index, 'dishes'] = str(dishes)

dataset.to_csv("food_order_updated.csv")