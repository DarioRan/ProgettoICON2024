import pandas as pd


def retrieve_dataframe(KB):
    # Query Prolog per ottenere i dati dei ristoranti

    list_query = KB.get_dishes_info()

    # Lista per memorizzare i dati espansi dei piatti
    dishes_expanded_list = []

    # Itera su ciascun risultato della query Prolog
    for result in list_query:
        # Esegui le operazioni desiderate con ciascun risultato
        restaurant_name = result['RestaurantName']
        restaurant_location = result['RestaurantLocation']
        day_of_the_week = result['DayOfWeek']
        dishes = result['Dishes']

        # Itera su ciascun piatto del ristorante
        for dish in dishes:
            # Aggiungi i dati del piatto alla lista espansa
            dishes_expanded_list.append({
                'restaurant_name': restaurant_name,
                'restaurant_location': restaurant_location,
                'day_of_the_week': day_of_the_week,
                'dish_name': dish['dish_name'],
                'preparation_time': dish['preparation_time']
            })

    # Crea il DataFrame dai dati espansi dei piatti
    dishes_df = pd.DataFrame(dishes_expanded_list)

    return dishes_df
