import pandas as pd


def write_fact_to_file(fact, file_path):
    """
    Write a fact to a file, if it is not already present

    :param fact: the fact to write

    :param file_path: the path of the file to write to

    :return: None
    """
    # Verifica se il fatto è già presente
    with open(file_path, 'r', encoding='utf-8') as file:
        existing_content = file.read()

    if fact not in existing_content:
        # Riapri il file in modalità append e scrivi il fatto
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(f"{fact}.\n")


def writeOrdersInfo(dataSet):
    """
    Write the orders information to a file

    :param dataSet: the dataset containing the orders information

    :return: None

    """
    file_path = 'knowledge_base_orders.pl'
    with open(file_path, "w", encoding="utf-8"):  # Sovrascrivi il file (svuotalo)
        write_fact_to_file(":- encoding(utf8)", file_path)
        for index, row in dataSet.iterrows():
            restaurant_name = row['restaurant_name']
            cuisine_type = row['cuisine_type']  # Non formattare come float
            cost_of_the_order = row['cost_of_the_order']
            day_of_the_week = row['day_of_the_week']  # Non formattare come float
            food_preparation_time = row['food_preparation_time']
            delivery_time = row['delivery_time']
            restaurant_location = row['restaurant_location']  # Non formattare come float
            customer_location = row['customer_location']
            dishes = row['dishes']
            prolog_clause = f"order({restaurant_name},'{cuisine_type}',{cost_of_the_order},'{day_of_the_week}'," \
                            f"{food_preparation_time},{delivery_time},{restaurant_location},'{customer_location}', '{dishes}')"
            write_fact_to_file(prolog_clause, file_path)


def writeDishesInfo():
    """

    Write the dishes information to a file

    :return: None

    """
    menu = {
        'Korean': ['Bibimbap', 'Kimchi', 'Bulgogi', 'Japchae', 'Samgyeopsal', 'Doenjang Jjigae', 'Tteokbokki', 'Gimbap',
                   'Sundubu Jjigae', 'Galbi'],
        'Mexican': ['Carnitas', 'Tacos', 'Enchiladas', 'Chiles Rellenos', 'Tamales', 'Pozole', 'Chiles en Nogada',
                    'Guacamole', 'Quesadillas', 'Salsa Verde'],
        'American': ['Hot Dogs', 'French Fries', 'Chicken Tenders', 'Pizza', 'Burgers', 'Buffalo Chicken Wings',
                     'Tater Tots', 'Apple Pie', 'Barbecue Ribs', 'Reuben Sandwich'],
        'Indian': ['Biryani', 'Butter Chicken', 'Chaat', 'Korma', 'Lamb Saag', 'Dal', 'Dosa', 'Samosa',
                   'Tandoori Chicken', 'Rogan Josh'],
        'Italian': ['Pasta Carbonara', 'Pizza', 'Spaghetti con Salsa di Pomodoro', 'Pasta alla Gricia',
                    'Rigatoni all Amatriciana', 'Ossobuco', 'Arancini', 'Ragù Bolognese', 'Risotto', 'Lasagna'],
        'Mediterranean': ['Greek Salad', 'Armenian Losh Kebab', 'Mediterranean White Bean Soup',
                          'Garlicky Sautéed Shrimp with Creamy White Beans and Blistered Tomatoes',
                          'Mediterranean Falafel Bowls', 'Mediterranean Diet Chicken', 'Mediterranean Baked Cod Recipe',
                          'Mediterranean Grilled Chicken + Dill Greek Yogurt Sauce', 'Mediterranean Pasta Salad',
                          'Mediterranean Quinoa Salad'],
        'Chinese': ['Peking Roasted Duck', 'Kung Pao Chicken', 'Sweet and Sour Pork', 'Hot Pot', 'Dumplings',
                    'Chow Mein', 'Yangzhou Fried Rice', 'Fish-Flavored Shredded Pork', 'Sweet and Sour Pork Fillet',
                    'Congee'],
        'Japanese': ['Sushi', 'Tempura', 'Yakitori', 'Tsukemono (sottaceti)', 'Yakisoba (noodles saltati)', 'Kaiseki',
                     'Ichiju-ju', 'Sashimi', 'Nigiri', 'Miso Soup'],
        'Middle Eastern': ['Jewelled rice', 'Cauliflower and chickpea tagine', 'Beef Dolmas with Apricots and Tamarind',
                           'Cauliflower Shawarma Berber', 'Roast Chicken with Sumac Flatbread (M’sakhan)',
                           'Persian Kuku Sabzi', 'Hummus', 'Baba Ghannouj', 'Falafel', 'Shawarma'],
        'Thai': ['Som Tum', 'Pad Thai', 'Tom Yum Goong', 'Khao Pad', 'Guay Tiew Reua', 'Tom Kha Kai', 'Massaman Curry',
                 'Green Curry', 'Thai Fried Rice', 'Nam Tok Mu'],
        'Spanish': ['Paella', 'Tortilla de Patatas', 'Patatas Bravas', 'Gazpacho', 'Churros', 'Jamón Serrano', 'Pisto',
                    'Fabada Asturiana', 'Pulpo a la Gallega', 'Tarta de Santiago'],
        'Southern': ['Buttermilk Biscuits', 'Baked Macaroni Cheese', 'Chicken and Dumplings', 'Fried Green Tomatoes',
                     'Pan-cooked Cornbread', 'Hoppin’ John', 'Gumbo', 'Mississippi Mud Pie', 'Red Rice',
                     'Tomato Cheddar and Bacon Cake'],
        'French': ['Soupe à l’oignon', 'Coq au vin', 'Cassoulet', 'Bœuf bourguignon', 'Chocolate soufflé', 'Flamiche',
                   'Confit de canard', 'Salade Niçoise', 'Ratatouille', 'Tarte Tatin'],
        'Vietnamese': ['Phở', 'Bánh Mì', 'Cơm Tấm', 'Xôi', 'Bánh cuốn', 'Gỏi cuốn', 'Bun cha', 'Chả giò',
                       'Cà phê sữa đá', 'Bánh xèo']
    }

    file_path = 'knowledge_base_orders.pl'
    for cuisine_type, dishes in menu.items():
        for dish in dishes:
            prolog_clause = f'dish(\'{cuisine_type}\', \'{dish}\')'
            write_fact_to_file(prolog_clause, file_path)


def add_prolog_queries(output_filename):
    """
    Add Prolog queries to the output file

    :param output_filename: the name of the file to write to

    :return: None
    """
    with open(output_filename, 'a') as f:
        f.write("\n% Queries\n")
        f.write("restaurants_by_cuisine(CuisineType, RestaurantName) :- "
                "order(RestaurantName, CuisineType, _, _, _, _, _, _, _).\n")
        f.write("dishes_by_cuisine(CuisineType, Dishes) :- dish(CuisineType, Dishes).\n")
        f.write("all_cuisine_types(CuisineTypes) :- setof(CuisineType, "
                "Dish^dish(CuisineType, Dish), CuisineTypes).\n")
        f.write("restaurant_loc_by_cuisine(CuisineType, RestaurantName, RestaurantLocation) :- "
                "order(RestaurantName, CuisineType, _, _, _, _, RestaurantLocation, _, _).\n")
        f.write("get_dishes_info(RestaurantName,RestaurantLocation,DayOfWeek,Dishes) :- "
                "order(RestaurantName, _, _, DayOfWeek, _, _, RestaurantLocation, _, Dishes).\n")


def fix_orders(file_path):
    """
    Fix the orders file

    :param file_path: the path of the file to fix

    :return: None
    """
    # Leggi il contenuto del file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Correggi gli ordini
    corrected_content = content.replace("[{", "[{")  # Mantieni le parentesi quadre iniziali
    corrected_content = corrected_content.replace("}]", "}]")  # Mantieni le parentesi quadre finali
    corrected_content = corrected_content.replace("'[{", "[{")  # Mantieni le parentesi quadre iniziali
    corrected_content = corrected_content.replace("}]'", "}]")  # Mantieni le parentesi quadre finali

    # Correggi i nomi dei ristoranti
    lines = corrected_content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("order("):
            parts = line.split(',')
            # Rimuovi eventuali apici singoli aggiuntivi nel nome del ristorante
            parts[0] = parts[0].replace("'", "")  # Rimuovi tutti gli apici singoli
            # Assicurati che il nome del ristorante sia racchiuso tra apici singoli
            parts[0] = f"'{parts[0]}'"
            parts[0] = parts[0].replace("'order(", "order('")
            lines[i] = ','.join(parts)

    corrected_content = '\n'.join(lines)

    # Scrivi il contenuto corretto nel file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(corrected_content)


dataSet = pd.read_csv('../dataset/food_order_final.csv')

writeOrdersInfo(dataSet)
fix_orders('knowledge_base_orders.pl')
writeDishesInfo()
add_prolog_queries('knowledge_base_orders.pl')