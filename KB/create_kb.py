import pandas as pd

def write_fact_to_file(fact, file_path):
    # Verifica se il fatto è già presente
    with open(file_path, 'r', encoding='utf-8') as file:
        existing_content = file.read()

    if fact not in existing_content:
        # Riapri il file in modalità append e scrivi il fatto
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(f"{fact}.\n")


def writeOrdersInfo(dataSet):
    file_path = 'knowledge_base_orders.pl'
    with open(file_path, "w", encoding="utf-8"):  # Sovrascrivi il file (svuotalo)
        write_fact_to_file(":- encoding(utf8)", file_path)
        for index, row in dataSet.iterrows():
            order_id = row['order_id']
            customer_id = row['customer_id']
            restaurant_name = row['restaurant_name']
            cuisine_type = row['cuisine_type']  # Non formattare come float
            cost_of_the_order = row['cost_of_the_order']
            day_of_the_week = row['day_of_the_week']  # Non formattare come float
            rating = row['rating']
            food_preparation_time = row['food_preparation_time']
            delivery_time = row['delivery_time']
            restaurant_location = row['restaurant_location']  # Non formattare come float
            customer_location = row['customer_location']
            dishes = row['dishes']
            prolog_clause = f"order({order_id},{customer_id},{restaurant_name},'{cuisine_type}',{cost_of_the_order},'{day_of_the_week}'," \
                            f"{rating},{food_preparation_time},{delivery_time},{restaurant_location},'{customer_location}', '{dishes}')"
            write_fact_to_file(prolog_clause, file_path)

def fix_orders(file_path):
    # Leggi il contenuto del file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Correggi gli ordini
    corrected_content = content.replace("Not given", "'Not given'")  # Aggiungi le virgolette intorno a 'Not given'
    corrected_content = corrected_content.replace("[{", "[{")  # Mantieni le parentesi quadre iniziali
    corrected_content = corrected_content.replace("}]", "}]")  # Mantieni le parentesi quadre finali
    corrected_content = corrected_content.replace("'[{", "[{")  # Mantieni le parentesi quadre iniziali
    corrected_content = corrected_content.replace("}]'", "}]")  # Mantieni le parentesi quadre finali

    # Correggi i nomi dei ristoranti
    lines = corrected_content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("order("):
            parts = line.split(',')
            # Rimuovi eventuali apici singoli aggiuntivi nel nome del ristorante
            parts[2] = parts[2].replace("'", "")  # Rimuovi tutti gli apici singoli
            # Assicurati che il nome del ristorante sia racchiuso tra apici singoli
            parts[2] = f"'{parts[2]}'"
            lines[i] = ','.join(parts)

    corrected_content = '\n'.join(lines)

    # Scrivi il contenuto corretto nel file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(corrected_content)


dataSet = pd.read_csv('../dataset/food_order_final.csv')

# Esegui la funzione writeOrdersInfo per ogni riga del dataset
writeOrdersInfo(dataSet)
# Correggi gli ordini nel file
fix_orders('knowledge_base_orders.pl')
