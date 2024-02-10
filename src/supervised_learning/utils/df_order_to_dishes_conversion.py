
def retrieve_dataframe(KB):
    # Query Prolog per ottenere i dati dei ristoranti

    dishes_df = KB.get_dishes_info()

    return dishes_df


from KB.KB import KB
KB = KB()
retrieve_dataframe(KB).to_csv('../../dataset/dishes_df')
