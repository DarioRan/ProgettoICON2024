from belief_network.belief_network import BeliefNetwork
from find_path.utils import calculate_distance, find_street_names
def predict_road_closure_probability(df,G, time, path):
    bn = BeliefNetwork(df=df)
    bn.train_model()
    prob = 1
    street_names= find_street_names(G, path)
    for i in range(len(street_names) - 1):
        #dovrebbe essere il max
        if street_names[i] in df['Street'].values:
            print(street_names[i])
            bn.get_road_closure_probability(time, street_names[i])

    return prob