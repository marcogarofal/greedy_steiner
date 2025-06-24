#generazione sotto grafici senza tutto il resto
#verificare cicli oppure verificare che mandatori siano connessi ad altri mandatori

#eliminato controllo presenti_tutti e lasciato controllo connesso

#se abbiamo una macchina con poca ram, potrebbe essere utile verificare  con il controllo presenti_tutti e dopo con is_connected, altrimenti saturiamo la ram (verificare); controllo is_connected usa più ram

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random

from collections import Counter
from itertools import chain
import copy
import time
import os

# Aggiungi backend matplotlib per evitare errori
import matplotlib
matplotlib.use('Agg')

debug0=True #plot graph1 and graph2
debug=False
debug2=False
debug3=False
debug_plot_graph=False
debug_save=False


#path plots
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'graphs/')
if not os.path.exists(path):
    os.makedirs(path)
else:
    if os.path.exists(path) and os.path.isdir(path):
        # Cancella il contenuto della cartella
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Cancella i file all'interno della cartella
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Cancella le sotto-cartelle all'interno della cartella
            except Exception as e:
                print(f"Unable to delete {file_path}: {e}")
        print("Folder contents successfully cleared.")
    else:
        print("The folder does not exist or is not a folder.")


def determine_edge_weight(node1, node2, weak_set, mandatory_set, discretionary_set):
    """
    Determina il peso dell'arco basato sui tipi di nodo
    """
    # Identifica i tipi di nodo
    is_node1_weak = node1 in weak_set
    is_node1_mandatory = node1 in mandatory_set
    is_node1_discretionary = node1 in discretionary_set
    
    is_node2_weak = node2 in weak_set
    is_node2_mandatory = node2 in mandatory_set
    is_node2_discretionary = node2 in discretionary_set
    
    # Logica di assegnazione pesi
    if is_node1_discretionary or is_node2_discretionary:
        # Almeno uno è discretionary
        if (is_node1_discretionary and is_node2_weak) or (is_node1_weak and is_node2_discretionary):
            # Discretionary <-> Weak: peso molto basso (favorisce connessione)
            return random.randint(1, 2)
        elif (is_node1_discretionary and is_node2_mandatory) or (is_node1_mandatory and is_node2_discretionary):
            # Discretionary <-> Mandatory: peso basso
            return random.randint(2, 4)
        elif is_node1_discretionary and is_node2_discretionary:
            # Discretionary <-> Discretionary: peso medio-basso
            return random.randint(2, 5)
    else:
        # Nessuno è discretionary
        if (is_node1_weak and is_node2_weak):
            # Weak <-> Weak: peso medio (non molto utile)
            return random.randint(6, 10)
        elif (is_node1_weak and is_node2_mandatory) or (is_node1_mandatory and is_node2_weak):
            # Weak <-> Mandatory: peso medio-alto
            return random.randint(5, 8)
        elif (is_node1_mandatory and is_node2_mandatory):
            # Mandatory <-> Mandatory: peso alto (per connessione backbone)
            return random.randint(3, 6)
    
    # Caso di fallback
    return random.randint(4, 10)


def create_graph(weak_nodes=None, power_nodes_mandatory=None, power_nodes_discretionary=None, 
                 discretionary_weight_range=(1, 3), normal_weight_range=(4, 10)):
    """
    Crea un grafo con pesi preferenziali per i nodi discretionary
    
    Args:
        discretionary_weight_range: Range di pesi per collegamenti da/verso nodi discretionary (default: 1-3)
        normal_weight_range: Range di pesi per altri collegamenti (default: 4-10)
    """
    G = nx.Graph()
    all_nodes = []
    
    # Aggiungi nodi con i loro tipi
    if (weak_nodes is not None) and (power_nodes_mandatory is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory) + list(power_nodes_discretionary)
    elif (weak_nodes is not None) and (power_nodes_mandatory is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory)
    elif (weak_nodes is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_discretionary)
    elif power_nodes_discretionary is not None:
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(power_nodes_discretionary)    
    else:
        print("\tnot a possible case")

    # Set per controllo veloce
    weak_set = set(weak_nodes) if weak_nodes else set()
    mandatory_set = set(power_nodes_mandatory) if power_nodes_mandatory else set()
    discretionary_set = set(power_nodes_discretionary) if power_nodes_discretionary else set()
    
    # Crea collegamenti con logica di peso specifica
    for i in all_nodes:
        for j in all_nodes:
            if i != j and not G.has_edge(i, j):
                weight = determine_edge_weight(i, j, weak_set, mandatory_set, discretionary_set)
                G.add_edge(i, j, weight=weight)

    return G


def draw_graph(G):
    plt.clf()
    global count_picture
    if debug2:
        print("\tnodes_", G.nodes(), " edges:", G.edges)
    pos = nx.spring_layout(G)
    node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
    colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]

    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}
    
    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Sostituisci plt.show() con salvataggio
    plt.savefig(f'{path}temp_graph_{count_picture:03d}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico temporaneo salvato come temp_graph_{count_picture:03d}.png")


count_picture=0
def save_graph(G, name=None):
    global count_picture
    
    # Pulisci completamente il plot
    plt.figure(figsize=(10, 8))  # Crea una nuova figura
    plt.clf()  # Pulisci
    
    if name == None or name == "best_tree":
        pos = nx.spring_layout(G)
        node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
        colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]

        edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}

        nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    path_to_save = f"{path}{count_picture:03d}_graph.png"
    
    if name is not None:
        path_to_save = f"{path}{count_picture:03d}_{name}_graph.png"
    
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.close()  # Chiudi la figura per liberare memoria
    count_picture += 1
    print(f"Saved: {path_to_save}")


added_edges=set()
def join_2_trees(graph1, graph2, weak_nodes=None, power_nodes_mandatory=None, power_nodes_discretionary=None):
    G = nx.Graph()
    all_nodes = []

    if (weak_nodes is not None) and (power_nodes_mandatory is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory) + list(power_nodes_discretionary)
    elif (weak_nodes is not None) and (power_nodes_mandatory is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory)
    elif power_nodes_discretionary is not None:
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(power_nodes_discretionary)
    else:
        print("\tnot a possible case2")

    # Set per controllo
    weak_set = set(weak_nodes) if weak_nodes else set()
    mandatory_set = set(power_nodes_mandatory) if power_nodes_mandatory else set()
    discretionary_set = set(power_nodes_discretionary) if power_nodes_discretionary else set()

    for i in all_nodes:
        for j in all_nodes:
            if i != j and not G.has_edge(i, j):
                # Cerca la tupla con (i, j) nel set basandoti sui primi due elementi
                matching_tuple = next((tup for tup in added_edges if set(tup[:2]) == set((i, j))), None)
                   
                if (graph1.has_edge(i, j) or graph1.has_edge(j, i)):
                    G.add_edge(i, j, weight=graph1[i][j]['weight'])
                elif (graph2.has_edge(i, j) or graph2.has_edge(j,i)):
                    G.add_edge(i, j, weight=graph2[i][j]['weight'])
                elif matching_tuple:
                    weight_value = next(iter(matching_tuple[2]))  # Estrai il valore dal set 'weight'
                    G.add_edge(i, j, weight=weight_value)
                else:
                    # Usa la logica di peso avanzata per nuovi archi
                    weight = determine_edge_weight(i, j, weak_set, mandatory_set, discretionary_set)
                    G.add_edge(i, j, weight=weight)
                    new_element = (i, j, frozenset({weight}))
                    added_edges.add(new_element)
    return G


def generate_combinations(elements):
    number_of_nodes=len(elements)
    for x in range(1, number_of_nodes+1):
        for combo in combinations(elements, x):
            yield combo


def generate_graphs(graph, power_nodes_discretionary):
    print("\n2nd graph")
    graph2 = create_graph(power_nodes_discretionary=power_nodes_discretionary)
    if debug0:
        draw_graph(graph2)

    combinations_only_power_nodes_discretionary = generate_combinations(graph2.nodes)
   
    count = 0
    
    for combo in combinations_only_power_nodes_discretionary:
        if combo:
            # combo è già una tupla di nodi - non serve ciclo interno
            lista_risultante = list(combo)  # Converti direttamente in lista

            if count == 0:
                graph3 = join_2_trees(graph, graph2, weak_nodes=weak_nodes, 
                                     power_nodes_mandatory=power_nodes_mandatory, 
                                     power_nodes_discretionary=lista_risultante)
                graph3_bak = graph3
            else:
                graph3 = join_2_trees(graph3_bak, graph2, weak_nodes=weak_nodes, 
                                     power_nodes_mandatory=power_nodes_mandatory, 
                                     power_nodes_discretionary=lista_risultante)

            count += 1
            print(f"\n\tGenerating graph {count} with discretionary nodes: {lista_risultante}")
            yield graph3


        
    
#funzioni non utilizzate ma che potrebbero servire
# def get_weight(item):
#     return item[1]['weight']


# def compare_2_trees(tree1, tree2, power_nodes_mandatory, power_nodes_discretionary, capacities):
    
#     if tree1==None:
#         tree1=tree2

#     print("\t\tcompare_2_trees")

#     number_nodes_tree1=len(tree1.nodes())
#     number_nodes_tree2=len(tree2.nodes())

#     edges_with_weights1 = [(edge, tree1.get_edge_data(edge[0], edge[1])) for edge in tree1.edges()]
#     max_edge_cost1=max(edges_with_weights1, key=get_weight)
#     max_edge_cost1=max_edge_cost1[1]["weight"]


#     edgecost1=0

#     edges_with_weights2 = [(edge, tree2.get_edge_data(edge[0], edge[1])) for edge in tree2.edges()]
#     max_edge_cost2=max(edges_with_weights2, key=get_weight)
#     max_edge_cost2=max_edge_cost2[1]["weight"]
    

#     edgecost2=0

#     set_power_nodes=set(list(power_nodes_mandatory)+list(power_nodes_discretionary))
#     cost_degree1=0
#     for x in tree1.nodes():
#         if x in set_power_nodes:
#             try:
#                 tree1.degree(x)
#                 cost_degree1+=tree1.degree(x)/capacities[x]
#             except AttributeError:
#                 print("error")

#     cost_degree2=0
#     for x in tree2.nodes():
#         if x in set_power_nodes:
#             try:
#                 tree2.degree(x)
#                 cost_degree2+=tree2.degree(x)/capacities[x]
#             except AttributeError:
#                 print("error")
        
#     if debug:
#         print("cost_degree_:", cost_degree1, cost_degree2)
#     if debug:
#         print("cost_degree:", cost_degree1/len(tree1.nodes()), cost_degree2/len(tree2.nodes()))
#     cost_degree1=cost_degree1/len(tree1.nodes())
#     cost_degree2=cost_degree2/len(tree2.nodes())

    
    

#     if max_edge_cost1>=max_edge_cost2:
#         if debug:
#             print("\tcaso max1>max2, len:", len(edges_with_weights1), " :", edges_with_weights1)
#         for edge1, data1 in edges_with_weights1:
#             edgecost1+=data1['weight']
#         if debug:
#             print("\t\t-edgecost1:", edgecost1, " max1:", max_edge_cost1, " numbernodes1:", number_nodes_tree1)
#         edgecost1=edgecost1/(max_edge_cost1*(len(edges_with_weights1)))


#         for edge2, data2 in edges_with_weights2:
#             edgecost2+=data2['weight']
#         if debug:
#             print("\t\t-edgecost2:", edgecost2, " max1:", max_edge_cost1, " numbernodes2:", number_nodes_tree2)
#         edgecost2=edgecost2/(max_edge_cost1*(len(edges_with_weights1)))
          
#     else:
#         if debug:
#             print("\tcaso max2>max1, len:", len(edges_with_weights2), " :", edges_with_weights2)
#         for edge1, data1 in edges_with_weights1:
#             edgecost1+=data1['weight']
#         if debug:
#             print("\t\t-edgecost1:", edgecost1, " max1", max_edge_cost1, " numbernodes1:", number_nodes_tree1)
#         edgecost1=edgecost1/(max_edge_cost2*(len(edges_with_weights2)))
      

#         for edge2, data2 in edges_with_weights2:
#             edgecost2+=data2['weight']
#         if debug:
#             print("\t\t-edgecost2:", edgecost2, " max2", max_edge_cost2, " numbernodes2:", number_nodes_tree2)
#         edgecost2=edgecost2/(max_edge_cost2*(len(edges_with_weights2)))
        
#     if debug:
#         print("\nedge:", edgecost1, edgecost2, "\ndegree:", cost_degree1, cost_degree2, "\nsum:",edgecost1+cost_degree1, edgecost2+cost_degree2)

#     #print("\t\tcost_degree_:", cost_degree1, cost_degree2)
#     #print("\t\tedgecost1:", edgecost1, " edgecost2:", edgecost2)
            

#     if edgecost1+cost_degree1<=edgecost2+cost_degree2:
#         if debug:
#             print("\n\n\nbest_tree")
#         #return tree1, edgecost1/number_nodes_tree1, cost_degree1
#         return tree1, edgecost1, cost_degree1

#     else:
#         if debug:
#             print("\n\n\nnew_tree")
#         #return tree2, edgecost2/number_nodes_tree2, cost_degree2
#         return tree2, edgecost2, cost_degree2


# def build_tree_from_list_edges(G, desired_edges, no_plot=None):
#     G_copy=copy.deepcopy(G)
#     edges_to_remove = [edge for edge in G_copy.edges() if edge not in desired_edges]
#     G_copy.remove_edges_from(edges_to_remove)

#     if no_plot==False or no_plot==None:
#         pos = nx.spring_layout(G_copy)
#         node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
#         colors = [node_colors[data['node_type']] for _, data in G_copy.nodes(data=True)]
#         edge_labels = {(i, j): G_copy[i][j].get('weight', None) for i, j in G_copy.edges()}

#         nx.draw(G_copy, pos, with_labels=True, node_color=colors, font_weight='bold')
#         nx.draw_networkx_edge_labels(G_copy, pos, edge_labels=edge_labels)
#         nx.draw_networkx_edges(G_copy, pos, edgelist=desired_edges)
#         plt.show()
#     return G_copy
    

# def draw_tree_highlighting_edges(G, list_edges, save=None):
#     pos = nx.spring_layout(G)

#     node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
#     colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]

#     edge_labels = {(i, j): G[i][j].get('weight', None) for i, j in G.edges()}
    
#     plt.clf()
    
#     nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     nx.draw_networkx_edges(G, pos, edgelist=list_edges, edge_color='blue', width=2)
    
#     if save==None or save==False:
#         plt.show()
#     else:
#         save_graph(G, "colored_intermediate_graph")


import sys
if __name__ == "__main__":
  
    start_time = time.time()

    num_nodes = 5

    #critical cases: 0 weak, 0 mandatory, 0 discretionary, 0 weak e 0 mandatory, 0weak e 0 discretinary, 0 discretionary e 0 mandatory
    num_weak_nodes = int(0.4 * num_nodes)
    num_power_nodes_mandatory = int(0.2 * num_nodes)
    num_power_nodes_discretionary = num_nodes - num_weak_nodes - num_power_nodes_mandatory

    if num_weak_nodes==num_nodes:
        print("only weak nodes")
        sys.exit()
    

    weak_nodes = range(1, num_weak_nodes + 1)
    power_nodes_mandatory = range(num_weak_nodes + 1, num_weak_nodes + num_power_nodes_mandatory + 1)
    power_nodes_discretionary = range(num_weak_nodes + num_power_nodes_mandatory + 1, num_weak_nodes + num_power_nodes_mandatory + num_power_nodes_discretionary + 1)
    
    capacities = {1: 10, 2: 30, 3:2, 4: 1, 5: 10, 6:4, 7:5, 8:5, 9:5, 10:5, 11:5, 12:5, 13:5, 14:5, 15:5, 16:5, 17:5, 18:5, 19:5, 20:5}
    
    print("num_weak_nodes+num_power_nodes_mandatory:", num_weak_nodes+num_power_nodes_mandatory)
    print(f"Weak nodes: {list(weak_nodes)}")
    print(f"Mandatory nodes: {list(power_nodes_mandatory)}")
    print(f"Discretionary nodes: {list(power_nodes_discretionary)}")

    list_graphs=[]
   
    if num_weak_nodes+num_power_nodes_mandatory>0:
        #Graph with power mandatory and weak nodes - USA LA NUOVA FUNZIONE
        graph = create_graph(weak_nodes, power_nodes_mandatory, power_nodes_discretionary=None)
        print("1st graph")
        #steinerTree()
        draw_graph(graph)
        save_graph(graph)
    else:
        graph = nx.Graph()
        best_tree=None

    list_graphs.append(graph)

    graphs=generate_graphs(graph, power_nodes_discretionary)
    
    for graph in graphs:
        #steinerTree()
        draw_graph(graph)
        save_graph(graph)
        list_graphs.append(graph)

    end_time = time.time()
    
    # Time elapsed
    elapsed_time = end_time - start_time
    print(f"Running time: {elapsed_time} seconds")
    print("len:", len(list_graphs))

    import pickle

    # Esporta ciascun grafo nella lista in un file separato
    for i, G in enumerate(list_graphs):
        nome_file = os.path.join(path, f"grafo_{i}.pickle")
        with open(nome_file, "wb") as f:
            pickle.dump(G, f)

    # Ora puoi caricare i grafi da file
    grafo_caricato = []
    for i in range(len(list_graphs)):
        nome_file = os.path.join(path, f"grafo_{i}.pickle")
        with open(nome_file, "rb") as f:
            grafo_caricato.append(pickle.load(f))

    # Verifica se i grafi sono uguali
    for G, H in zip(list_graphs, grafo_caricato):
        print(nx.is_isomorphic(G, H))

    print("\n=== RIEPILOGO PESI ===")
    print("I nodi discretionary avranno collegamenti con pesi più bassi:")
    print("- Discretionary ↔ Weak: peso 1-2 (molto favorito)")
    print("- Discretionary ↔ Mandatory: peso 2-4 (favorito)")
    print("- Discretionary ↔ Discretionary: peso 2-5 (medio-favorito)")
    print("- Weak ↔ Mandatory: peso 5-8 (normale)")
    print("- Mandatory ↔ Mandatory: peso 3-6 (per backbone)")
    print("- Weak ↔ Weak: peso 6-10 (meno favorito)")