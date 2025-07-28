import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import os

# Aggiungi backend matplotlib per evitare errori
import matplotlib
matplotlib.use('Agg')

debug_plot = True  # Per visualizzare il grafo generato

# Path plots
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
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Unable to delete {file_path}: {e}")
        print("Folder contents successfully cleared.")


def determine_edge_weight(node1, node2, weak_set, mandatory_set, discretionary_set):
    """
    Determina il peso dell'arco basato sui tipi di nodo
    MODIFICATO: i nodi weak non si collegano tra loro
    """
    # Identifica i tipi di nodo
    is_node1_weak = node1 in weak_set
    is_node1_mandatory = node1 in mandatory_set
    is_node1_discretionary = node1 in discretionary_set

    is_node2_weak = node2 in weak_set
    is_node2_mandatory = node2 in mandatory_set
    is_node2_discretionary = node2 in discretionary_set

    # NUOVO: Evita collegamenti tra nodi weak
    if is_node1_weak and is_node2_weak:
        return None  # Nessun collegamento tra nodi weak

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
        if (is_node1_weak and is_node2_mandatory) or (is_node1_mandatory and is_node2_weak):
            # Weak <-> Mandatory: peso medio-alto
            return random.randint(5, 8)
        elif (is_node1_mandatory and is_node2_mandatory):
            # Mandatory <-> Mandatory: peso alto (per connessione backbone)
            return random.randint(3, 6)

    # Caso di fallback
    return random.randint(4, 10)


def create_single_graph(total_nodes, mandatory_percentage, discretionary_percentage):
    """
    Crea UN SINGOLO grafo con percentuali specifiche di nodi mandatory e discretionary

    Args:
        total_nodes: Numero totale di nodi
        mandatory_percentage: Percentuale di nodi mandatory (0-100)
        discretionary_percentage: Percentuale di nodi discretionary (0-100)

    Returns:
        tuple: (grafo, weak_nodes, mandatory_nodes, discretionary_nodes)
    """
    if mandatory_percentage + discretionary_percentage > 100:
        raise ValueError("La somma delle percentuali non può superare 100%")

    # Calcola il numero di nodi per ogni tipo
    num_mandatory = int(total_nodes * mandatory_percentage / 100)
    num_discretionary = int(total_nodes * discretionary_percentage / 100)
    num_weak = total_nodes - num_mandatory - num_discretionary

    # Crea i range di nodi
    weak_nodes = list(range(1, num_weak + 1)) if num_weak > 0 else []
    mandatory_nodes = list(range(num_weak + 1, num_weak + num_mandatory + 1)) if num_mandatory > 0 else []
    discretionary_nodes = list(range(num_weak + num_mandatory + 1,
                                   num_weak + num_mandatory + num_discretionary + 1)) if num_discretionary > 0 else []

    print(f"=== CONFIGURAZIONE GRAFO ===")
    print(f"Nodi totali: {total_nodes}")
    print(f"Weak nodes ({num_weak}/{total_nodes} = {num_weak/total_nodes*100:.1f}%): {weak_nodes}")
    print(f"Mandatory nodes ({num_mandatory}/{total_nodes} = {num_mandatory/total_nodes*100:.1f}%): {mandatory_nodes}")
    print(f"Discretionary nodes ({num_discretionary}/{total_nodes} = {num_discretionary/total_nodes*100:.1f}%): {discretionary_nodes}")
    print()

    # Crea il grafo con TUTTI i nodi
    G = nx.Graph()
    all_nodes = weak_nodes + mandatory_nodes + discretionary_nodes

    # Aggiungi tutti i nodi con i loro tipi
    if weak_nodes:
        G.add_nodes_from(weak_nodes, node_type='weak')
    if mandatory_nodes:
        G.add_nodes_from(mandatory_nodes, node_type='power_mandatory')
    if discretionary_nodes:
        G.add_nodes_from(discretionary_nodes, node_type='power_discretionary')

    # Set per controllo veloce
    weak_set = set(weak_nodes)
    mandatory_set = set(mandatory_nodes)
    discretionary_set = set(discretionary_nodes)

    # Crea collegamenti con logica di peso specifica
    # MODIFICATO: evita collegamenti tra nodi weak
    edges_added = 0
    edges_skipped = 0

    for i in all_nodes:
        for j in all_nodes:
            if i != j and not G.has_edge(i, j):
                weight = determine_edge_weight(i, j, weak_set, mandatory_set, discretionary_set)
                if weight is not None:  # Solo se il peso non è None (evita weak-weak)
                    G.add_edge(i, j, weight=weight)
                    edges_added += 1
                else:
                    edges_skipped += 1

    print(f"Collegamenti creati: {edges_added}")
    print(f"Collegamenti weak-weak evitati: {edges_skipped}")

    return G, weak_nodes, mandatory_nodes, discretionary_nodes


def draw_and_save_graph(G, filename="final_graph"):
    """
    Disegna e salva il grafo
    """
    plt.figure(figsize=(12, 10))
    plt.clf()

    pos = nx.spring_layout(G, k=2, iterations=50)
    node_colors = {'weak': 'lightgreen', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
    colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]

    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}

    # Disegna il grafo con stile migliorato
    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold',
            node_size=800, font_size=10, width=1.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Aggiungi legenda
    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color='lightgreen', label='Weak nodes')
    red_patch = mpatches.Patch(color='red', label='Mandatory nodes')
    orange_patch = mpatches.Patch(color='orange', label='Discretionary nodes')
    plt.legend(handles=[green_patch, red_patch, orange_patch], loc='upper right')

    plt.title(f"Grafo con {len(G.nodes())} nodi e {len(G.edges())} collegamenti")

    # Salva il grafo
    path_to_save = f"{path}{filename}.png"
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafo salvato come: {path_to_save}")


def verify_no_weak_connections(G, weak_nodes):
    """
    Verifica che non ci siano collegamenti tra nodi weak
    """
    weak_set = set(weak_nodes)
    weak_connections = []

    for edge in G.edges():
        if edge[0] in weak_set and edge[1] in weak_set:
            weak_connections.append(edge)

    if weak_connections:
        print(f"❌ ERRORE: Trovati {len(weak_connections)} collegamenti tra nodi weak: {weak_connections}")
        return False
    else:
        print("✅ Verificato: nessun collegamento tra nodi weak")
        return True


def analyze_graph(G, weak_nodes, mandatory_nodes, discretionary_nodes):
    """
    Analizza le proprietà del grafo generato
    """
    print(f"\n=== ANALISI DEL GRAFO ===")
    print(f"Numero totale di nodi: {len(G.nodes())}")
    print(f"Numero totale di collegamenti: {len(G.edges())}")
    print(f"Grafo connesso: {nx.is_connected(G)}")

    # Analisi per tipo di nodo
    weak_set = set(weak_nodes)
    mandatory_set = set(mandatory_nodes)
    discretionary_set = set(discretionary_nodes)

    # Conta collegamenti per tipo
    weak_to_mandatory = 0
    weak_to_discretionary = 0
    mandatory_to_mandatory = 0
    mandatory_to_discretionary = 0
    discretionary_to_discretionary = 0

    for edge in G.edges():
        node1, node2 = edge

        if (node1 in weak_set and node2 in mandatory_set) or (node1 in mandatory_set and node2 in weak_set):
            weak_to_mandatory += 1
        elif (node1 in weak_set and node2 in discretionary_set) or (node1 in discretionary_set and node2 in weak_set):
            weak_to_discretionary += 1
        elif node1 in mandatory_set and node2 in mandatory_set:
            mandatory_to_mandatory += 1
        elif (node1 in mandatory_set and node2 in discretionary_set) or (node1 in discretionary_set and node2 in mandatory_set):
            mandatory_to_discretionary += 1
        elif node1 in discretionary_set and node2 in discretionary_set:
            discretionary_to_discretionary += 1

    print(f"\nTipi di collegamenti:")
    print(f"  Weak ↔ Mandatory: {weak_to_mandatory}")
    print(f"  Weak ↔ Discretionary: {weak_to_discretionary}")
    print(f"  Mandatory ↔ Mandatory: {mandatory_to_mandatory}")
    print(f"  Mandatory ↔ Discretionary: {mandatory_to_discretionary}")
    print(f"  Discretionary ↔ Discretionary: {discretionary_to_discretionary}")
    print(f"  Weak ↔ Weak: 0 (come richiesto)")


if __name__ == "__main__":
    start_time = time.time()

    # PARAMETRI CONFIGURABILI
    total_nodes = 30
    mandatory_percentage = 10    # 30% di nodi mandatory
    discretionary_percentage = 10 # 40% di nodi discretionary
    # I rimanenti 30% saranno weak nodes

    # Genera UN SINGOLO grafo con le percentuali specificate
    graph, weak_nodes, mandatory_nodes, discretionary_nodes = create_single_graph(
        total_nodes, mandatory_percentage, discretionary_percentage
    )

    # Verifica che non ci siano collegamenti tra nodi weak
    verify_no_weak_connections(graph, weak_nodes)

    # Analizza le proprietà del grafo
    analyze_graph(graph, weak_nodes, mandatory_nodes, discretionary_nodes)

    # Disegna e salva il grafo
    if debug_plot:
        draw_and_save_graph(graph, "single_complete_graph")

    # Salva il grafo in formato pickle
    import pickle
    graph_file = os.path.join(path, "grafo_3.pickle")
    with open(graph_file, "wb") as f:
        pickle.dump(graph, f)
    print(f"Grafo salvato come: {graph_file}")

    # Verifica caricamento
    with open(graph_file, "rb") as f:
        loaded_graph = pickle.load(f)
    print(f"Verifica caricamento: {nx.is_isomorphic(graph, loaded_graph)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTempo di esecuzione: {elapsed_time:.4f} secondi")

    print("\n=== RIEPILOGO FINALE ===")
    print("✅ Generato UN SINGOLO grafo completo")
    print("✅ Nessun collegamento tra nodi weak")
    print("✅ Percentuali configurabili per mandatory/discretionary")
    print("✅ Logica di peso ottimizzata mantenuta")
    print("\nLogica dei pesi:")
    print("- Discretionary ↔ Weak: peso 1-2 (molto favorito)")
    print("- Discretionary ↔ Mandatory: peso 2-4 (favorito)")
    print("- Discretionary ↔ Discretionary: peso 2-5 (medio-favorito)")
    print("- Weak ↔ Mandatory: peso 5-8 (normale)")
    print("- Mandatory ↔ Mandatory: peso 3-6 (per backbone)")
    print("- Weak ↔ Weak: NESSUN COLLEGAMENTO")
