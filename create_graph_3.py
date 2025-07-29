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


def determine_edge_weight_dijkstra_loses(node1, node2, weak_set, mandatory_set, discretionary_set):
    """
    Determina il peso dell'arco basato sui tipi di nodo
    OTTIMIZZATO PER FAR PERDERE DIJKSTRA
    """
    # Identifica i tipi di nodo
    is_node1_weak = node1 in weak_set
    is_node1_mandatory = node1 in mandatory_set
    is_node1_discretionary = node1 in discretionary_set

    is_node2_weak = node2 in weak_set
    is_node2_mandatory = node2 in mandatory_set
    is_node2_discretionary = node2 in discretionary_set

    # Evita collegamenti tra nodi weak
    if is_node1_weak and is_node2_weak:
        return None

    # STRATEGIA: Rendi i percorsi attraverso discretionary molto costosi
    if is_node1_discretionary or is_node2_discretionary:
        # Almeno uno è discretionary
        if (is_node1_discretionary and is_node2_weak) or (is_node1_weak and is_node2_discretionary):
            # W-D: Non più super economici! Ora simili a W-M
            return random.randint(4, 7)  # Era 1-2, ora 4-7
            
        elif (is_node1_discretionary and is_node2_mandatory) or (is_node1_mandatory and is_node2_discretionary):
            # D-M: ESTREMAMENTE costosi per penalizzare la "backbone" di Dijkstra
            if random.random() < 0.3:
                return random.randint(80, 120)  # 30% super costosi
            else:
                return random.randint(50, 80)   # 70% molto costosi
                
        elif is_node1_discretionary and is_node2_discretionary:
            # D-D: Costosi per evitare "ponti" tra discretionary
            return random.randint(25, 40)  # Era 2-5, ora molto più alto
    else:
        # Nessuno è discretionary
        if (is_node1_weak and is_node2_mandatory) or (is_node1_mandatory and is_node2_weak):
            # W-M: Ora MOLTO economici per favorire connessioni dirette
            return random.randint(2, 4)  # Era 4-7, ora 2-4
            
        elif (is_node1_mandatory and is_node2_mandatory):
            # M-M: Economici per backbone mandatory efficiente
            return random.randint(1, 3)  # Era 3-6, ora 1-3

    # Caso di fallback
    return random.randint(5, 10)


def create_graph_topology_against_dijkstra(total_nodes, mandatory_percentage, discretionary_percentage):
    """
    Crea un grafo con topologia sfavorevole per Dijkstra
    I discretionary sono posizionati in modo svantaggioso
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

    print(f"=== CONFIGURAZIONE GRAFO ANTI-DIJKSTRA ===")
    print(f"Nodi totali: {total_nodes}")
    print(f"Weak nodes ({num_weak}/{total_nodes} = {num_weak/total_nodes*100:.1f}%): {weak_nodes}")
    print(f"Mandatory nodes ({num_mandatory}/{total_nodes} = {num_mandatory/total_nodes*100:.1f}%): {mandatory_nodes}")
    print(f"Discretionary nodes ({num_discretionary}/{total_nodes} = {num_discretionary/total_nodes*100:.1f}%): {discretionary_nodes}")
    print()

    # Crea il grafo
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

    # STRATEGIA TOPOLOGICA: Crea più connessioni dirette W-M
    edges_added = 0
    edges_skipped = 0
    high_cost_edges = 0
    strategic_wm_connections = 0

    # FASE 1: Assicura che ogni weak abbia almeno 2-3 connessioni dirette economiche ai mandatory
    print("FASE 1: Creazione connessioni strategiche W-M...")
    for w in weak_nodes:
        # Ogni weak si connette a 2-3 mandatory con pesi bassi
        num_connections = min(len(mandatory_nodes), random.randint(2, 3))
        chosen_mandatory = random.sample(mandatory_nodes, num_connections)
        for m in chosen_mandatory:
            weight = random.randint(1, 3)  # Pesi molto bassi
            G.add_edge(w, m, weight=weight)
            edges_added += 1
            strategic_wm_connections += 1

    # FASE 2: Connetti mandatory tra loro con pesi bassi (backbone efficiente)
    print("FASE 2: Creazione backbone mandatory...")
    for i, m1 in enumerate(mandatory_nodes):
        for m2 in mandatory_nodes[i+1:]:
            weight = random.randint(1, 2)  # Backbone molto efficiente
            G.add_edge(m1, m2, weight=weight)
            edges_added += 1

    # FASE 3: Connetti discretionary con pesi alti e limitazioni
    print("FASE 3: Connessioni discretionary penalizzanti...")
    for d in discretionary_nodes:
        # Ogni discretionary si connette solo a ALCUNI weak (non tutti)
        num_weak_connections = random.randint(3, 6)  # Connessioni limitate
        reachable_weak = random.sample(weak_nodes, min(num_weak_connections, len(weak_nodes)))
        
        for w in reachable_weak:
            weight = determine_edge_weight_dijkstra_loses(d, w, weak_set, mandatory_set, discretionary_set)
            if weight:
                G.add_edge(d, w, weight=weight)
                edges_added += 1
        
        # Connetti a mandatory con pesi MOLTO alti
        for m in mandatory_nodes:
            weight = determine_edge_weight_dijkstra_loses(d, m, weak_set, mandatory_set, discretionary_set)
            if weight:
                G.add_edge(d, m, weight=weight)
                edges_added += 1
                if weight >= 50:
                    high_cost_edges += 1
        
        # Connessioni limitate tra discretionary
        other_disc = [x for x in discretionary_nodes if x != d]
        if other_disc and random.random() < 0.5:  # Solo 50% di probabilità
            chosen_disc = random.choice(other_disc)
            weight = random.randint(30, 50)  # Costose
            if not G.has_edge(d, chosen_disc):
                G.add_edge(d, chosen_disc, weight=weight)
                edges_added += 1

    # FASE 4: Completa il grafo con le connessioni rimanenti
    print("FASE 4: Completamento grafo...")
    for i in all_nodes:
        for j in all_nodes:
            if i != j and not G.has_edge(i, j):
                weight = determine_edge_weight_dijkstra_loses(i, j, weak_set, mandatory_set, discretionary_set)
                if weight is not None:
                    G.add_edge(i, j, weight=weight)
                    edges_added += 1
                    
                    if ((i in discretionary_set and j in mandatory_set) or 
                        (i in mandatory_set and j in discretionary_set)) and weight >= 50:
                        high_cost_edges += 1
                else:
                    edges_skipped += 1

    print(f"\n=== STATISTICHE CREAZIONE ===")
    print(f"Collegamenti totali creati: {edges_added}")
    print(f"Collegamenti W-M strategici (basso costo): {strategic_wm_connections}")
    print(f"Collegamenti D-M ad alto costo (≥50): {high_cost_edges}")
    print(f"Collegamenti weak-weak evitati: {edges_skipped}")

    return G, weak_nodes, mandatory_nodes, discretionary_nodes


def draw_and_save_graph(G, filename="anti_dijkstra_graph"):
    """
    Disegna e salva il grafo evidenziando la strategia anti-Dijkstra
    """
    plt.figure(figsize=(16, 14))
    plt.clf()

    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Colori dei nodi
    node_colors = []
    node_sizes = []
    for node, data in G.nodes(data=True):
        if data['node_type'] == 'weak':
            node_colors.append('lightgreen')
            node_sizes.append(600)
        elif data['node_type'] == 'power_mandatory':
            node_colors.append('red')
            node_sizes.append(1000)
        else:  # discretionary
            node_colors.append('orange')
            node_sizes.append(800)
    
    # Categorizza gli edge per tipo e costo
    wm_edges = []  # Weak-Mandatory
    wd_edges = []  # Weak-Discretionary
    dm_edges_high = []  # Discretionary-Mandatory alto costo
    dm_edges_low = []   # Discretionary-Mandatory basso costo
    other_edges = []
    
    weak_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'weak']
    mandatory_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'power_mandatory']
    discretionary_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'power_discretionary']
    
    weak_set = set(weak_nodes)
    mandatory_set = set(mandatory_nodes)
    discretionary_set = set(discretionary_nodes)
    
    for edge in G.edges():
        node1, node2 = edge
        weight = G[node1][node2]['weight']
        
        if (node1 in weak_set and node2 in mandatory_set) or (node1 in mandatory_set and node2 in weak_set):
            wm_edges.append(edge)
        elif (node1 in weak_set and node2 in discretionary_set) or (node1 in discretionary_set and node2 in weak_set):
            wd_edges.append(edge)
        elif (node1 in discretionary_set and node2 in mandatory_set) or (node1 in mandatory_set and node2 in discretionary_set):
            if weight >= 50:
                dm_edges_high.append(edge)
            else:
                dm_edges_low.append(edge)
        else:
            other_edges.append(edge)
    
    # Disegna gli edge con stili diversi
    nx.draw_networkx_edges(G, pos, edgelist=wm_edges, edge_color='darkgreen', width=3, alpha=0.7, style='solid')
    nx.draw_networkx_edges(G, pos, edgelist=wd_edges, edge_color='gray', width=1.5, alpha=0.5, style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=dm_edges_high, edge_color='darkred', width=4, alpha=0.8, style='solid')
    nx.draw_networkx_edges(G, pos, edgelist=dm_edges_low, edge_color='red', width=2, alpha=0.6, style='solid')
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='lightgray', width=1, alpha=0.4)
    
    # Disegna i nodi
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Etichette degli edge (solo per i più importanti)
    edge_labels = {}
    for edge in wm_edges + dm_edges_high:
        weight = G[edge[0]][edge[1]]['weight']
        edge_labels[edge] = weight
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Legenda dettagliata
    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color='lightgreen', label='Weak nodes')
    red_patch = mpatches.Patch(color='red', label='Mandatory nodes')
    orange_patch = mpatches.Patch(color='orange', label='Discretionary nodes')
    
    from matplotlib.lines import Line2D
    wm_line = Line2D([0], [0], color='darkgreen', linewidth=3, label='W-M (economici)')
    dm_high_line = Line2D([0], [0], color='darkred', linewidth=4, label='D-M (molto costosi ≥50)')
    wd_line = Line2D([0], [0], color='gray', linewidth=1.5, linestyle='dashed', label='W-D (medi)')
    
    plt.legend(handles=[green_patch, red_patch, orange_patch, wm_line, dm_high_line, wd_line], 
              loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.title(f"Grafo Anti-Dijkstra: {len(G.nodes())} nodi, {len(G.edges())} collegamenti\n"
              f"W-M diretti: {len(wm_edges)}, D-M costosi: {len(dm_edges_high)}")

    # Salva il grafo
    path_to_save = f"{path}{filename}.png"
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafo salvato come: {path_to_save}")


def analyze_graph_anti_dijkstra(G, weak_nodes, mandatory_nodes, discretionary_nodes):
    """
    Analizza il grafo per verificare che sia sfavorevole a Dijkstra
    """
    print(f"\n=== ANALISI GRAFO ANTI-DIJKSTRA ===")
    
    weak_set = set(weak_nodes)
    mandatory_set = set(mandatory_nodes)
    discretionary_set = set(discretionary_nodes)
    
    # Analisi dei costi medi per tipo di percorso
    wm_weights = []
    wd_weights = []
    dm_weights = []
    mm_weights = []
    dd_weights = []
    
    for edge in G.edges(data=True):
        node1, node2, data = edge
        weight = data['weight']
        
        if (node1 in weak_set and node2 in mandatory_set) or (node1 in mandatory_set and node2 in weak_set):
            wm_weights.append(weight)
        elif (node1 in weak_set and node2 in discretionary_set) or (node1 in discretionary_set and node2 in weak_set):
            wd_weights.append(weight)
        elif (node1 in mandatory_set and node2 in discretionary_set) or (node1 in discretionary_set and node2 in mandatory_set):
            dm_weights.append(weight)
        elif node1 in mandatory_set and node2 in mandatory_set:
            mm_weights.append(weight)
        elif node1 in discretionary_set and node2 in discretionary_set:
            dd_weights.append(weight)
    
    print(f"\n=== ANALISI COSTI MEDI PER TIPO DI CONNESSIONE ===")
    if wm_weights:
        print(f"Weak ↔ Mandatory: {len(wm_weights)} link, peso medio: {sum(wm_weights)/len(wm_weights):.1f}")
    if wd_weights:
        print(f"Weak ↔ Discretionary: {len(wd_weights)} link, peso medio: {sum(wd_weights)/len(wd_weights):.1f}")
    if dm_weights:
        print(f"Discretionary ↔ Mandatory: {len(dm_weights)} link, peso medio: {sum(dm_weights)/len(dm_weights):.1f}")
    if mm_weights:
        print(f"Mandatory ↔ Mandatory: {len(mm_weights)} link, peso medio: {sum(mm_weights)/len(mm_weights):.1f}")
    if dd_weights:
        print(f"Discretionary ↔ Discretionary: {len(dd_weights)} link, peso medio: {sum(dd_weights)/len(dd_weights):.1f}")
    
    # Stima costi percorsi tipici
    print(f"\n=== STIMA COSTI PERCORSI ===")
    print(f"Percorso Steiner (diretto W→M): ~{sum(wm_weights)/len(wm_weights) if wm_weights else 0:.1f} per connessione")
    
    dijkstra_cost = 0
    if wd_weights and dm_weights:
        avg_wd = sum(wd_weights)/len(wd_weights)
        avg_dm = sum(dm_weights)/len(dm_weights)
        dijkstra_cost = avg_wd + avg_dm/3  # Diviso per 3 perché il costo D-M è condiviso
        print(f"Percorso Dijkstra (W→D→M): ~{avg_wd:.1f} + {avg_dm/3:.1f} = ~{dijkstra_cost:.1f} per connessione")
    
    # Verifica connettività per weak nodes
    print(f"\n=== CONNETTIVITÀ WEAK NODES ===")
    wm_connections = 0
    wd_connections = 0
    for w in weak_nodes:
        for neighbor in G.neighbors(w):
            if neighbor in mandatory_set:
                wm_connections += 1
            elif neighbor in discretionary_set:
                wd_connections += 1
    
    print(f"Totale connessioni W→M dirette: {wm_connections}")
    print(f"Totale connessioni W→D: {wd_connections}")
    print(f"Rapporto W→M / W→D: {wm_connections/wd_connections if wd_connections > 0 else 'inf':.2f}")


if __name__ == "__main__":
    start_time = time.time()

    # PARAMETRI CONFIGURABILI
    total_nodes = 50
    mandatory_percentage = 10    # 10% di nodi mandatory
    discretionary_percentage = 10 # 10% di nodi discretionary
    # I rimanenti 80% saranno weak nodes

    # Genera il grafo anti-Dijkstra
    graph, weak_nodes, mandatory_nodes, discretionary_nodes = create_graph_topology_against_dijkstra(
        total_nodes, mandatory_percentage, discretionary_percentage
    )

    # Analizza il grafo
    analyze_graph_anti_dijkstra(graph, weak_nodes, mandatory_nodes, discretionary_nodes)

    # Disegna e salva il grafo
    if debug_plot:
        draw_and_save_graph(graph, "anti_dijkstra_graph")

    # Salva il grafo in formato pickle
    import pickle
    graph_file = os.path.join(path, "grafo_3.pickle")
    with open(graph_file, "wb") as f:
        pickle.dump(graph, f)
    print(f"\nGrafo salvato come: {graph_file}")

    # Verifica caricamento
    with open(graph_file, "rb") as f:
        loaded_graph = pickle.load(f)
    print(f"Verifica caricamento: {nx.is_isomorphic(graph, loaded_graph)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTempo di esecuzione: {elapsed_time:.4f} secondi")

    print("\n=== STRATEGIA ANTI-DIJKSTRA ===")
    print("✅ Connessioni W→M dirette economiche (peso 1-4)")
    print("✅ Connessioni W→D costose quanto W→M (peso 4-7)")
    print("✅ Connessioni D→M MOLTO costose (peso 50-120)")
    print("✅ Connessioni D→D costose (peso 25-40)")
    print("✅ Backbone mandatory efficiente (peso 1-3)")
    print("\n🎯 RISULTATO ATTESO:")
    print("- Steiner userà le connessioni dirette W→M economiche")
    print("- Dijkstra pagherà molto per usare i discretionary come hub")
    print("- Il costo totale di Dijkstra sarà superiore a Steiner")
