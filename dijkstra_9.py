import networkx as nx
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import pickle
import os
import datetime
import json
import time

# Add matplotlib backend to avoid display errors
import matplotlib
matplotlib.use('Agg')

# Path for plots - CORREZIONE: assicurati che sia una stringa
script_dir = os.path.dirname(os.path.abspath(__file__))
graphs_path = os.path.join(script_dir, 'graphs')  # Rinominato per evitare conflitti

# Crea la directory se non esiste
if not os.path.exists(graphs_path):
    os.makedirs(graphs_path)

class DijkstraSolution:
    """
    Classe per memorizzare i risultati dell'algoritmo di Dijkstra
    """
    def __init__(self, graph, shortest_paths, distances, predecessor_matrix,
                 weak_nodes, mandatory_nodes, discretionary_nodes,
                 power_capacities, alpha=0.5):
        self.graph = graph
        self.shortest_paths = shortest_paths  # Dict di dict con i percorsi minimi
        self.distances = distances  # Matrice delle distanze
        self.predecessor_matrix = predecessor_matrix  # Matrice dei predecessori
        self.weak_nodes = weak_nodes
        self.mandatory_nodes = mandatory_nodes
        self.discretionary_nodes = discretionary_nodes
        self.power_capacities = power_capacities
        self.alpha = alpha
        self.timestamp = datetime.datetime.now().isoformat()

        # Calcola statistiche
        self.total_paths = len(shortest_paths)
        self.average_distance = self._calculate_average_distance()
        self.connectivity_matrix = self._build_connectivity_matrix()

    def _calculate_average_distance(self):
        """Calcola la distanza media tra tutte le coppie di nodi"""
        total_distance = 0
        count = 0
        for source in self.distances:
            for target, dist in self.distances[source].items():
                if dist != float('inf') and source != target:
                    total_distance += dist
                    count += 1
        return total_distance / count if count > 0 else float('inf')

    def _build_connectivity_matrix(self):
        """Costruisce una matrice di connettività"""
        matrix = {}
        for source in self.graph.nodes():
            matrix[source] = {}
            for target in self.graph.nodes():
                if source == target:
                    matrix[source][target] = True
                else:
                    matrix[source][target] = self.distances.get(source, {}).get(target, float('inf')) != float('inf')
        return matrix

    def get_path(self, source, target):
        """Restituisce il percorso minimo da source a target"""
        if source not in self.shortest_paths or target not in self.shortest_paths[source]:
            return None
        return self.shortest_paths[source][target]

    def get_distance(self, source, target):
        """Restituisce la distanza minima da source a target"""
        if source not in self.distances or target not in self.distances[source]:
            return float('inf')
        return self.distances[source][target]

    def get_weak_to_power_paths(self):
        """Restituisce tutti i percorsi minimi dai nodi weak ai nodi power"""
        paths = {}
        all_power_nodes = self.mandatory_nodes + self.discretionary_nodes

        for weak in self.weak_nodes:
            paths[weak] = {}
            for power in all_power_nodes:
                if self.get_distance(weak, power) != float('inf'):
                    paths[weak][power] = {
                        'path': self.get_path(weak, power),
                        'distance': self.get_distance(weak, power)
                    }
        return paths

def dijkstra_single_source(graph, source):
    """
    Implementazione standard di Dijkstra per un singolo nodo sorgente.
    Restituisce distanze e predecessori per ricostruire i percorsi.
    """
    # Inizializzazione
    distances = {node: float('inf') for node in graph.nodes()}
    distances[source] = 0
    predecessors = {node: None for node in graph.nodes()}

    # Priority queue: (distanza, nodo)
    pq = [(0, source)]
    visited = set()

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)

        # Esplora i vicini
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_dist + weight

            # Se troviamo un percorso più breve
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return distances, predecessors

def reconstruct_path(predecessors, source, target):
    """
    Ricostruisce il percorso da source a target usando la matrice dei predecessori
    """
    if predecessors[target] is None and source != target:
        return None  # Non esiste percorso

    path = []
    current = target

    while current is not None:
        path.append(current)
        current = predecessors[current]
        if current == source:
            path.append(source)
            break

    path.reverse()
    return path if path[0] == source else None

def dijkstra_all_pairs(graph):
    """
    Esegue Dijkstra da ogni nodo per trovare i percorsi minimi tra TUTTE le coppie.
    Restituisce:
    - shortest_paths: dizionario di dizionari con i percorsi
    - distances: dizionario di dizionari con le distanze
    - predecessor_matrix: dizionario di dizionari con i predecessori
    """
    shortest_paths = {}
    distances = {}
    predecessor_matrix = {}

    total_nodes = len(graph.nodes())
    print(f"\n🔄 Esecuzione Dijkstra per {total_nodes} nodi...")
    print(f"   Questo calcolerà {total_nodes * (total_nodes - 1)} percorsi minimi")

    # Esegui Dijkstra da ogni nodo
    for i, source in enumerate(graph.nodes()):
        print(f"   📍 Elaborazione nodo {source} ({i+1}/{total_nodes})...", end='\r')

        # Dijkstra dal nodo corrente
        dist, pred = dijkstra_single_source(graph, source)

        distances[source] = dist
        predecessor_matrix[source] = pred
        shortest_paths[source] = {}

        # Ricostruisci tutti i percorsi da questo source
        for target in graph.nodes():
            if source != target:
                path = reconstruct_path(pred, source, target)
                if path:
                    shortest_paths[source][target] = path

    print(f"\n   ✅ Completato! Calcolati tutti i percorsi minimi.")

    return shortest_paths, distances, predecessor_matrix

def analyze_dijkstra_results(solution):
    """
    Analizza i risultati di Dijkstra e stampa statistiche
    """
    print(f"\n📊 ANALISI RISULTATI DIJKSTRA:")
    print(f"{'='*60}")

    # Statistiche generali
    print(f"\n🔍 Statistiche Generali:")
    print(f"   - Nodi totali: {len(solution.graph.nodes())}")
    print(f"   - Archi totali: {len(solution.graph.edges())}")
    print(f"   - Percorsi calcolati: {solution.total_paths}")
    print(f"   - Distanza media: {solution.average_distance:.2f}")

    # Analisi per tipo di nodo
    print(f"\n🔍 Analisi per Tipo di Nodo:")
    print(f"   - Nodi weak: {len(solution.weak_nodes)}")
    print(f"   - Nodi mandatory: {len(solution.mandatory_nodes)}")
    print(f"   - Nodi discretionary: {len(solution.discretionary_nodes)}")

    # Percorsi weak -> power
    weak_to_power = solution.get_weak_to_power_paths()
    total_weak_power_paths = sum(len(paths) for paths in weak_to_power.values())

    print(f"\n🔍 Percorsi Weak → Power:")
    print(f"   - Totale percorsi possibili: {total_weak_power_paths}")

    # Trova il percorso weak->power più corto e più lungo
    min_dist = float('inf')
    max_dist = 0
    min_path = None
    max_path = None

    for weak, power_paths in weak_to_power.items():
        for power, path_info in power_paths.items():
            dist = path_info['distance']
            if dist < min_dist:
                min_dist = dist
                min_path = (weak, power, path_info['path'])
            if dist > max_dist:
                max_dist = dist
                max_path = (weak, power, path_info['path'])

    if min_path:
        print(f"   - Percorso più corto: {min_path[0]} → {min_path[1]} (distanza: {min_dist})")
        print(f"     Percorso: {' → '.join(map(str, min_path[2]))}")

    if max_path:
        print(f"   - Percorso più lungo: {max_path[0]} → {max_path[1]} (distanza: {max_dist})")
        print(f"     Percorso: {' → '.join(map(str, max_path[2]))}")

    # Analisi connettività
    print(f"\n🔍 Analisi Connettività:")
    disconnected_pairs = []
    for source in solution.graph.nodes():
        for target in solution.graph.nodes():
            if source != target and not solution.connectivity_matrix[source][target]:
                disconnected_pairs.append((source, target))

    if disconnected_pairs:
        print(f"   ⚠️  Trovate {len(disconnected_pairs)} coppie di nodi non connessi!")
        print(f"   Esempi: {disconnected_pairs[:5]}")
    else:
        print(f"   ✅ Il grafo è completamente connesso!")

    return weak_to_power

def visualize_shortest_paths(graph, solution, node_pairs=None, save_name="dijkstra_paths"):
    """
    Visualizza alcuni percorsi minimi selezionati
    """
    plt.figure(figsize=(15, 12))

    # Layout del grafo
    pos = nx.spring_layout(graph, k=3, iterations=100)

    # Colori dei nodi
    node_colors = []
    for node in graph.nodes():
        if node in solution.weak_nodes:
            node_colors.append('lightgreen')
        elif node in solution.mandatory_nodes:
            node_colors.append('red')
        elif node in solution.discretionary_nodes:
            node_colors.append('orange')
        else:
            node_colors.append('gray')

    # Disegna il grafo base
    nx.draw(graph, pos, with_labels=True, node_color=node_colors,
            node_size=1500, font_size=10, font_color="black",
            alpha=0.3, edge_color='lightgray', width=0.5)

    # Se non sono specificate coppie, mostra alcuni esempi
    if node_pairs is None:
        # Mostra i percorsi da un nodo weak a tutti i mandatory
        if solution.weak_nodes and solution.mandatory_nodes:
            source = solution.weak_nodes[0]
            node_pairs = [(source, target) for target in solution.mandatory_nodes[:3]]

    # Colori per diversi percorsi
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Disegna i percorsi selezionati
    if node_pairs:
        for i, (source, target) in enumerate(node_pairs):
            path = solution.get_path(source, target)
            if path:
                # Disegna gli archi del percorso
                path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
                nx.draw_networkx_edges(graph, pos, edgelist=path_edges,
                                     edge_color=colors[i % len(colors)],
                                     width=3, alpha=0.8)

                # Evidenzia i nodi del percorso
                nx.draw_networkx_nodes(graph, pos, nodelist=path,
                                     node_color=colors[i % len(colors)],
                                     node_size=2000, alpha=0.6)

    # Titolo
    title = f"Dijkstra - Percorsi Minimi\n"
    if node_pairs:
        title += f"Esempi: {', '.join([f'{s}→{t}' for s, t in node_pairs[:3]])}"
    plt.title(title, fontsize=14, weight='bold')

    # Legenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
               markersize=12, label='Nodi Weak'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=12, label='Nodi Mandatory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=12, label='Nodi Discretionary'),
    ]

    # Aggiungi linee per i percorsi mostrati
    if node_pairs:
        for i, (s, t) in enumerate(node_pairs[:len(colors)]):
            if solution.get_path(s, t):
                legend_elements.append(
                    Line2D([0], [0], color=colors[i % len(colors)], linewidth=3,
                           label=f'Percorso {s}→{t}')
                )

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    # Usa il percorso corretto per salvare
    filepath = os.path.join(graphs_path, f"{save_name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n🎨 Visualizzazione salvata: {save_name}.png")

def save_dijkstra_solution(solution, filename=None):
    """
    Salva la soluzione Dijkstra in formato pickle
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dijkstra_solution_{timestamp}.pickle"

    filepath = os.path.join(graphs_path, filename)

    # Prepara i dati da salvare
    solution_data = {
        'algorithm': 'dijkstra_all_pairs',
        'timestamp': solution.timestamp,
        'graph_info': {
            'nodes': len(solution.graph.nodes()),
            'edges': len(solution.graph.edges()),
            'weak_nodes': solution.weak_nodes,
            'mandatory_nodes': solution.mandatory_nodes,
            'discretionary_nodes': solution.discretionary_nodes
        },
        'shortest_paths': solution.shortest_paths,
        'distances': solution.distances,
        'predecessor_matrix': solution.predecessor_matrix,
        'statistics': {
            'total_paths': solution.total_paths,
            'average_distance': solution.average_distance,
        },
        'power_capacities': solution.power_capacities,
        'alpha': solution.alpha
    }

    # Salva in pickle
    with open(filepath, 'wb') as f:
        pickle.dump(solution_data, f)

    print(f"\n💾 Soluzione Dijkstra salvata: {filename}")
    print(f"   📁 Path completo: {filepath}")

    return filepath

def save_paths_summary(solution, filename=None):
    """
    Salva un riepilogo testuale dei percorsi minimi
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dijkstra_paths_summary_{timestamp}.txt"

    filepath = os.path.join(graphs_path, filename)

    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DIJKSTRA - RIEPILOGO PERCORSI MINIMI\n")
        f.write("="*80 + "\n\n")

        f.write(f"Data analisi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nodi totali: {len(solution.graph.nodes())}\n")
        f.write(f"Distanza media: {solution.average_distance:.2f}\n\n")

        # Percorsi weak -> power
        f.write("PERCORSI WEAK → POWER:\n")
        f.write("-"*50 + "\n")

        weak_to_power = solution.get_weak_to_power_paths()
        for weak in solution.weak_nodes:
            f.write(f"\nDa nodo weak {weak}:\n")

            # Ordina per distanza
            if weak in weak_to_power:
                sorted_targets = sorted(weak_to_power[weak].items(),
                                      key=lambda x: x[1]['distance'])

                for power, path_info in sorted_targets:
                    path = path_info['path']
                    dist = path_info['distance']
                    path_str = ' → '.join(map(str, path))

                    node_type = "mandatory" if power in solution.mandatory_nodes else "discretionary"
                    f.write(f"  → {power} ({node_type}): distanza={dist}, percorso=[{path_str}]\n")

        # Matrice delle distanze (solo prime 10x10 per brevità)
        f.write("\n\nMATRICE DELLE DISTANZE (estratto 10x10):\n")
        f.write("-"*50 + "\n")

        nodes_sample = list(solution.graph.nodes())[:10]

        # Header
        f.write("     ")
        for node in nodes_sample:
            f.write(f"{str(node):>6}")
        f.write("\n")

        # Righe
        for source in nodes_sample:
            f.write(f"{str(source):>4} ")
            for target in nodes_sample:
                dist = solution.get_distance(source, target)
                if dist == float('inf'):
                    f.write("   INF")
                else:
                    f.write(f"{dist:>6.1f}")
            f.write("\n")

    print(f"📋 Riepilogo percorsi salvato: {filename}")
    return filepath

def main():
    """
    Funzione principale
    """
    print("🚀 DIJKSTRA - Calcolo Percorsi Minimi per Tutte le Coppie di Nodi")
    print("="*60)

    # Carica il grafo
    graph_index = 3  # Modifica per caricare grafi diversi

    try:
        # file_name = os.path.join(graphs_path, f"grafo_{graph_index}.pickle")
        file_name = os.path.join(graphs_path, f"grafo_3.pickle")

        with open(file_name, "rb") as f:
            graph = pickle.load(f)
        print(f"✅ Caricato {file_name}")
    except FileNotFoundError:
        print(f"❌ Errore: File {file_name} non trovato!")
        return

    # Estrai nodi per tipo
    weak_nodes = []
    mandatory_nodes = []
    discretionary_nodes = []

    for node_name, data in graph.nodes(data=True):
        if data['node_type'] == "weak":
            weak_nodes.append(node_name)
        elif data['node_type'] == "power_mandatory":
            mandatory_nodes.append(node_name)
        elif data['node_type'] == "power_discretionary":
            discretionary_nodes.append(node_name)

    print(f"\n📊 Struttura del grafo:")
    print(f"   - Nodi weak: {len(weak_nodes)} {weak_nodes[:5]}{'...' if len(weak_nodes) > 5 else ''}")
    print(f"   - Nodi mandatory: {len(mandatory_nodes)} {mandatory_nodes}")
    print(f"   - Nodi discretionary: {len(discretionary_nodes)} {discretionary_nodes[:5]}{'...' if len(discretionary_nodes) > 5 else ''}")

    # Capacità dei nodi (per compatibilità con il formato originale)
    def generate_capacities_50_nodes():
        capacities = {}
        for i in range(1, 41):
            capacities[i] = 1
        for i in range(41, 46):
            capacities[i] = 3
        for i in range(46, 51):
            capacities[i] = 4
        return capacities

    power_capacities = generate_capacities_50_nodes()

    # Esegui Dijkstra
    print("\n⏱️  Inizio calcolo percorsi minimi...")
    start_time = time.time()

    shortest_paths, distances, predecessor_matrix = dijkstra_all_pairs(graph)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n⏱️  Calcolo completato in {execution_time:.3f} secondi")

    # Crea oggetto soluzione
    solution = DijkstraSolution(
        graph=graph,
        shortest_paths=shortest_paths,
        distances=distances,
        predecessor_matrix=predecessor_matrix,
        weak_nodes=weak_nodes,
        mandatory_nodes=mandatory_nodes,
        discretionary_nodes=discretionary_nodes,
        power_capacities=power_capacities,
        alpha=0.5  # Per compatibilità
    )

    # Analizza risultati
    weak_to_power = analyze_dijkstra_results(solution)

    # Visualizza alcuni percorsi esempio
    if weak_nodes and mandatory_nodes:
        # Mostra percorsi dal primo weak a primi 3 mandatory
        example_pairs = [(weak_nodes[0], m) for m in mandatory_nodes[:3]]
        visualize_shortest_paths(graph, solution, example_pairs, "dijkstra_example_paths")

    # Salva risultati
    pickle_file = save_dijkstra_solution(solution)
    summary_file = save_paths_summary(solution)

    print(f"\n🏁 COMPLETATO!")
    print(f"   - Percorsi calcolati: {solution.total_paths}")
    print(f"   - File pickle: {pickle_file}")
    print(f"   - Riepilogo testuale: {summary_file}")

    # Esempio di utilizzo
    print(f"\n💡 Esempio di utilizzo:")
    print(f"   import pickle")
    print(f"   with open('{os.path.basename(pickle_file)}', 'rb') as f:")
    print(f"       data = pickle.load(f)")
    print(f"   # Percorso da nodo 1 a nodo 5:")
    print(f"   path = data['shortest_paths'][1][5]")
    print(f"   distance = data['distances'][1][5]")
    print(f"\n⏱️  Calcolo completato in {execution_time:.3f} secondi")

if __name__ == "__main__":
    main()
