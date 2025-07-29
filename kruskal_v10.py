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

# Path for plots
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'graphs/')

# Ensure directory exists
if not os.path.exists(path):
    os.makedirs(path)

class KruskalSolution:
    """
    Classe per memorizzare i risultati dell'algoritmo di Kruskal MST
    """
    def __init__(self, mst_tree, capacity_usage, connected_weak, failed_connections,
                 total_cost, capacity_cost, discretionary_used, graph_info="",
                 acc_cost=0, aoc_cost=0, alpha=0.5, all_edges_sorted=None,
                 power_capacities=None):
        self.mst_tree = mst_tree  # Il Minimum Spanning Tree risultante
        self.capacity_usage = capacity_usage
        self.connected_weak = connected_weak
        self.failed_connections = failed_connections
        self.total_cost = total_cost
        self.capacity_cost = capacity_cost
        self.discretionary_used = discretionary_used
        self.graph_info = graph_info
        self.acc_cost = acc_cost
        self.aoc_cost = aoc_cost
        self.alpha = alpha
        self.all_edges_sorted = all_edges_sorted  # Lista di tutti gli archi ordinati per peso
        self.power_capacities = power_capacities  # Aggiungi questo

        # Calculate overall score
        self.score = self.calculate_score()

    def calculate_cost_function(self, graph, selected_edges, selected_nodes, alpha=0.5):
        """
        Calculate the custom cost function C(G) = α * ACC + (1-α) * AOC
        """
        n = len(graph.nodes())

        # Calculate ACC
        total_edge_weight = sum(graph[u][v]['weight'] for u, v in selected_edges)
        acc = total_edge_weight / (n * (n - 1)) if n > 1 else 0

        # Calculate AOC (simplified for MST)
        aoc = 0

        # Combined cost
        cost = alpha * acc + (1 - alpha) * aoc

        return cost, acc, aoc

    def calculate_score(self):
        """
        Calculate score for MST solution
        """
        # For MST, the score is primarily the total edge weight
        # since MST minimizes total cost
        return self.total_cost

class UnionFind:
    """Union-Find data structure for Kruskal's algorithm"""
    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by rank"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        return True

def kruskal_mst(graph):
    """
    Implementazione dell'algoritmo di Kruskal per trovare il Minimum Spanning Tree
    """
    # Step 1: Ordina tutti gli archi per peso
    edges = []
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1)
        edges.append((weight, u, v))

    edges.sort()  # Ordina per peso crescente

    print(f"\n📊 Kruskal MST Algorithm:")
    print(f"   Total edges in graph: {len(edges)}")
    print(f"   Nodes in graph: {len(graph.nodes())}")

    # Step 2: Inizializza Union-Find
    uf = UnionFind(graph.nodes())

    # Step 3: Costruisci MST
    mst = nx.Graph()
    total_weight = 0
    edges_added = 0

    print(f"\n🔨 Building MST:")
    for weight, u, v in edges:
        if uf.union(u, v):  # Se non crea un ciclo
            mst.add_edge(u, v, weight=weight)
            total_weight += weight
            edges_added += 1

            if edges_added <= 10 or edges_added % 10 == 0:  # Log primi 10 e poi ogni 10
                print(f"   Added edge {u}-{v} (weight: {weight}). Total edges: {edges_added}")

            # MST completo quando ha n-1 archi
            if edges_added == len(graph.nodes()) - 1:
                break

    print(f"\n✅ MST Complete:")
    print(f"   Edges in MST: {edges_added}")
    print(f"   Total weight: {total_weight}")

    return mst, edges

def analyze_mst_connectivity(graph, mst, weak_nodes, mandatory_nodes, discretionary_nodes):
    """
    Analizza come l'MST connette i diversi tipi di nodi
    """
    # Verifica quali nodi sono nell'MST
    nodes_in_mst = set(mst.nodes())

    # Analizza connettività per tipo
    weak_in_mst = [n for n in weak_nodes if n in nodes_in_mst]
    mandatory_in_mst = [n for n in mandatory_nodes if n in nodes_in_mst]
    discretionary_in_mst = [n for n in discretionary_nodes if n in nodes_in_mst]

    print(f"\n📊 MST Connectivity Analysis:")
    print(f"   Weak nodes: {len(weak_in_mst)}/{len(weak_nodes)}")
    print(f"   Mandatory nodes: {len(mandatory_in_mst)}/{len(mandatory_nodes)}")
    print(f"   Discretionary nodes: {len(discretionary_in_mst)}/{len(discretionary_nodes)}")

    # Trova percorsi nell'MST da weak a power nodes
    power_nodes = mandatory_nodes + discretionary_nodes
    weak_connectivity = {}

    for weak in weak_in_mst:
        weak_connectivity[weak] = {}
        for power in power_nodes:
            if power in nodes_in_mst:
                try:
                    path = nx.shortest_path(mst, weak, power)
                    length = nx.shortest_path_length(mst, weak, power, weight='weight')
                    weak_connectivity[weak][power] = {
                        'path': path,
                        'length': length
                    }
                except nx.NetworkXNoPath:
                    pass

    # Calcola capacity usage basato sui percorsi nell'MST
    capacity_usage = {node: 0 for node in mandatory_nodes + discretionary_nodes}

    # Per ogni weak node, trova il power node più vicino nell'MST
    for weak, connections in weak_connectivity.items():
        if connections:
            # Trova il power node più vicino
            closest_power = min(connections.items(), key=lambda x: x[1]['length'])
            power_node = closest_power[0]
            capacity_usage[power_node] = capacity_usage.get(power_node, 0) + 1

    return weak_connectivity, capacity_usage, discretionary_in_mst

def create_kruskal_solution(graph, weak_nodes, mandatory_nodes, discretionary_nodes,
                           power_capacities, alpha=0.5):
    """
    Crea una soluzione completa usando l'algoritmo di Kruskal
    """
    print(f"\n{'='*60}")
    print(f"KRUSKAL MST APPROACH")
    print(f"Alpha = {alpha}")
    print(f"{'='*60}")

    # Esegui Kruskal
    mst, all_edges_sorted = kruskal_mst(graph)

    # Analizza connettività
    weak_connectivity, capacity_usage, discretionary_used = analyze_mst_connectivity(
        graph, mst, weak_nodes, mandatory_nodes, discretionary_nodes
    )

    # Determina quali weak nodes sono connessi
    connected_weak = set()
    failed_connections = []

    for weak in weak_nodes:
        if weak in mst.nodes():
            # Verifica se c'è un percorso a un power node
            has_path = False
            for power in mandatory_nodes + discretionary_nodes:
                if power in mst.nodes():
                    try:
                        nx.shortest_path(mst, weak, power)
                        has_path = True
                        break
                    except nx.NetworkXNoPath:
                        continue

            if has_path:
                connected_weak.add(weak)
            else:
                failed_connections.append(weak)
        else:
            failed_connections.append(weak)

    # Calcola statistiche
    total_cost = sum(data['weight'] for _, _, data in mst.edges(data=True))

    # Calcola capacity cost
    capacity_cost = 0
    nodes_with_capacity = [n for n in capacity_usage if capacity_usage[n] > 0 and n in power_capacities]

    if nodes_with_capacity:
        capacity_ratios = []
        for node in nodes_with_capacity:
            if power_capacities.get(node, 0) > 0:
                ratio = capacity_usage[node] / power_capacities[node]
                capacity_ratios.append(ratio)

        capacity_cost = sum(capacity_ratios) / len(capacity_ratios) if capacity_ratios else 0

    # Crea soluzione
    solution = KruskalSolution(
        mst_tree=mst,
        capacity_usage=capacity_usage,
        connected_weak=connected_weak,
        failed_connections=failed_connections,
        total_cost=total_cost,
        capacity_cost=capacity_cost,
        discretionary_used=sorted(discretionary_used),
        graph_info="Kruskal MST",
        alpha=alpha,
        all_edges_sorted=all_edges_sorted,
        power_capacities=power_capacities  # Passa anche le capacità
    )

    # Calcola ACC e AOC
    selected_edges = list(mst.edges())
    selected_nodes = set(mst.nodes())

    _, acc, aoc = solution.calculate_cost_function(graph, selected_edges, selected_nodes, alpha)
    solution.acc_cost = acc
    solution.aoc_cost = aoc

    return solution

def visualize_kruskal_mst(graph, solution, weak_nodes, mandatory_nodes, discretionary_nodes,
                         save_name="kruskal_mst"):
    """
    Visualizza il Minimum Spanning Tree risultante
    """
    plt.figure(figsize=(18, 14))

    # Layout
    pos = nx.spring_layout(graph, k=3, iterations=100, seed=42)

    # Colori dei nodi
    node_colors = []
    node_sizes = []

    mst = solution.mst_tree
    power_capacities = solution.power_capacities  # Usa le capacità dalla soluzione

    for node in graph.nodes():
        if node not in mst.nodes():
            node_colors.append('lightgray')
            node_sizes.append(800)
        elif node in solution.failed_connections:
            node_colors.append('black')
            node_sizes.append(1500)
        elif node in solution.connected_weak:
            node_colors.append('lightgreen')
            node_sizes.append(1800)
        elif node in weak_nodes:
            node_colors.append('darkgreen')
            node_sizes.append(1600)
        elif node in mandatory_nodes:
            node_colors.append('red')
            node_sizes.append(2000)
        elif node in solution.discretionary_used:
            node_colors.append('orange')
            node_sizes.append(1800)
        elif node in discretionary_nodes:
            node_colors.append('lightyellow')
            node_sizes.append(1400)
        else:
            node_colors.append('gray')
            node_sizes.append(1200)

    # Disegna grafo base (molto leggero)
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=0.5, alpha=0.3)

    # Disegna nodi
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)

    # Etichette nodi
    labels = {}
    for node in graph.nodes():
        if node in weak_nodes:
            labels[node] = f"W{node}"
        elif node in mandatory_nodes:
            labels[node] = f"M{node}"
        elif node in discretionary_nodes:
            labels[node] = f"D{node}"
        else:
            labels[node] = str(node)

    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_weight='bold')

    # Evidenzia archi MST
    if mst.edges():
        nx.draw_networkx_edges(mst, pos, edge_color='blue', width=4, alpha=0.8)

        # Pesi degli archi MST
        edge_labels = {}
        for u, v in mst.edges():
            if graph.has_edge(u, v):
                edge_labels[(u, v)] = graph[u][v]['weight']

        nx.draw_networkx_edge_labels(mst, pos, edge_labels, font_size=9,
                                    font_color='blue', font_weight='bold',
                                    bbox=dict(boxstyle="round,pad=0.3",
                                             facecolor="lightblue",
                                             edgecolor="blue",
                                             alpha=0.9))

    # Capacity labels per power nodes
    capacity_labels = {}
    for node in mandatory_nodes + discretionary_nodes:
        if node in mst.nodes():
            used = solution.capacity_usage.get(node, 0)
            max_cap = power_capacities.get(node, 0)
            capacity_labels[node] = f"{used}/{max_cap}"

    capacity_pos = {node: (pos[node][0], pos[node][1] - 0.15) for node in capacity_labels}

    for node, label in capacity_labels.items():
        color = "yellow" if solution.capacity_usage.get(node, 0) > 0 else "lightgray"
        plt.text(capacity_pos[node][0], capacity_pos[node][1], label,
                fontsize=9, ha='center', va='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                         alpha=0.9, edgecolor='black'))

    # Titolo
    title = (f"Kruskal Minimum Spanning Tree\n"
             f"Total Weight: {solution.total_cost} | "
             f"Edges: {mst.number_of_edges()} | "
             f"Connected Weak: {len(solution.connected_weak)}/{len(weak_nodes)}")
    plt.title(title, fontsize=16, weight='bold', pad=20)

    # Legenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
               markersize=12, label='Connected weak (W)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=12, label='Mandatory (M)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=12, label='Discretionary used (D)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
               markersize=10, label='Not in MST'),
        Line2D([0], [0], color='blue', linewidth=4, label='MST edges')
    ]

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)

    # Statistiche
    stats_text = (f"MST Statistics:\n"
                  f"Total nodes: {mst.number_of_nodes()}\n"
                  f"Total edges: {mst.number_of_edges()}\n"
                  f"Total weight: {solution.total_cost}\n"
                  f"Avg edge weight: {solution.total_cost/mst.number_of_edges():.2f}")

    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.axis('off')
    plt.tight_layout()

    filepath = os.path.join(path, f"{save_name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n🎨 MST visualization saved: {save_name}.png")

def save_kruskal_summary(solution, save_name="kruskal_summary"):
    """
    Salva un riepilogo testuale della soluzione Kruskal
    """
    filepath = os.path.join(path, f"{save_name}.txt")

    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("KRUSKAL MINIMUM SPANNING TREE - SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Algorithm: Kruskal's MST\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("MST STATISTICS:\n")
        f.write("-"*30 + "\n")
        f.write(f"Total nodes in MST: {solution.mst_tree.number_of_nodes()}\n")
        f.write(f"Total edges in MST: {solution.mst_tree.number_of_edges()}\n")
        f.write(f"Total weight: {solution.total_cost}\n")
        f.write(f"Average edge weight: {solution.total_cost/solution.mst_tree.number_of_edges():.2f}\n\n")

        f.write("NODE CONNECTIVITY:\n")
        f.write("-"*30 + "\n")
        f.write(f"Connected weak nodes: {len(solution.connected_weak)}\n")
        f.write(f"Failed connections: {len(solution.failed_connections)}\n")
        f.write(f"Discretionary nodes used: {len(solution.discretionary_used)}\n")
        f.write(f"Discretionary nodes list: {solution.discretionary_used}\n\n")

        f.write("CAPACITY USAGE:\n")
        f.write("-"*30 + "\n")
        for node, usage in sorted(solution.capacity_usage.items()):
            if usage > 0:
                max_cap = solution.power_capacities.get(node, 0)
                f.write(f"Node {node}: {usage}/{max_cap} connections\n")

        f.write("\nMST EDGES (sorted by weight):\n")
        f.write("-"*30 + "\n")

        # Ordina gli archi dell'MST per peso
        mst_edges = []
        for u, v, data in solution.mst_tree.edges(data=True):
            mst_edges.append((data['weight'], u, v))
        mst_edges.sort()

        for weight, u, v in mst_edges[:20]:  # Mostra primi 20
            f.write(f"{u} -- {v}: weight = {weight}\n")

        if len(mst_edges) > 20:
            f.write(f"... and {len(mst_edges) - 20} more edges\n")

    print(f"📋 Summary saved: {save_name}.txt")
    return filepath

def main():
    """
    Funzione principale per Kruskal MST
    """
    print("🚀 KRUSKAL MINIMUM SPANNING TREE")
    print("="*60)

    # Parametri
    graph_index = 3
    alpha = 0.5

    # Carica grafo
    try:
        file_name = os.path.join(path, f"grafo_{graph_index}.pickle")
        with open(file_name, "rb") as f:
            graph = pickle.load(f)
        print(f"✅ Loaded {file_name}")
    except FileNotFoundError:
        print(f"❌ Error: File {file_name} not found!")
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

    print(f"\n📊 Graph structure:")
    print(f"   - Total nodes: {len(graph.nodes())}")
    print(f"   - Total edges: {len(graph.edges())}")
    print(f"   - Weak nodes: {len(weak_nodes)}")
    print(f"   - Mandatory nodes: {len(mandatory_nodes)}")
    print(f"   - Discretionary nodes: {len(discretionary_nodes)}")

    # Capacità dei nodi
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

    # Timer
    print("\n⏱️  Starting Kruskal MST algorithm...")
    start_time = time.time()

    # Crea soluzione Kruskal
    solution = create_kruskal_solution(
        graph, weak_nodes, mandatory_nodes, discretionary_nodes,
        power_capacities, alpha
    )

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n⏱️  Execution completed in {execution_time:.3f} seconds")

    # Visualizza e salva risultati
    visualize_kruskal_mst(graph, solution, weak_nodes, mandatory_nodes,
                         discretionary_nodes, "kruskal_mst_result")

    summary_file = save_kruskal_summary(solution, "kruskal_mst_summary")

    # Salva soluzione in pickle
    pickle_filename = f"kruskal_mst_solution_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pickle"
    pickle_filepath = os.path.join(path, pickle_filename)

    solution_data = {
        'algorithm': 'kruskal_mst',
        'mst_tree': solution.mst_tree,
        'graph_info': {
            'total_nodes': len(graph.nodes()),
            'total_edges': len(graph.edges()),
            'weak_nodes': weak_nodes,
            'mandatory_nodes': mandatory_nodes,
            'discretionary_nodes': discretionary_nodes
        },
        'solution_stats': {
            'total_weight': solution.total_cost,
            'mst_edges': solution.mst_tree.number_of_edges(),
            'connected_weak': len(solution.connected_weak),
            'failed_connections': len(solution.failed_connections),
            'discretionary_used': solution.discretionary_used,
            'capacity_usage': solution.capacity_usage
        },
        'execution_time': execution_time,
        'timestamp': datetime.datetime.now().isoformat(),
        'power_capacities': power_capacities
    }

    with open(pickle_filepath, 'wb') as f:
        pickle.dump(solution_data, f)

    print(f"\n💾 Solution saved: {pickle_filename}")

    print(f"\n🏁 KRUSKAL MST COMPLETE!")
    print(f"   - MST weight: {solution.total_cost}")
    print(f"   - MST edges: {solution.mst_tree.number_of_edges()}")
    print(f"   - Files saved:")
    print(f"     • Visualization: kruskal_mst_result.png")
    print(f"     • Summary: {summary_file}")
    print(f"     • Pickle: {pickle_filename}")

if __name__ == "__main__":
    main()
