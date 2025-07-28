import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

class LatencyAnalyzer:
    """
    Analizzatore per calcolare latenze tra nodi mandatory e weak in alberi Steiner
    """

    def __init__(self):
        self.trees = {}  # Dizionario per memorizzare gli alberi caricati
        self.original_graphs = {}  # Grafici originali se disponibili

    def load_pickle_file(self, file_path, identifier):
        """
        Carica un file pickle contenente l'albero/soluzione

        Args:
            file_path (str): Percorso al file pickle
            identifier (str): Identificatore per questo albero (es. "tree_1", "tree_2")
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Gestisce diversi formati di dati pickle
            if isinstance(data, dict):
                # Se è un dizionario (formato dal tuo codice)
                if 'solution_tree' in data:
                    tree = data['solution_tree']
                    metadata = data
                else:
                    # Potrebbe essere un grafo salvato direttamente
                    tree = data
                    metadata = {}
            elif isinstance(data, nx.Graph):
                # Se è direttamente un grafo NetworkX
                tree = data
                metadata = {}
            else:
                raise ValueError(f"Formato file non riconosciuto: {type(data)}")

            self.trees[identifier] = {
                'tree': tree,
                'metadata': metadata,
                'file_path': file_path
            }

            print(f"✅ Caricato {identifier}: {len(tree.nodes())} nodi, {len(tree.edges())} archi")
            self._analyze_tree_structure(identifier)

        except Exception as e:
            print(f"❌ Errore nel caricamento di {file_path}: {e}")
            return False

        return True

    def _analyze_tree_structure(self, identifier):
        """Analizza la struttura dell'albero caricato"""
        tree_data = self.trees[identifier]
        tree = tree_data['tree']

        # Conta i tipi di nodi
        node_types = defaultdict(list)

        for node, data in tree.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_types[node_type].append(node)

        # Se non ci sono metadati sui tipi, prova a inferirli dalla struttura
        if not node_types or all(nt == 'unknown' for nt in node_types.keys()):
            print(f"   ⚠️  Tipi di nodi non trovati nei metadati, inferendo dalla struttura...")
            # Qui potresti aggiungere logica per inferire i tipi di nodi
            # Per ora, elenca tutti i nodi come "unknown"
            node_types = {'unknown': list(tree.nodes())}

        tree_data['node_types'] = dict(node_types)

        print(f"   📊 Struttura nodi:")
        for node_type, nodes in node_types.items():
            print(f"      - {node_type}: {len(nodes)} nodi → {nodes}")

    def get_available_mandatory_nodes(self, identifier):
        """
        Restituisce la lista dei nodi mandatory disponibili
        """
        if identifier not in self.trees:
            print(f"❌ Albero {identifier} non trovato")
            return []

        node_types = self.trees[identifier]['node_types']
        mandatory_nodes = node_types.get('power_mandatory', [])

        if not mandatory_nodes:
            # Se non ci sono nodi mandatory, mostra tutti i nodi disponibili
            print(f"⚠️  Nessun nodo 'power_mandatory' trovato in {identifier}")
            print(f"   Nodi disponibili: {list(self.trees[identifier]['tree'].nodes())}")
            return list(self.trees[identifier]['tree'].nodes())

        return mandatory_nodes

    def get_weak_nodes(self, identifier):
        """
        Restituisce la lista dei nodi weak
        """
        if identifier not in self.trees:
            return []

        node_types = self.trees[identifier]['node_types']
        return node_types.get('weak', [])

    def calculate_latencies(self, identifier, mandatory_node):
        """
        Calcola le latenze dal nodo mandatory a tutti i nodi weak

        Args:
            identifier (str): Identificatore dell'albero
            mandatory_node: Nodo mandatory di partenza

        Returns:
            dict: {weak_node: latency} per tutti i nodi weak raggiungibili
        """
        if identifier not in self.trees:
            print(f"❌ Albero {identifier} non trovato")
            return {}

        tree = self.trees[identifier]['tree']
        weak_nodes = self.get_weak_nodes(identifier)

        if mandatory_node not in tree.nodes():
            print(f"❌ Nodo {mandatory_node} non trovato nell'albero {identifier}")
            return {}

        print(f"\n🔍 Calcolando latenze da {mandatory_node} a tutti i nodi weak in {identifier}...")

        latencies = {}
        unreachable = []

        for weak_node in weak_nodes:
            try:
                # Calcola il percorso più breve nell'albero
                if nx.has_path(tree, mandatory_node, weak_node):
                    path = nx.shortest_path(tree, mandatory_node, weak_node, weight='weight')

                    # Calcola la latenza totale sommando i pesi degli archi
                    total_latency = 0
                    for i in range(len(path) - 1):
                        edge_weight = tree[path[i]][path[i+1]].get('weight', 1)
                        total_latency += edge_weight

                    latencies[weak_node] = total_latency
                    print(f"   ✓ {weak_node}: latenza = {total_latency} (percorso: {' → '.join(map(str, path))})")
                else:
                    unreachable.append(weak_node)
                    print(f"   ✗ {weak_node}: NON RAGGIUNGIBILE")

            except Exception as e:
                print(f"   ❌ Errore nel calcolo per {weak_node}: {e}")
                unreachable.append(weak_node)

        if unreachable:
            print(f"   ⚠️  Nodi non raggiungibili: {unreachable}")

        return latencies

    def compare_latencies(self, mandatory_node, tree1_id="tree_1", tree2_id="tree_2"):
        """
        Confronta le latenze tra due alberi per lo stesso nodo mandatory
        """
        print(f"\n{'='*60}")
        print(f"CONFRONTO LATENZE: Nodo Mandatory {mandatory_node}")
        print(f"Albero 1: {tree1_id} vs Albero 2: {tree2_id}")
        print(f"{'='*60}")

        # Calcola latenze per entrambi gli alberi
        latencies_1 = self.calculate_latencies(tree1_id, mandatory_node)
        latencies_2 = self.calculate_latencies(tree2_id, mandatory_node)

        # Trova tutti i nodi weak presenti in entrambi
        all_weak_nodes = set(latencies_1.keys()) | set(latencies_2.keys())

        if not all_weak_nodes:
            print("❌ Nessun nodo weak trovato in entrambi gli alberi")
            return None

        # Crea DataFrame per il confronto
        comparison_data = []
        for weak_node in sorted(all_weak_nodes):
            lat_1 = latencies_1.get(weak_node, float('inf'))
            lat_2 = latencies_2.get(weak_node, float('inf'))

            # Calcola la differenza
            if lat_1 != float('inf') and lat_2 != float('inf'):
                diff = lat_2 - lat_1
                better_tree = tree1_id if lat_1 < lat_2 else tree2_id if lat_2 < lat_1 else "Equal"
            else:
                diff = None
                better_tree = tree1_id if lat_1 != float('inf') else tree2_id if lat_2 != float('inf') else "Both unreachable"

            comparison_data.append({
                'Nodo_Weak': weak_node,
                f'Latenza_{tree1_id}': lat_1 if lat_1 != float('inf') else 'N/R',
                f'Latenza_{tree2_id}': lat_2 if lat_2 != float('inf') else 'N/R',
                'Differenza': f"{diff:+.2f}" if diff is not None else "N/A",
                'Migliore': better_tree
            })

        df = pd.DataFrame(comparison_data)

        print(f"\n📊 TABELLA CONFRONTO LATENZE:")
        print(df.to_string(index=False))

        # Statistiche riassuntive
        valid_diffs = [float(row['Differenza']) for _, row in df.iterrows()
                      if row['Differenza'] != "N/A"]

        if valid_diffs:
            print(f"\n📈 STATISTICHE:")
            print(f"   - Differenza media: {sum(valid_diffs)/len(valid_diffs):+.2f}")
            print(f"   - Differenza massima: {max(valid_diffs):+.2f}")
            print(f"   - Differenza minima: {min(valid_diffs):+.2f}")

            tree1_wins = sum(1 for _, row in df.iterrows() if row['Migliore'] == tree1_id)
            tree2_wins = sum(1 for _, row in df.iterrows() if row['Migliore'] == tree2_id)
            ties = sum(1 for _, row in df.iterrows() if row['Migliore'] == "Equal")

            print(f"   - {tree1_id} migliore: {tree1_wins} nodi")
            print(f"   - {tree2_id} migliore: {tree2_wins} nodi")
            print(f"   - Pareggi: {ties} nodi")

        return df

    def visualize_latencies(self, mandatory_node, tree1_id="tree_1", tree2_id="tree_2", save_plot=True):
        """
        Crea un grafico di confronto delle latenze
        """
        latencies_1 = self.calculate_latencies(tree1_id, mandatory_node)
        latencies_2 = self.calculate_latencies(tree2_id, mandatory_node)

        # Prepara i dati per il grafico
        all_weak_nodes = sorted(set(latencies_1.keys()) | set(latencies_2.keys()))

        if not all_weak_nodes:
            print("❌ Nessun dato per la visualizzazione")
            return

        lat_1_values = [latencies_1.get(node, 0) for node in all_weak_nodes]
        lat_2_values = [latencies_2.get(node, 0) for node in all_weak_nodes]

        # Crea il grafico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Grafico a barre
        x = range(len(all_weak_nodes))
        width = 0.35

        ax1.bar([i - width/2 for i in x], lat_1_values, width, label=tree1_id, alpha=0.8)
        ax1.bar([i + width/2 for i in x], lat_2_values, width, label=tree2_id, alpha=0.8)

        ax1.set_xlabel('Nodi Weak')
        ax1.set_ylabel('Latenza')
        ax1.set_title(f'Confronto Latenze da Nodo Mandatory {mandatory_node}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_weak_nodes, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        valid_pairs = [(lat_1_values[i], lat_2_values[i]) for i in range(len(all_weak_nodes))
                      if lat_1_values[i] > 0 and lat_2_values[i] > 0]

        if valid_pairs:
            x_vals, y_vals = zip(*valid_pairs)
            ax2.scatter(x_vals, y_vals, alpha=0.7, s=100)

            # Linea di parità
            max_val = max(max(x_vals), max(y_vals))
            ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Parità')

            ax2.set_xlabel(f'Latenza {tree1_id}')
            ax2.set_ylabel(f'Latenza {tree2_id}')
            ax2.set_title('Scatter Plot Latenze')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Aggiungi etichette per i punti
            for i, (x_val, y_val) in enumerate(valid_pairs):
                ax2.annotate(all_weak_nodes[i], (x_val, y_val), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)

        plt.tight_layout()

        if save_plot:
            filename = f'latency_comparison_mandatory_{mandatory_node}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Grafico salvato: {filename}")

        plt.show()

    def interactive_analysis(self):
        """
        Modalità interattiva per l'analisi delle latenze
        """
        print(f"\n🔍 ANALISI INTERATTIVA LATENZE")
        print(f"{'='*40}")

        if len(self.trees) < 2:
            print("❌ Servono almeno 2 alberi caricati per il confronto")
            return

        # Mostra alberi disponibili
        tree_ids = list(self.trees.keys())
        print(f"📊 Alberi disponibili: {tree_ids}")

        # Prendi i primi due per default o chiedi all'utente
        tree1_id = tree_ids[0]
        tree2_id = tree_ids[1]

        print(f"📈 Confronto tra: {tree1_id} e {tree2_id}")

        # Mostra nodi mandatory disponibili
        mandatory_1 = self.get_available_mandatory_nodes(tree1_id)
        mandatory_2 = self.get_available_mandatory_nodes(tree2_id)
        common_mandatory = set(mandatory_1) & set(mandatory_2)

        if not common_mandatory:
            print("❌ Nessun nodo mandatory comune trovato")
            print(f"   {tree1_id}: {mandatory_1}")
            print(f"   {tree2_id}: {mandatory_2}")
            return

        print(f"⚡ Nodi mandatory comuni: {sorted(common_mandatory)}")

        # Analizza tutti i nodi mandatory comuni
        for mandatory_node in sorted(common_mandatory):
            print(f"\n{'='*50}")
            df = self.compare_latencies(mandatory_node, tree1_id, tree2_id)

            # Opzione per visualizzare il grafico
            create_viz = input(f"\nCreare visualizzazione per nodo {mandatory_node}? (y/n): ").strip().lower()
            if create_viz == 'y':
                self.visualize_latencies(mandatory_node, tree1_id, tree2_id)

        return True

def main():
    """
    Funzione principale per l'utilizzo dello script
    """
    analyzer = LatencyAnalyzer()

    print("🌳 ANALIZZATORE LATENZE ALBERI STEINER")
    print("="*50)

    # Carica i file pickle
    print("\n📁 Caricamento file pickle...")

    # Puoi modificare questi percorsi con i tuoi file
    file1_path = input("Inserisci il percorso del primo file pickle: ").strip()
    if not file1_path:
        # Percorso di default per test
        file1_path = "graphs/steiner_GRAPH_3_CUSTOM_COST_solution.pickle"

    file2_path = input("Inserisci il percorso del secondo file pickle: ").strip()
    if not file2_path:
        # Percorso di default per test
        file2_path = "graphs/dijkstra_solution_graph_3_alpha_0.5_20250728_202318.pickle"

    # Carica i file
    success1 = analyzer.load_pickle_file(file1_path, "tree_1")
    success2 = analyzer.load_pickle_file(file2_path, "tree_2")

    if not (success1 and success2):
        print("❌ Impossibile caricare entrambi i file")
        return

    # Avvia analisi interattiva
    analyzer.interactive_analysis()

# Funzioni di utilità per uso programmatico
def quick_latency_analysis(file1_path, file2_path, mandatory_node):
    """
    Funzione rapida per analizzare le latenze tra due file

    Usage:
        quick_latency_analysis("tree1.pickle", "tree2.pickle", 1)
    """
    analyzer = LatencyAnalyzer()

    analyzer.load_pickle_file(file1_path, "tree_1")
    analyzer.load_pickle_file(file2_path, "tree_2")

    df = analyzer.compare_latencies(mandatory_node, "tree_1", "tree_2")
    analyzer.visualize_latencies(mandatory_node, "tree_1", "tree_2")

    return df

if __name__ == "__main__":
    main()
