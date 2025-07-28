import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SimplePickleAnalyzer:
    """
    Analizzatore semplificato per file pickle di grafi/alberi
    """

    def __init__(self):
        self.data = {}

    def load_pickle(self, filepath, name=None):
        """
        Carica un file pickle e analizza il suo contenuto
        """
        if name is None:
            name = os.path.basename(filepath)

        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)

            print(f"\n{'='*60}")
            print(f"📁 File: {filepath}")
            print(f"📊 Nome: {name}")
            print(f"{'='*60}")

            # Analizza il tipo di dati
            print(f"📌 Tipo di dati: {type(loaded_data)}")

            if isinstance(loaded_data, dict):
                print(f"🔑 Chiavi disponibili: {list(loaded_data.keys())}")

                # Cerca il grafo/albero
                graph = None
                graph_key = None

                # Chiavi comuni per grafi
                possible_keys = ['steiner_tree', 'solution_tree', 'tree', 'graph', 'result']

                for key in possible_keys:
                    if key in loaded_data and isinstance(loaded_data[key], nx.Graph):
                        graph = loaded_data[key]
                        graph_key = key
                        break

                # Se non trovato, cerca qualsiasi grafo NetworkX
                if graph is None:
                    for key, value in loaded_data.items():
                        if isinstance(value, nx.Graph):
                            graph = value
                            graph_key = key
                            break

                if graph:
                    print(f"✅ Grafo trovato nella chiave: '{graph_key}'")
                    self._analyze_graph(graph, name)

                # Analizza altri dati importanti
                if 'solution_metadata' in loaded_data:
                    print(f"\n📊 Metadati soluzione:")
                    for k, v in loaded_data['solution_metadata'].items():
                        print(f"   - {k}: {v}")

                if 'alpha' in loaded_data:
                    print(f"\n⚡ Alpha: {loaded_data['alpha']}")
                if 'score' in loaded_data:
                    print(f"📈 Score: {loaded_data['score']}")
                if 'connected_weak' in loaded_data:
                    print(f"🔗 Nodi weak connessi: {len(loaded_data['connected_weak'])}")
                if 'discretionary_used' in loaded_data:
                    print(f"🎯 Nodi discretionary usati: {loaded_data['discretionary_used']}")

            elif isinstance(loaded_data, nx.Graph):
                print(f"✅ Il file contiene direttamente un grafo NetworkX")
                graph = loaded_data
                self._analyze_graph(graph, name)
            else:
                print(f"⚠️ Tipo di dati non riconosciuto")

            self.data[name] = loaded_data
            return loaded_data

        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
            return None

    def _analyze_graph(self, graph, name):
        """
        Analizza un grafo NetworkX
        """
        print(f"\n🌳 Analisi del grafo '{name}':")
        print(f"   - Numero di nodi: {graph.number_of_nodes()}")
        print(f"   - Numero di archi: {graph.number_of_edges()}")
        print(f"   - È connesso: {nx.is_connected(graph)}")

        # Analizza i tipi di nodi
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node)

        print(f"\n📍 Tipi di nodi:")
        for node_type, nodes in node_types.items():
            print(f"   - {node_type}: {len(nodes)} nodi → {nodes[:5]}{'...' if len(nodes) > 5 else ''}")

        # Calcola statistiche sui pesi degli archi
        if graph.number_of_edges() > 0:
            weights = []
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight', 1)
                weights.append(weight)

            print(f"\n⚖️ Statistiche pesi archi:")
            print(f"   - Min: {min(weights)}")
            print(f"   - Max: {max(weights)}")
            print(f"   - Media: {sum(weights)/len(weights):.2f}")
            print(f"   - Totale: {sum(weights)}")

    def compare_graphs(self, name1, name2):
        """
        Confronta due grafi caricati
        """
        if name1 not in self.data or name2 not in self.data:
            print("❌ Uno o entrambi i grafi non sono stati caricati")
            return

        data1 = self.data[name1]
        data2 = self.data[name2]

        # Estrai i grafi
        graph1 = self._extract_graph(data1)
        graph2 = self._extract_graph(data2)

        if not graph1 or not graph2:
            print("❌ Impossibile estrarre i grafi per il confronto")
            return

        print(f"\n{'='*60}")
        print(f"📊 CONFRONTO TRA {name1} e {name2}")
        print(f"{'='*60}")

        # Confronto base
        print(f"\n📈 Metriche base:")
        print(f"{'Metrica':<20} {name1:<15} {name2:<15} {'Differenza':<15}")
        print(f"{'-'*65}")

        metrics = {
            'Nodi': (graph1.number_of_nodes(), graph2.number_of_nodes()),
            'Archi': (graph1.number_of_edges(), graph2.number_of_edges()),
            'Connesso': (nx.is_connected(graph1), nx.is_connected(graph2))
        }

        for metric, (val1, val2) in metrics.items():
            diff = f"{val2 - val1:+d}" if isinstance(val1, int) else "-"
            print(f"{metric:<20} {str(val1):<15} {str(val2):<15} {diff:<15}")

        # Confronto nodi
        nodes1 = set(graph1.nodes())
        nodes2 = set(graph2.nodes())

        print(f"\n🔗 Confronto nodi:")
        print(f"   - Nodi comuni: {len(nodes1 & nodes2)}")
        print(f"   - Solo in {name1}: {len(nodes1 - nodes2)}")
        print(f"   - Solo in {name2}: {len(nodes2 - nodes1)}")

        # Confronto archi (se hanno nodi comuni)
        if nodes1 & nodes2:
            edges1 = set(graph1.edges())
            edges2 = set(graph2.edges())

            print(f"\n🔗 Confronto archi:")
            print(f"   - Archi comuni: {len(edges1 & edges2)}")
            print(f"   - Solo in {name1}: {len(edges1 - edges2)}")
            print(f"   - Solo in {name2}: {len(edges2 - edges1)}")

        # Confronto pesi totali
        if graph1.number_of_edges() > 0 and graph2.number_of_edges() > 0:
            weight1 = sum(data.get('weight', 1) for _, _, data in graph1.edges(data=True))
            weight2 = sum(data.get('weight', 1) for _, _, data in graph2.edges(data=True))

            print(f"\n⚖️ Pesi totali:")
            print(f"   - {name1}: {weight1}")
            print(f"   - {name2}: {weight2}")
            print(f"   - Differenza: {weight2 - weight1:+d} ({(weight2-weight1)/weight1*100:+.1f}%)")

    def _extract_graph(self, data):
        """
        Estrae un grafo NetworkX da vari formati di dati
        """
        if isinstance(data, nx.Graph):
            return data

        if isinstance(data, dict):
            # Cerca il grafo nelle chiavi comuni
            for key in ['steiner_tree', 'solution_tree', 'tree', 'graph']:
                if key in data and isinstance(data[key], nx.Graph):
                    return data[key]

            # Cerca qualsiasi grafo
            for value in data.values():
                if isinstance(value, nx.Graph):
                    return value

        return None

    def visualize_graph(self, name, save_path=None):
        """
        Visualizza un grafo caricato
        """
        if name not in self.data:
            print(f"❌ Grafo {name} non trovato")
            return

        graph = self._extract_graph(self.data[name])
        if not graph:
            print(f"❌ Impossibile estrarre il grafo da {name}")
            return

        plt.figure(figsize=(12, 8))

        # Layout
        pos = nx.spring_layout(graph, k=2, iterations=50)

        # Colori per tipo di nodo
        node_colors = []
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            if node_type == 'weak':
                node_colors.append('lightgreen')
            elif node_type == 'power_mandatory':
                node_colors.append('red')
            elif node_type == 'power_discretionary':
                node_colors.append('orange')
            else:
                node_colors.append('gray')

        # Disegna il grafo
        nx.draw(graph, pos, with_labels=True, node_color=node_colors,
                node_size=1000, font_size=10, font_weight='bold')

        # Etichette degli archi
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            edge_labels[(u, v)] = data.get('weight', 1)

        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=8)

        plt.title(f"Grafo: {name}")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Grafico salvato: {save_path}")

        plt.show()

    def export_summary(self, output_file="analysis_summary.txt"):
        """
        Esporta un riepilogo dell'analisi
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RIEPILOGO ANALISI FILE PICKLE\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            for name, data in self.data.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"FILE: {name}\n")
                f.write(f"{'='*60}\n")

                graph = self._extract_graph(data)
                if graph:
                    f.write(f"Nodi: {graph.number_of_nodes()}\n")
                    f.write(f"Archi: {graph.number_of_edges()}\n")
                    f.write(f"Connesso: {nx.is_connected(graph)}\n")

                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key not in ['steiner_tree', 'solution_tree', 'tree', 'graph']:
                                f.write(f"{key}: {value}\n")

                f.write("\n")

        print(f"✅ Riepilogo salvato: {output_file}")

# Funzioni di utilità per uso rapido
def analyze_pickle_files(file1, file2=None):
    """
    Funzione rapida per analizzare uno o due file pickle

    Uso:
        analyze_pickle_files("file1.pickle")
        analyze_pickle_files("file1.pickle", "file2.pickle")
    """
    analyzer = SimplePickleAnalyzer()

    # Carica il primo file
    data1 = analyzer.load_pickle(file1, "File_1")

    if file2:
        # Carica il secondo file
        data2 = analyzer.load_pickle(file2, "File_2")

        # Confronta i due file
        if data1 and data2:
            analyzer.compare_graphs("File_1", "File_2")

    return analyzer

# Esempio di utilizzo
if __name__ == "__main__":
    print("🔍 ANALIZZATORE FILE PICKLE")
    print("="*50)

    # Opzione 1: Analisi interattiva
    file1 = "graphs/dijkstra_solution_graph_3_alpha_0.5_20250728_202318.pickle"
    if not file1:
        print("❌ Nessun file specificato")
        exit()

    file2 = "graphs/steiner_GRAPH_3_CUSTOM_COST_solution.pickle"

    analyzer = SimplePickleAnalyzer()
    analyzer.load_pickle(file1, "File_1")

    if file2:
        analyzer.load_pickle(file2, "File_2")
        analyzer.compare_graphs("File_1", "File_2")

    # Visualizza?
    if input("\nVuoi visualizzare i grafi? (s/n): ").lower() == 's':
        analyzer.visualize_graph("File_1", "djistra_grafo_1.png")
        if file2:
            analyzer.visualize_graph("File_2", "steiner_grafo_2.png")

    # Esporta riepilogo
    if input("\nVuoi esportare un riepilogo? (s/n): ").lower() == 's':
        analyzer.export_summary()
