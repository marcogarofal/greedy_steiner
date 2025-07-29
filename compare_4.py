import pickle
import networkx as nx
import pandas as pd
import os
from datetime import datetime
import copy

class NodeRemovalAnalyzer:
    """
    Analizza l'impatto della rimozione di nodi intermedi dai percorsi
    tra nodi weak nelle soluzioni Dijkstra e Steiner
    MODIFICATO: Salva i grafi modificati in modo generico per entrambi gli algoritmi
    """

    def __init__(self):
        self.solutions = {}
        self.original_graph = None
        self.paths_info = {}

    def load_solution(self, filepath, name):
        """
        Carica una soluzione da file pickle
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            solution_info = {
                'filepath': filepath,
                'name': name,
                'data': data,
                'raw_data': copy.deepcopy(data)  # Mantieni copia originale
            }

            # Determina il tipo di algoritmo
            filename_lower = os.path.basename(filepath).lower()
            if 'dijkstra' in filename_lower or 'dijistra' in filename_lower:
                solution_info['algorithm'] = 'dijkstra'
            elif 'steiner' in filename_lower:
                solution_info['algorithm'] = 'steiner'
            else:
                metadata = data.get('solution_metadata', {})
                solution_info['algorithm'] = metadata.get('algorithm', 'unknown')

            # Estrai l'albero
            if 'steiner_tree' in data:
                solution_info['tree'] = data['steiner_tree']
                solution_info['tree_key'] = 'steiner_tree'
            elif 'dijistra_tree' in data:
                solution_info['tree'] = data['dijistra_tree']
                solution_info['tree_key'] = 'dijistra_tree'
            elif 'dijkstra_tree' in data:
                solution_info['tree'] = data['dijkstra_tree']
                solution_info['tree_key'] = 'dijkstra_tree'
            elif 'solution_tree' in data:
                solution_info['tree'] = data['solution_tree']
                solution_info['tree_key'] = 'solution_tree'
            else:
                print(f"⚠️ Nessun albero trovato in {name}")
                return False

            # Estrai nodi weak
            solution_info['weak_nodes'] = data.get('weak_nodes', [])
            solution_info['connected_weak'] = data.get('connected_weak_nodes', [])

            # Estrai altri tipi di nodi
            solution_info['mandatory_nodes'] = data.get('mandatory_nodes', [])
            solution_info['discretionary_nodes'] = data.get('discretionary_nodes', [])
            solution_info['discretionary_used'] = data.get('discretionary_used', [])

            self.solutions[name] = solution_info

            print(f"✅ Caricata soluzione '{name}'")
            print(f"   Algoritmo: {solution_info['algorithm']}")
            print(f"   Nodi weak connessi: {len(solution_info['connected_weak'])}")

            return True

        except Exception as e:
            print(f"❌ Errore nel caricamento di {filepath}: {e}")
            return False

    def load_original_graph(self, filepath):
        """
        Carica il grafo originale da file pickle
        """
        try:
            with open(filepath, 'rb') as f:
                self.original_graph = pickle.load(f)

            if isinstance(self.original_graph, nx.Graph):
                print(f"✅ Grafo originale caricato")
                print(f"   Nodi: {self.original_graph.number_of_nodes()}")
                print(f"   Archi: {self.original_graph.number_of_edges()}")
                return True
            else:
                print("❌ Il file non contiene un grafo NetworkX valido")
                return False

        except Exception as e:
            print(f"❌ Errore nel caricamento del grafo: {e}")
            return False

    def save_graph_with_removed_node(self, node_to_remove, output_dir="graphs2"):
        """
        Salva il grafo con il nodo rimosso nel formato generico
        """
        os.makedirs(output_dir, exist_ok=True)

        # Crea una copia del grafo originale
        modified_graph = copy.deepcopy(self.original_graph)

        # Rimuovi il nodo
        if node_to_remove in modified_graph:
            modified_graph.remove_node(node_to_remove)

            # Nome del file generico
            filename = f"grafo_without_node_{node_to_remove}.pickle"
            filepath = os.path.join(output_dir, filename)

            # Salva il grafo
            with open(filepath, 'wb') as f:
                pickle.dump(modified_graph, f)

            print(f"💾 Grafo salvato: {filename}")
            print(f"   Nodi: {modified_graph.number_of_nodes()}")
            print(f"   Archi: {modified_graph.number_of_edges()}")

            return filepath
        else:
            print(f"⚠️ Nodo {node_to_remove} non trovato nel grafo")
            return None

    def find_all_paths_between_weak_nodes(self, tree, weak_nodes):
        """
        Trova tutti i percorsi tra coppie di nodi weak nell'albero
        """
        paths = []
        weak_set = set(weak_nodes)

        # Per ogni coppia di nodi weak
        for i, node1 in enumerate(weak_nodes):
            for node2 in weak_nodes[i+1:]:
                if node1 in tree and node2 in tree:
                    try:
                        # Trova il percorso nell'albero
                        path = nx.shortest_path(tree, node1, node2)

                        # Conta i nodi intermedi (escludendo source e target)
                        intermediate_nodes = path[1:-1]

                        if intermediate_nodes:  # Solo se ci sono nodi intermedi
                            paths.append({
                                'source': node1,
                                'target': node2,
                                'path': path,
                                'intermediate_nodes': intermediate_nodes,
                                'hop_count': len(path) - 1,
                                'intermediate_count': len(intermediate_nodes)
                            })
                    except nx.NetworkXNoPath:
                        pass

        return sorted(paths, key=lambda x: x['intermediate_count'], reverse=True)

    def select_nodes_to_remove(self):
        """
        Seleziona i nodi da rimuovere basandosi sui percorsi più lunghi
        """
        all_intermediate_nodes = set()

        for name, sol in self.solutions.items():
            paths = self.find_all_paths_between_weak_nodes(
                sol['tree'],
                sol['connected_weak']
            )

            print(f"\n📊 {name}:")
            print(f"   Percorsi trovati: {len(paths)}")

            # Prendi i nodi intermedi dai percorsi più lunghi (top 5)
            for path in paths[:5]:
                all_intermediate_nodes.update(path['intermediate_nodes'])
                if len(paths) > 0 and paths.index(path) == 0:
                    print(f"   Percorso più lungo: {path['source']} -> {path['target']}")
                    print(f"   Nodi intermedi: {path['intermediate_nodes']}")

        return list(all_intermediate_nodes)

    def analyze_and_save_all(self):
        """
        Esegue l'analisi completa e salva i grafi modificati
        """
        print("\n🔍 ANALISI RIMOZIONE NODI")
        print("="*60)

        # Seleziona i nodi da rimuovere
        nodes_to_remove = self.select_nodes_to_remove()

        if not nodes_to_remove:
            print("\n❌ Nessun nodo intermedio trovato")
            return None

        print(f"\n📋 NODI DA RIMUOVERE ({len(nodes_to_remove)}):")
        print(f"   {nodes_to_remove}")

        # Salva i grafi modificati
        print(f"\n💾 SALVATAGGIO GRAFI MODIFICATI:")
        saved_graphs = []

        for node in nodes_to_remove:
            filepath = self.save_graph_with_removed_node(node)
            if filepath:
                saved_graphs.append({
                    'node_removed': node,
                    'filepath': filepath
                })

        # Crea report
        self.create_analysis_report(nodes_to_remove, saved_graphs)

        return {
            'nodes_removed': nodes_to_remove,
            'saved_graphs': saved_graphs
        }

    def create_analysis_report(self, nodes_removed, saved_graphs):
        """
        Crea un report dell'analisi
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"node_removal_report_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORT RIMOZIONE NODI - GRAFI MODIFICATI\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Nodi rimossi
            f.write(f"NODI RIMOSSI ({len(nodes_removed)}):\n")
            f.write("-"*40 + "\n")
            for i, node in enumerate(nodes_removed, 1):
                f.write(f"{i}. Nodo {node}\n")

            # File grafi salvati
            f.write(f"\n\nGRAFI MODIFICATI SALVATI ({len(saved_graphs)}):\n")
            f.write("-"*40 + "\n")
            for graph_info in saved_graphs:
                f.write(f"\nNodo rimosso: {graph_info['node_removed']}\n")
                f.write(f"  - File: {os.path.basename(graph_info['filepath'])}\n")
                f.write(f"  - Path completo: {graph_info['filepath']}\n")

            # Istruzioni
            f.write("\n\nISTRUZIONI PER L'USO:\n")
            f.write("-"*40 + "\n")
            f.write("1. I grafi modificati sono nella cartella 'graphs2'\n")
            f.write("2. I file sono pronti per essere usati con entrambi gli algoritmi\n")
            f.write("3. Per utilizzarli:\n")
            f.write("   - In paste.txt (Steiner): modifica il path da 'graphs' a 'graphs2'\n")
            f.write("   - In paste-2.txt (Dijkstra): modifica il path da 'graphs' a 'graphs2'\n")
            f.write("   - Carica il file 'grafo_without_node_X.pickle'\n")
            f.write("\nNOTA: Entrambi gli algoritmi (Steiner e Dijkstra) possono usare\n")
            f.write("      lo stesso grafo modificato per calcolare le loro soluzioni.\n")

        print(f"\n📄 Report salvato: {filename}")

# Funzione di utilità per uso rapido
def prepare_graphs_for_analysis(dijkstra_file, steiner_file, graph_file):
    """
    Funzione principale per preparare i grafi modificati
    """
    analyzer = NodeRemovalAnalyzer()

    # Carica i file
    print("📂 CARICAMENTO FILE...")
    if not analyzer.load_solution(dijkstra_file, "Dijkstra"):
        return None

    if not analyzer.load_solution(steiner_file, "Steiner"):
        return None

    if not analyzer.load_original_graph(graph_file):
        return None

    # Esegui l'analisi e salva
    results = analyzer.analyze_and_save_all()

    if results:
        print("\n✅ COMPLETATO!")
        print(f"   Nodi rimossi: {len(results['nodes_removed'])}")
        print(f"   Grafi salvati: {len(results['saved_graphs'])}")
        print("\n📁 File pronti nella cartella 'graphs2'")
        print("\n🚀 Per usare i grafi modificati:")
        print("   1. Modifica il path in paste.txt e paste-2.txt")
        print("      da 'graphs' a 'graphs2'")
        print("   2. Carica 'grafo_without_node_X.pickle'")
        print("   3. Entrambi gli algoritmi useranno lo stesso grafo")

    return analyzer

# Esempio di utilizzo
if __name__ == "__main__":
    print("🔬 PREPARAZIONE GRAFI MODIFICATI PER TEST")
    print("="*50)

    # File di input
    dijkstra_file = "graphs/dijistra_GRAPH_3_CUSTOM_COST_solution.pickle"
    steiner_file = "graphs/steiner_GRAPH_3_CUSTOM_COST_solution.pickle"
    graph_file = "graphs/grafo_3.pickle"

    # Esegui analisi e salvataggio
    analyzer = prepare_graphs_for_analysis(dijkstra_file, steiner_file, graph_file)
