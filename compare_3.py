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
                'data': data
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
            elif 'dijistra_tree' in data:
                solution_info['tree'] = data['dijistra_tree']
            elif 'dijkstra_tree' in data:
                solution_info['tree'] = data['dijkstra_tree']
            elif 'solution_tree' in data:
                solution_info['tree'] = data['solution_tree']
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
                data = pickle.load(f)
            
            # Il grafo potrebbe essere salvato con chiavi diverse
            if isinstance(data, nx.Graph):
                self.original_graph = data
            elif 'graph' in data:
                self.original_graph = data['graph']
            elif 'original_graph' in data:
                self.original_graph = data['original_graph']
            else:
                # Prova a cercare una chiave che contiene un grafo
                for key, value in data.items():
                    if isinstance(value, nx.Graph):
                        self.original_graph = value
                        break
            
            if self.original_graph:
                print(f"✅ Grafo originale caricato")
                print(f"   Nodi: {self.original_graph.number_of_nodes()}")
                print(f"   Archi: {self.original_graph.number_of_edges()}")
                return True
            else:
                print("❌ Grafo non trovato nel file")
                return False
                
        except Exception as e:
            print(f"❌ Errore nel caricamento del grafo: {e}")
            return False
    
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
    
    def find_matching_paths(self):
        """
        Trova percorsi con lo stesso numero di hop tra le due soluzioni
        """
        if len(self.solutions) != 2:
            print("❌ Servono esattamente 2 soluzioni")
            return None
        
        names = list(self.solutions.keys())
        sol1 = self.solutions[names[0]]
        sol2 = self.solutions[names[1]]
        
        # Trova tutti i percorsi per entrambe le soluzioni
        paths1 = self.find_all_paths_between_weak_nodes(
            sol1['tree'], 
            sol1['connected_weak']
        )
        paths2 = self.find_all_paths_between_weak_nodes(
            sol2['tree'], 
            sol2['connected_weak']
        )
        
        print(f"\n📊 ANALISI PERCORSI TRA NODI WEAK:")
        print(f"{'='*60}")
        print(f"\n{names[0]}:")
        print(f"  Percorsi trovati: {len(paths1)}")
        if paths1:
            print(f"  Percorso più lungo: {paths1[0]['intermediate_count']} nodi intermedi")
        
        print(f"\n{names[1]}:")
        print(f"  Percorsi trovati: {len(paths2)}")
        if paths2:
            print(f"  Percorso più lungo: {paths2[0]['intermediate_count']} nodi intermedi")
        
        # Cerca percorsi con stesso numero di hop
        matching_paths = []
        
        for p1 in paths1:
            for p2 in paths2:
                # Stesso numero di hop e stessi nodi weak (in qualsiasi ordine)
                if (p1['hop_count'] == p2['hop_count'] and 
                    {p1['source'], p1['target']} == {p2['source'], p2['target']}):
                    
                    matching_paths.append({
                        'weak_nodes': (p1['source'], p1['target']),
                        'hop_count': p1['hop_count'],
                        'path1': p1,
                        'path2': p2,
                        'common_intermediate': set(p1['intermediate_nodes']) & set(p2['intermediate_nodes'])
                    })
        
        # Ordina per numero di hop (decrescente)
        matching_paths.sort(key=lambda x: x['hop_count'], reverse=True)
        
        return paths1, paths2, matching_paths
    
    def analyze_node_types(self, nodes):
        """
        Analizza il tipo di ciascun nodo (mandatory, discretionary, etc.)
        """
        node_types = {}
        
        for name, sol in self.solutions.items():
            node_types[name] = {}
            
            for node in nodes:
                if node in sol['weak_nodes']:
                    node_types[name][node] = 'weak'
                elif node in sol['mandatory_nodes']:
                    node_types[name][node] = 'mandatory'
                elif node in sol['discretionary_nodes']:
                    if node in sol['discretionary_used']:
                        node_types[name][node] = 'discretionary_used'
                    else:
                        node_types[name][node] = 'discretionary_not_used'
                else:
                    node_types[name][node] = 'unknown'
        
        return node_types
    
    def select_best_path_for_analysis(self):
        """
        Seleziona il miglior percorso per l'analisi
        """
        paths1, paths2, matching_paths = self.find_matching_paths()
        
        if not paths1 and not paths2:
            print("\n❌ Nessun percorso trovato tra nodi weak")
            return None
        
        # Se ci sono percorsi matching, usa il più lungo
        if matching_paths:
            print(f"\n✅ Trovati {len(matching_paths)} percorsi con stesso numero di hop")
            best_match = matching_paths[0]
            
            print(f"\n🎯 PERCORSO SELEZIONATO (matching):")
            print(f"   Nodi weak: {best_match['weak_nodes']}")
            print(f"   Numero hop: {best_match['hop_count']}")
            print(f"   Nodi intermedi comuni: {len(best_match['common_intermediate'])}")
            
            return {
                'type': 'matching',
                'data': best_match,
                'selected_path': best_match['path1']  # Usa il percorso della prima soluzione
            }
        
        # Altrimenti usa il percorso più lungo disponibile
        longest_path = None
        longest_solution = None
        
        if paths1 and (not paths2 or paths1[0]['intermediate_count'] >= paths2[0]['intermediate_count']):
            longest_path = paths1[0]
            longest_solution = list(self.solutions.keys())[0]
        elif paths2:
            longest_path = paths2[0]
            longest_solution = list(self.solutions.keys())[1]
        
        if longest_path:
            print(f"\n🎯 PERCORSO SELEZIONATO (più lungo):")
            print(f"   Soluzione: {longest_solution}")
            print(f"   Nodi weak: {longest_path['source']} -> {longest_path['target']}")
            print(f"   Numero nodi intermedi: {longest_path['intermediate_count']}")
            
            return {
                'type': 'longest',
                'data': longest_path,
                'selected_path': longest_path,
                'solution': longest_solution
            }
        
        return None
    
    def generate_modified_graphs(self, nodes_to_remove):
        """
        Genera grafi modificati rimuovendo un nodo alla volta
        """
        if not self.original_graph:
            print("❌ Grafo originale non caricato")
            return []
        
        modified_graphs = []
        
        for node in nodes_to_remove:
            # Crea una copia del grafo originale
            modified_graph = copy.deepcopy(self.original_graph)
            
            # Rimuovi il nodo
            if node in modified_graph:
                modified_graph.remove_node(node)
                
                modified_graphs.append({
                    'removed_node': node,
                    'graph': modified_graph,
                    'remaining_nodes': modified_graph.number_of_nodes(),
                    'remaining_edges': modified_graph.number_of_edges()
                })
                
                print(f"   ✅ Creato grafo senza nodo {node}")
            else:
                print(f"   ⚠️ Nodo {node} non presente nel grafo originale")
        
        return modified_graphs
    
    def save_modified_graphs(self, modified_graphs, output_dir="modified_graphs"):
        """
        Salva i grafi modificati su file
        """
        # Crea directory se non esiste
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        for mg in modified_graphs:
            filename = f"graph_without_node_{mg['removed_node']}_{timestamp}.pickle"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'graph': mg['graph'],
                    'removed_node': mg['removed_node'],
                    'timestamp': timestamp,
                    'original_nodes': self.original_graph.number_of_nodes(),
                    'original_edges': self.original_graph.number_of_edges(),
                    'remaining_nodes': mg['remaining_nodes'],
                    'remaining_edges': mg['remaining_edges']
                }, f)
            
            saved_files.append(filepath)
            print(f"   💾 Salvato: {filename}")
        
        return saved_files
    
    def analyze_and_prepare_graphs(self):
        """
        Esegue l'analisi completa e prepara i grafi modificati
        """
        print("\n🔍 ANALISI RIMOZIONE NODI")
        print("="*60)
        
        # Seleziona il percorso migliore
        best_path_info = self.select_best_path_for_analysis()
        
        if not best_path_info:
            print("\n❌ Impossibile trovare un percorso adatto per l'analisi")
            return None
        
        # Estrai i nodi intermedi
        if best_path_info['type'] == 'matching':
            path_data = best_path_info['data']['path1']
        else:
            path_data = best_path_info['data']
        
        intermediate_nodes = path_data['intermediate_nodes']
        
        print(f"\n📋 NODI INTERMEDI DA ANALIZZARE ({len(intermediate_nodes)}):")
        print(f"   Percorso completo: {' -> '.join(map(str, path_data['path']))}")
        
        # Analizza il tipo di ciascun nodo
        node_types = self.analyze_node_types(intermediate_nodes)
        
        print(f"\n📊 TIPOLOGIA NODI:")
        for i, node in enumerate(intermediate_nodes, 1):
            types = [f"{sol}: {node_types[sol][node]}" for sol in node_types]
            print(f"   {i}. Nodo {node}: {', '.join(types)}")
        
        # Genera i grafi modificati
        print(f"\n🔧 GENERAZIONE GRAFI MODIFICATI:")
        modified_graphs = self.generate_modified_graphs(intermediate_nodes)
        
        # Salva i grafi
        if modified_graphs:
            print(f"\n💾 SALVATAGGIO GRAFI:")
            saved_files = self.save_modified_graphs(modified_graphs)
            
            # Crea report
            self.create_analysis_report(
                best_path_info, 
                intermediate_nodes, 
                node_types, 
                saved_files
            )
            
            return {
                'path_info': best_path_info,
                'intermediate_nodes': intermediate_nodes,
                'node_types': node_types,
                'modified_graphs': modified_graphs,
                'saved_files': saved_files
            }
        
        return None
    
    def create_analysis_report(self, path_info, intermediate_nodes, node_types, saved_files):
        """
        Crea un report dell'analisi
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"node_removal_analysis_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORT ANALISI RIMOZIONE NODI\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Info sul percorso selezionato
            f.write("PERCORSO SELEZIONATO:\n")
            f.write("-"*40 + "\n")
            
            if path_info['type'] == 'matching':
                data = path_info['data']
                f.write(f"Tipo: Percorso con matching tra le due soluzioni\n")
                f.write(f"Nodi weak: {data['weak_nodes']}\n")
                f.write(f"Numero hop: {data['hop_count']}\n")
                f.write(f"Percorso sol1: {' -> '.join(map(str, data['path1']['path']))}\n")
                f.write(f"Percorso sol2: {' -> '.join(map(str, data['path2']['path']))}\n")
            else:
                data = path_info['data']
                f.write(f"Tipo: Percorso più lungo\n")
                f.write(f"Soluzione: {path_info['solution']}\n")
                f.write(f"Nodi weak: {data['source']} -> {data['target']}\n")
                f.write(f"Percorso: {' -> '.join(map(str, data['path']))}\n")
            
            # Nodi da rimuovere
            f.write(f"\n\nNODI INTERMEDI DA TESTARE ({len(intermediate_nodes)}):\n")
            f.write("-"*40 + "\n")
            
            for i, node in enumerate(intermediate_nodes, 1):
                f.write(f"\n{i}. Nodo {node}:\n")
                for sol_name, types in node_types.items():
                    f.write(f"   - {sol_name}: {types[node]}\n")
            
            # File generati
            f.write("\n\nFILE GENERATI:\n")
            f.write("-"*40 + "\n")
            for i, filepath in enumerate(saved_files, 1):
                f.write(f"{i}. {os.path.basename(filepath)}\n")
            
            # Istruzioni per l'uso
            f.write("\n\nISTRUZIONI PER L'USO:\n")
            f.write("-"*40 + "\n")
            f.write("1. I grafi modificati sono stati salvati nella cartella 'modified_graphs'\n")
            f.write("2. Ogni file contiene il grafo originale senza uno specifico nodo\n")
            f.write("3. Utilizzare questi grafi per rilanciare gli algoritmi Dijkstra e Steiner\n")
            f.write("4. Confrontare i risultati per valutare l'impatto di ogni rimozione\n")
        
        print(f"\n📄 Report salvato: {filename}")

# Funzione di utilità per uso rapido
def analyze_node_removal(dijkstra_file, steiner_file, graph_file):
    """
    Funzione principale per l'analisi della rimozione dei nodi
    
    Args:
        dijkstra_file: path al file pickle della soluzione Dijkstra
        steiner_file: path al file pickle della soluzione Steiner
        graph_file: path al file pickle del grafo originale
    """
    analyzer = NodeRemovalAnalyzer()
    
    # Carica le soluzioni
    print("📂 CARICAMENTO FILE...")
    if not analyzer.load_solution(dijkstra_file, "Dijkstra"):
        return None
    
    if not analyzer.load_solution(steiner_file, "Steiner"):
        return None
    
    # Carica il grafo originale
    if not analyzer.load_original_graph(graph_file):
        return None
    
    # Esegui l'analisi
    results = analyzer.analyze_and_prepare_graphs()
    
    if results:
        print("\n✅ ANALISI COMPLETATA!")
        print(f"   Nodi da testare: {len(results['intermediate_nodes'])}")
        print(f"   Grafi generati: {len(results['saved_files'])}")
        print("\n🚀 Prossimi passi:")
        print("   1. Utilizzare i grafi nella cartella 'modified_graphs'")
        print("   2. Rilanciare gli algoritmi su ogni grafo modificato")
        print("   3. Confrontare i risultati per valutare l'impatto")
    
    return analyzer

# Esempio di utilizzo
if __name__ == "__main__":
    print("🔬 ANALISI RIMOZIONE NODI PER TEST LATENZA")
    print("="*50)
    
    # File di input
    dijkstra_file = "graphs/dijistra_GRAPH_3_CUSTOM_COST_solution.pickle"
    steiner_file = "graphs/steiner_GRAPH_3_CUSTOM_COST_solution.pickle"
    graph_file = "graphs/grafo_3.pickle"

    # Esegui analisi
    analyzer = analyze_node_removal(dijkstra_file, steiner_file, graph_file)