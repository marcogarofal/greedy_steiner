import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SolutionCostComparator:
    """
    Confronta i costi e le metriche tra due soluzioni salvate in file pickle
    """

    def __init__(self):
        self.solutions = {}

    def load_solution(self, filepath, name):
        """
        Carica una soluzione da file pickle
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Estrai le informazioni chiave
            solution_info = {
                'filepath': filepath,
                'name': name,
                'data': data
            }

            # Determina il formato e estrai metriche standardizzate
            if 'solution_tree' in data:  # Formato Dijkstra
                solution_info['format'] = 'dijkstra'
                solution_info['tree'] = data['solution_tree']
                solution_info['algorithm'] = data.get('algorithm', 'dijkstra')
                solution_info['alpha'] = data.get('alpha', 0.5)
                solution_info['score'] = data.get('score', float('inf'))
                solution_info['acc_cost'] = data.get('acc_cost', 0)
                solution_info['aoc_cost'] = data.get('aoc_cost', 0)
                solution_info['total_cost'] = data.get('total_cost', 0)
                solution_info['connected_weak'] = data.get('connected_weak', [])
                solution_info['discretionary_used'] = data.get('discretionary_used', [])
                solution_info['capacity_usage'] = data.get('capacity_usage', {})
                solution_info['solution_edges'] = data.get('solution_edges', [])

            elif 'steiner_tree' in data:  # Formato Steiner
                solution_info['format'] = 'steiner'
                solution_info['tree'] = data['steiner_tree']

                # Estrai da solution_metadata se presente
                metadata = data.get('solution_metadata', {})
                solution_info['algorithm'] = 'steiner'
                solution_info['alpha'] = metadata.get('alpha', 0.5)
                solution_info['score'] = metadata.get('final_score', float('inf'))
                solution_info['acc_cost'] = metadata.get('acc_cost', 0)
                solution_info['aoc_cost'] = metadata.get('aoc_cost', 0)
                solution_info['total_cost'] = metadata.get('total_edge_cost', 0)
                solution_info['connected_weak'] = data.get('connected_weak_nodes', [])
                solution_info['discretionary_used'] = data.get('discretionary_used', [])
                solution_info['capacity_usage'] = data.get('capacity_usage', {})
                solution_info['solution_edges'] = data.get('solution_edges', [])

            else:
                print(f"⚠️ Formato non riconosciuto per {name}")
                return False

            # Calcola metriche aggiuntive
            if solution_info['tree']:
                solution_info['num_nodes'] = solution_info['tree'].number_of_nodes()
                solution_info['num_edges'] = solution_info['tree'].number_of_edges()
                solution_info['is_connected'] = nx.is_connected(solution_info['tree'])

                # Calcola peso totale degli archi se non presente
                if solution_info['total_cost'] == 0:
                    total_weight = sum(data.get('weight', 1) for _, _, data in solution_info['tree'].edges(data=True))
                    solution_info['total_cost'] = total_weight

            self.solutions[name] = solution_info

            print(f"✅ Caricata soluzione '{name}' dal file {os.path.basename(filepath)}")
            print(f"   Format: {solution_info['format']}")
            print(f"   Algorithm: {solution_info['algorithm']}")
            print(f"   Alpha: {solution_info['alpha']}")

            return True

        except Exception as e:
            print(f"❌ Errore nel caricamento di {filepath}: {e}")
            return False

    def compare_costs(self):
        """
        Confronta i costi tra tutte le soluzioni caricate
        """
        if len(self.solutions) < 2:
            print("❌ Servono almeno 2 soluzioni per il confronto")
            return None

        print(f"\n{'='*80}")
        print(f"CONFRONTO COSTI TRA SOLUZIONI")
        print(f"{'='*80}\n")

        # Crea tabella di confronto
        comparison_data = []

        for name, sol in self.solutions.items():
            row = {
                'Soluzione': name,
                'Algoritmo': sol['algorithm'],
                'Alpha (α)': sol['alpha'],
                'Score Finale': f"{sol['score']:.2f}",
                'ACC Cost': f"{sol['acc_cost']:.6f}",
                'AOC Cost': f"{sol['aoc_cost']:.6f}",
                'Cost Function (ACC+AOC)': f"{sol['acc_cost'] + sol['aoc_cost']:.6f}",
                'Edge Cost Totale': sol['total_cost'],
                'Nodi': sol.get('num_nodes', 'N/A'),
                'Archi': sol.get('num_edges', 'N/A'),
                'Weak Connessi': len(sol['connected_weak']),
                'Discretionary Usati': len(sol['discretionary_used'])
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        print("📊 TABELLA COMPARATIVA:")
        print(df.to_string(index=False))

        # Analisi dettagliata se ci sono esattamente 2 soluzioni
        if len(self.solutions) == 2:
            names = list(self.solutions.keys())
            sol1 = self.solutions[names[0]]
            sol2 = self.solutions[names[1]]

            print(f"\n📈 ANALISI DIFFERENZE ({names[0]} vs {names[1]}):")
            print(f"{'='*60}")

            # Confronto scores
            score_diff = sol2['score'] - sol1['score']
            score_pct = (score_diff / sol1['score'] * 100) if sol1['score'] != 0 else 0

            print(f"\n🎯 SCORE FINALE:")
            print(f"   {names[0]}: {sol1['score']:.2f}")
            print(f"   {names[1]}: {sol2['score']:.2f}")
            print(f"   Differenza: {score_diff:+.2f} ({score_pct:+.1f}%)")
            print(f"   Migliore: {'⭐ ' + names[0] if sol1['score'] < sol2['score'] else '⭐ ' + names[1]}")

            # Confronto componenti di costo
            print(f"\n💰 COMPONENTI DI COSTO:")

            acc_diff = sol2['acc_cost'] - sol1['acc_cost']
            print(f"   ACC (Average Communication Cost):")
            print(f"      {names[0]}: {sol1['acc_cost']:.6f}")
            print(f"      {names[1]}: {sol2['acc_cost']:.6f}")
            print(f"      Differenza: {acc_diff:+.6f}")

            aoc_diff = sol2['aoc_cost'] - sol1['aoc_cost']
            print(f"   AOC (Average Operational Cost):")
            print(f"      {names[0]}: {sol1['aoc_cost']:.6f}")
            print(f"      {names[1]}: {sol2['aoc_cost']:.6f}")
            print(f"      Differenza: {aoc_diff:+.6f}")

            total_cost_func = (sol1['acc_cost'] + sol1['aoc_cost'], sol2['acc_cost'] + sol2['aoc_cost'])
            cost_func_diff = total_cost_func[1] - total_cost_func[0]
            print(f"   Cost Function Totale (ACC+AOC):")
            print(f"      {names[0]}: {total_cost_func[0]:.6f}")
            print(f"      {names[1]}: {total_cost_func[1]:.6f}")
            print(f"      Differenza: {cost_func_diff:+.6f}")

            # Confronto edge costs
            edge_diff = sol2['total_cost'] - sol1['total_cost']
            edge_pct = (edge_diff / sol1['total_cost'] * 100) if sol1['total_cost'] != 0 else 0

            print(f"\n🔗 COSTO TOTALE ARCHI:")
            print(f"   {names[0]}: {sol1['total_cost']}")
            print(f"   {names[1]}: {sol2['total_cost']}")
            print(f"   Differenza: {edge_diff:+d} ({edge_pct:+.1f}%)")

            # Confronto struttura
            print(f"\n🌳 STRUTTURA ALBERO:")
            print(f"   Nodi:")
            print(f"      {names[0]}: {sol1.get('num_nodes', 'N/A')}")
            print(f"      {names[1]}: {sol2.get('num_nodes', 'N/A')}")
            print(f"   Archi:")
            print(f"      {names[0]}: {sol1.get('num_edges', 'N/A')}")
            print(f"      {names[1]}: {sol2.get('num_edges', 'N/A')}")

            # Confronto capacity usage
            if sol1['capacity_usage'] and sol2['capacity_usage']:
                print(f"\n⚡ UTILIZZO CAPACITÀ:")
                all_nodes = set(sol1['capacity_usage'].keys()) | set(sol2['capacity_usage'].keys())

                for node in sorted(all_nodes):
                    usage1 = sol1['capacity_usage'].get(node, 0)
                    usage2 = sol2['capacity_usage'].get(node, 0)
                    if usage1 > 0 or usage2 > 0:
                        print(f"   Nodo {node}: {names[0]}={usage1}, {names[1]}={usage2} (diff: {usage2-usage1:+d})")

        return df

    def visualize_cost_comparison(self):
        """
        Crea grafici di confronto dei costi
        """
        if len(self.solutions) < 2:
            print("❌ Servono almeno 2 soluzioni per la visualizzazione")
            return

        # Prepara i dati
        names = list(self.solutions.keys())
        scores = [self.solutions[name]['score'] for name in names]
        acc_costs = [self.solutions[name]['acc_cost'] for name in names]
        aoc_costs = [self.solutions[name]['aoc_cost'] for name in names]
        edge_costs = [self.solutions[name]['total_cost'] for name in names]

        # Crea figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Score finale
        bars1 = ax1.bar(names, scores, color=['skyblue', 'lightcoral'])
        ax1.set_title('Score Finale (minore è meglio)', fontsize=14, weight='bold')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)

        # Aggiungi valori sulle barre
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.2f}', ha='center', va='bottom')

        # 2. Componenti di costo (ACC vs AOC)
        x_pos = range(len(names))
        width = 0.35

        bars2_acc = ax2.bar([p - width/2 for p in x_pos], acc_costs, width,
                           label='ACC', color='lightblue')
        bars2_aoc = ax2.bar([p + width/2 for p in x_pos], aoc_costs, width,
                           label='AOC', color='lightgreen')

        ax2.set_title('Componenti Cost Function', fontsize=14, weight='bold')
        ax2.set_ylabel('Costo')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Costo totale archi
        bars3 = ax3.bar(names, edge_costs, color=['orange', 'gold'])
        ax3.set_title('Costo Totale Archi', fontsize=14, weight='bold')
        ax3.set_ylabel('Peso totale')
        ax3.grid(True, alpha=0.3)

        for bar, cost in zip(bars3, edge_costs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost}', ha='center', va='bottom')

        # 4. Confronto percentuale (se ci sono 2 soluzioni)
        if len(self.solutions) == 2:
            sol1 = self.solutions[names[0]]
            sol2 = self.solutions[names[1]]

            # Calcola differenze percentuali
            metrics = ['Score', 'ACC', 'AOC', 'Edge Cost']
            values1 = [sol1['score'], sol1['acc_cost'], sol1['aoc_cost'], sol1['total_cost']]
            values2 = [sol2['score'], sol2['acc_cost'], sol2['aoc_cost'], sol2['total_cost']]

            differences = []
            for v1, v2 in zip(values1, values2):
                if v1 != 0:
                    diff_pct = ((v2 - v1) / v1) * 100
                else:
                    diff_pct = 0
                differences.append(diff_pct)

            colors = ['red' if d > 0 else 'green' for d in differences]
            bars4 = ax4.bar(metrics, differences, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_title(f'Differenza % ({names[1]} vs {names[0]})', fontsize=14, weight='bold')
            ax4.set_ylabel('Differenza %')
            ax4.grid(True, alpha=0.3)

            # Aggiungi valori
            for bar, diff in zip(bars4, differences):
                height = bar.get_height()
                y_pos = height + 1 if height > 0 else height - 1
                ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{diff:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')

        plt.tight_layout()

        # Salva il grafico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'cost_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Grafico salvato: {filename}")

        plt.show()

    def export_comparison_report(self, filename=None):
        """
        Esporta un report dettagliato del confronto
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'cost_comparison_report_{timestamp}.txt'

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORT CONFRONTO COSTI SOLUZIONI\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Info sui file caricati
            f.write("FILE ANALIZZATI:\n")
            f.write("-"*40 + "\n")
            for name, sol in self.solutions.items():
                f.write(f"\n{name}:\n")
                f.write(f"  File: {os.path.basename(sol['filepath'])}\n")
                f.write(f"  Formato: {sol['format']}\n")
                f.write(f"  Algoritmo: {sol['algorithm']}\n")
                f.write(f"  Alpha: {sol['alpha']}\n")

            # Tabella comparativa
            f.write("\n\nTABELLA COMPARATIVA:\n")
            f.write("-"*40 + "\n")

            df = self.compare_costs()
            if df is not None:
                f.write(df.to_string(index=False))

            # Analisi dettagliata per coppie
            if len(self.solutions) == 2:
                names = list(self.solutions.keys())
                sol1 = self.solutions[names[0]]
                sol2 = self.solutions[names[1]]

                f.write(f"\n\nANALISI DETTAGLIATA ({names[0]} vs {names[1]}):\n")
                f.write("="*60 + "\n")

                # Winner analysis
                winners = {
                    'Score': names[0] if sol1['score'] < sol2['score'] else names[1],
                    'ACC': names[0] if sol1['acc_cost'] < sol2['acc_cost'] else names[1],
                    'AOC': names[0] if sol1['aoc_cost'] < sol2['aoc_cost'] else names[1],
                    'Edge Cost': names[0] if sol1['total_cost'] < sol2['total_cost'] else names[1],
                }

                f.write("\nVINCITORI PER METRICA:\n")
                for metric, winner in winners.items():
                    f.write(f"  {metric}: {winner}\n")

                # Conclusioni
                f.write("\nCONCLUSIONI:\n")
                f.write("-"*30 + "\n")

                overall_winner = names[0] if sol1['score'] < sol2['score'] else names[1]
                f.write(f"Soluzione migliore (score più basso): {overall_winner}\n")
                f.write(f"Score: {min(sol1['score'], sol2['score']):.2f}\n")

        print(f"📋 Report salvato: {filename}")
        return filename

# Funzioni di utilità per uso rapido
def compare_two_pickles(file1, file2, name1="Solution 1", name2="Solution 2"):
    """
    Confronta rapidamente due file pickle

    Uso:
        compare_two_pickles("steiner.pickle", "dijkstra.pickle")
    """
    comparator = SolutionCostComparator()

    # Carica i file
    success1 = comparator.load_solution(file1, name1)
    success2 = comparator.load_solution(file2, name2)

    if not (success1 and success2):
        print("❌ Impossibile caricare entrambi i file")
        return None

    # Confronta
    df = comparator.compare_costs()

    # Visualizza
    comparator.visualize_cost_comparison()

    # Esporta report
    comparator.export_comparison_report()

    return comparator

# Esempio di utilizzo
if __name__ == "__main__":
    print("🔍 CONFRONTO COSTI TRA SOLUZIONI")
    print("="*50)

    file1 = "graphs/dijkstra_solution_graph_3_alpha_0.5_20250728_202318.pickle"
    if not file1:
        print("❌ File 1 non specificato")
        exit()

    file2 = "graphs/steiner_GRAPH_3_CUSTOM_COST_solution.pickle"
    if not file2:
        print("❌ File 2 non specificato")
        exit()

    name1 = input("Nome per la prima soluzione (default: dal nome file): ").strip()
    if not name1:
        name1 = os.path.splitext(os.path.basename(file1))[0]

    name2 = input("Nome per la seconda soluzione (default: dal nome file): ").strip()
    if not name2:
        name2 = os.path.splitext(os.path.basename(file2))[0]

    # Esegui il confronto
    comparator = compare_two_pickles(file1, file2, name1, name2)
