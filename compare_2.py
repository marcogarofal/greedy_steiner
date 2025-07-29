import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SolutionCostComparator:
    """
    Confronta i costi e le metriche tra due soluzioni salvate in file pickle
    VERSIONE CORRETTA per gestire sia Dijkstra che Steiner con stesso formato
    """

    def __init__(self):
        self.solutions = {}

    def load_solution(self, filepath, name):
        """
        Carica una soluzione da file pickle - VERSIONE CORRETTA
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

            # NUOVO: Determina il tipo di algoritmo dal nome del file o dai metadati
            filename_lower = os.path.basename(filepath).lower()
            
            # Determina l'algoritmo
            if 'dijkstra' in filename_lower or 'dijistra' in filename_lower:
                solution_info['algorithm'] = 'dijkstra'
            elif 'steiner' in filename_lower:
                solution_info['algorithm'] = 'steiner'
            else:
                # Prova a determinarlo dai metadati
                metadata = data.get('solution_metadata', {})
                solution_info['algorithm'] = metadata.get('algorithm', 'unknown')

            # Estrai l'albero (campo diverso per ogni algoritmo)
            if 'steiner_tree' in data:
                solution_info['format'] = 'steiner'
                solution_info['tree'] = data['steiner_tree']
            elif 'dijistra_tree' in data:
                solution_info['format'] = 'dijkstra'
                solution_info['tree'] = data['dijistra_tree']
                solution_info['algorithm'] = 'dijkstra'  # Forza algoritmo dijkstra
            elif 'dijkstra_tree' in data:
                solution_info['format'] = 'dijkstra'
                solution_info['tree'] = data['dijkstra_tree']
                solution_info['algorithm'] = 'dijkstra'
            elif 'solution_tree' in data:
                solution_info['format'] = 'generic'
                solution_info['tree'] = data['solution_tree']
            else:
                print(f"⚠️ Nessun albero trovato in {name}")
                print(f"   Chiavi disponibili: {list(data.keys())}")
                return False

            # Estrai metriche dai metadati (stessa struttura per entrambi)
            metadata = data.get('solution_metadata', {})
            solution_info['alpha'] = metadata.get('alpha', 0.5)
            solution_info['score'] = metadata.get('final_score', float('inf'))
            solution_info['acc_cost'] = metadata.get('acc_cost', 0)
            solution_info['aoc_cost'] = metadata.get('aoc_cost', 0)
            solution_info['total_cost'] = metadata.get('total_edge_cost', 0)
            
            # Altri dati
            solution_info['connected_weak'] = data.get('connected_weak_nodes', [])
            solution_info['discretionary_used'] = data.get('discretionary_used', [])
            solution_info['capacity_usage'] = data.get('capacity_usage', {})
            solution_info['solution_edges'] = data.get('solution_edges', [])

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
            print(f"   Score: {solution_info['score']:.2f}")

            return True

        except Exception as e:
            print(f"❌ Errore nel caricamento di {filepath}: {e}")
            import traceback
            traceback.print_exc()
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
        Crea grafici di confronto dei costi con legende chiare
        """
        if len(self.solutions) < 2:
            print("❌ Servono almeno 2 soluzioni per la visualizzazione")
            return

        # Prepara i dati
        names = list(self.solutions.keys())
        algorithms = [self.solutions[name]['algorithm'] for name in names]
        scores = [self.solutions[name]['score'] for name in names]
        acc_costs = [self.solutions[name]['acc_cost'] for name in names]
        aoc_costs = [self.solutions[name]['aoc_cost'] for name in names]
        edge_costs = [self.solutions[name]['total_cost'] for name in names]

        # Colori distintivi per algoritmo
        algo_colors = {
            'steiner': '#2E86AB',  # Blu
            'dijkstra': '#A23B72'  # Rosa/Viola
        }
        bar_colors = [algo_colors.get(algo.lower(), '#666666') for algo in algorithms]

        # Crea figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Confronto Metriche tra Algoritmi', fontsize=16, weight='bold')

        # 1. Score finale
        bars1 = ax1.bar(names, scores, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_title('Score Finale Complessivo', fontsize=14, weight='bold')
        ax1.set_ylabel('Score (minore è migliore)', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')

        # Aggiungi valori sulle barre con box
        for bar, score, algo in zip(bars1, scores, algorithms):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(scores)*0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=11, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Evidenzia il vincitore
        min_score_idx = scores.index(min(scores))
        bars1[min_score_idx].set_linewidth(3)
        bars1[min_score_idx].set_edgecolor('gold')
        ax1.text(0.02, 0.98, f'🏆 Migliore: {names[min_score_idx]}',
                transform=ax1.transAxes, fontsize=12, weight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

        # 2. Componenti di costo (ACC vs AOC)
        x_pos = range(len(names))
        width = 0.35

        bars2_acc = ax2.bar([p - width/2 for p in x_pos], acc_costs, width,
                           label='ACC (Communication)', color='#4ECDC4', alpha=0.8,
                           edgecolor='black', linewidth=1.5)
        bars2_aoc = ax2.bar([p + width/2 for p in x_pos], aoc_costs, width,
                           label='AOC (Operational)', color='#F38375', alpha=0.8,
                           edgecolor='black', linewidth=1.5)

        ax2.set_title('Componenti della Cost Function C(G) = α·ACC + (1-α)·AOC', fontsize=14, weight='bold')
        ax2.set_ylabel('Valore del Costo', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names)
        ax2.grid(True, alpha=0.3, axis='y')

        # Aggiungi valori sulle barre
        for bars, costs, offset in [(bars2_acc, acc_costs, -width/2), (bars2_aoc, aoc_costs, width/2)]:
            for i, (bar, cost) in enumerate(zip(bars, costs)):
                height = bar.get_height()
                ax2.text(i + offset, height + 0.00001,
                        f'{cost:.5f}', ha='center', va='bottom', fontsize=9)

        # Legenda migliorata
        legend2 = ax2.legend(loc='upper left', fontsize=11, framealpha=0.9,
                            title='Componenti', title_fontsize=12)
        legend2.get_frame().set_edgecolor('black')

        # Aggiungi info su alpha
        alpha_values = [self.solutions[name]['alpha'] for name in names]
        alpha_text = ', '.join([f'{name}: α={alpha}' for name, alpha in zip(names, alpha_values)])
        ax2.text(0.98, 0.02, alpha_text, transform=ax2.transAxes,
                fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))

        # 3. Costo totale archi
        bars3 = ax3.bar(names, edge_costs, color=bar_colors, alpha=0.8,
                       edgecolor='black', linewidth=2)
        ax3.set_title('Costo Totale degli Archi (Somma Pesi)', fontsize=14, weight='bold')
        ax3.set_ylabel('Peso Totale', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, cost in zip(bars3, edge_costs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(edge_costs)*0.01,
                    f'{cost}', ha='center', va='bottom', fontsize=11, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Info su numero di archi
        num_edges = [self.solutions[name].get('num_edges', 'N/A') for name in names]
        for i, (name, n_edges) in enumerate(zip(names, num_edges)):
            ax3.text(i, -max(edge_costs)*0.05, f'{n_edges} archi',
                    ha='center', va='top', fontsize=10, style='italic')

        # 4. Confronto percentuale (se ci sono 2 soluzioni)
        if len(self.solutions) == 2:
            sol1 = self.solutions[names[0]]
            sol2 = self.solutions[names[1]]

            # Calcola differenze percentuali
            metrics = ['Score\nFinale', 'ACC\n(Comm.)', 'AOC\n(Oper.)', 'Costo\nArchi']
            values1 = [sol1['score'], sol1['acc_cost'], sol1['aoc_cost'], sol1['total_cost']]
            values2 = [sol2['score'], sol2['acc_cost'], sol2['aoc_cost'], sol2['total_cost']]

            differences = []
            for v1, v2 in zip(values1, values2):
                if v1 != 0:
                    diff_pct = ((v2 - v1) / v1) * 100
                else:
                    diff_pct = 0
                differences.append(diff_pct)

            # Colori basati su positivo/negativo
            colors = ['#E63946' if d > 0 else '#2A9D8F' for d in differences]
            bars4 = ax4.bar(metrics, differences, color=colors, alpha=0.8,
                           edgecolor='black', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax4.set_title(f'Differenza Percentuale: {names[1]} rispetto a {names[0]}',
                         fontsize=14, weight='bold')
            ax4.set_ylabel('Differenza %', fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')

            # Aggiungi valori con frecce
            for bar, diff in zip(bars4, differences):
                height = bar.get_height()
                if abs(height) > 0.1:  # Solo se la differenza è significativa
                    y_pos = height + (2 if height > 0 else -2)
                    arrow = '↑' if height > 0 else '↓'
                    ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                            f'{arrow} {diff:+.1f}%', ha='center',
                            va='bottom' if height > 0 else 'top',
                            fontsize=11, weight='bold',
                            bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor='white', alpha=0.8))

            # Legenda per i colori
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2A9D8F', label=f'{names[1]} migliore', alpha=0.8),
                Patch(facecolor='#E63946', label=f'{names[0]} migliore', alpha=0.8)
            ]
            ax4.legend(handles=legend_elements, loc='best', fontsize=11,
                      framealpha=0.9, title='Interpretazione', title_fontsize=12)
        else:
            # Se più di 2 soluzioni, mostra una tabella riassuntiva
            ax4.axis('off')

            # Crea tabella riassuntiva
            cell_text = []
            for name in names:
                sol = self.solutions[name]
                row = [
                    sol['algorithm'],
                    f"{sol['score']:.2f}",
                    f"{sol['acc_cost']:.5f}",
                    f"{sol['aoc_cost']:.5f}",
                    f"{sol['total_cost']}"
                ]
                cell_text.append(row)

            table = ax4.table(cellText=cell_text,
                            rowLabels=names,
                            colLabels=['Algoritmo', 'Score', 'ACC', 'AOC', 'Edge Cost'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])

            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 2)

            # Colora celle
            for i in range(len(names)):
                table[(i+1, 0)].set_facecolor(bar_colors[i])
                table[(i+1, 0)].set_alpha(0.3)

            ax4.set_title('Tabella Riassuntiva', fontsize=14, weight='bold')

        # Aggiungi legenda globale per gli algoritmi
        if len(set(algorithms)) > 1:
            from matplotlib.patches import Patch
            algo_patches = []
            for algo in set(algorithms):
                color = algo_colors.get(algo.lower(), '#666666')
                algo_patches.append(Patch(facecolor=color, label=algo.upper(), alpha=0.8))

            fig.legend(handles=algo_patches, loc='upper right',
                      bbox_to_anchor=(0.98, 0.98), fontsize=12,
                      title='Algoritmi', title_fontsize=13,
                      framealpha=0.9, edgecolor='black')

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

    file1 = "graphs/dijistra_GRAPH_3_CUSTOM_COST_solution.pickle"
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