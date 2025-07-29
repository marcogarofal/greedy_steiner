import networkx as nx
import pickle
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import time


def load_graph_and_solution(graph_pickle_path, solution_pickle_path):
    """
    Load the original graph and the Steiner tree solution
    """
    # Load original graph
    with open(graph_pickle_path, 'rb') as f:
        graph = pickle.load(f)

    # Load Steiner tree solution
    with open(solution_pickle_path, 'rb') as f:
        solution_data = pickle.load(f)

    steiner_tree = solution_data['steiner_tree']

    return graph, steiner_tree, solution_data

def calculate_all_dijkstra_paths(graph):
    """
    Calculate shortest paths between all pairs of nodes using Dijkstra
    """
    print("🔍 Calculating Dijkstra shortest paths for all node pairs...")
    dijkstra_distances = {}
    dijkstra_paths = {}

    nodes = list(graph.nodes())
    total_pairs = len(nodes) * (len(nodes) - 1) // 2
    processed = 0

    for i, source in enumerate(nodes):
        # Calculate shortest paths from source to all other nodes
        lengths, paths = nx.single_source_dijkstra(graph, source, weight='weight')

        for target in nodes[i+1:]:  # Only calculate for pairs once
            if target in lengths:
                dijkstra_distances[(source, target)] = lengths[target]
                dijkstra_distances[(target, source)] = lengths[target]  # Symmetric
                dijkstra_paths[(source, target)] = paths[target]
                dijkstra_paths[(target, source)] = paths[target][::-1]  # Reverse path
                processed += 1

        if (i + 1) % 10 == 0:
            print(f"   Progress: {processed}/{total_pairs} pairs processed ({processed/total_pairs*100:.1f}%)")

    print(f"✅ Calculated {len(dijkstra_distances)//2} unique shortest paths")
    return dijkstra_distances, dijkstra_paths

def calculate_steiner_distances(steiner_tree):
    """
    Calculate distances between all pairs of nodes in the Steiner tree
    """
    print("🌳 Calculating distances in Steiner tree...")
    steiner_distances = {}
    steiner_paths = {}

    # Get all nodes in the Steiner tree
    steiner_nodes = set()
    for u, v in steiner_tree.edges():
        steiner_nodes.add(u)
        steiner_nodes.add(v)

    steiner_nodes = list(steiner_nodes)

    for i, source in enumerate(steiner_nodes):
        try:
            # Calculate shortest paths in the Steiner tree
            lengths, paths = nx.single_source_dijkstra(steiner_tree, source, weight='weight')

            for target in steiner_nodes[i+1:]:
                if target in lengths:
                    steiner_distances[(source, target)] = lengths[target]
                    steiner_distances[(target, source)] = lengths[target]
                    steiner_paths[(source, target)] = paths[target]
                    steiner_paths[(target, source)] = paths[target][::-1]
        except nx.NetworkXNoPath:
            # Some nodes might not be connected in the Steiner tree
            continue

    print(f"✅ Calculated {len(steiner_distances)//2} unique paths in Steiner tree")
    return steiner_distances, steiner_paths, steiner_nodes

def compare_paths(graph, dijkstra_distances, steiner_distances, steiner_nodes):
    """
    Compare Dijkstra and Steiner tree distances
    """
    print("\n📊 Comparing paths...")

    comparison_results = {
        'dijkstra_wins': [],
        'steiner_wins': [],
        'ties': [],
        'steiner_only': []  # Pairs only connected in Steiner tree
    }

    # Only compare pairs that exist in both Steiner tree
    compared_pairs = set()

    for (u, v), steiner_dist in steiner_distances.items():
        if (u, v) in compared_pairs or (v, u) in compared_pairs:
            continue

        compared_pairs.add((u, v))

        if (u, v) in dijkstra_distances:
            dijkstra_dist = dijkstra_distances[(u, v)]

            # Calculate difference
            diff = dijkstra_dist - steiner_dist

            result = {
                'pair': (u, v),
                'dijkstra_distance': dijkstra_dist,
                'steiner_distance': steiner_dist,
                'difference': diff,
                'ratio': steiner_dist / dijkstra_dist if dijkstra_dist > 0 else float('inf')
            }

            # Since Dijkstra finds the shortest path, it should always be <= Steiner
            if abs(diff) < 1e-9:  # Tie (considering floating point precision)
                comparison_results['ties'].append(result)
            elif diff > 0:  # Dijkstra > Steiner (this should not happen!)
                comparison_results['steiner_wins'].append(result)
                print(f"⚠️  ANOMALY: Steiner shorter than Dijkstra for {u}-{v}!")
            else:  # Dijkstra < Steiner (expected)
                comparison_results['dijkstra_wins'].append(result)
        else:
            # This pair doesn't exist in the original graph (shouldn't happen)
            comparison_results['steiner_only'].append({
                'pair': (u, v),
                'steiner_distance': steiner_dist
            })

    return comparison_results

def print_detailed_results(comparison_results, solution_metadata):
    """
    Print detailed comparison results with latency differences
    """
    print("\n" + "="*80)
    print("DETAILED COMPARISON RESULTS: STEINER TREE vs DIJKSTRA")
    print("="*80)

    # Print solution metadata
    print(f"\n📋 SOLUTION METADATA:")
    print(f"   Graph index: {solution_metadata.get('graph_index', 'N/A')}")
    print(f"   Alpha: {solution_metadata.get('alpha', 'N/A')}")
    print(f"   Final score: {solution_metadata.get('final_score', 'N/A'):.2f}")
    print(f"   ACC cost: {solution_metadata.get('acc_cost', 'N/A'):.6f}")
    print(f"   AOC cost: {solution_metadata.get('aoc_cost', 'N/A'):.6f}")

    # Summary statistics
    total_comparisons = (len(comparison_results['dijkstra_wins']) +
                        len(comparison_results['steiner_wins']) +
                        len(comparison_results['ties']))

    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total node pairs compared: {total_comparisons}")
    print(f"   Dijkstra wins (shorter): {len(comparison_results['dijkstra_wins'])} ({len(comparison_results['dijkstra_wins'])/total_comparisons*100:.1f}%)")
    print(f"   Ties (equal distance): {len(comparison_results['ties'])} ({len(comparison_results['ties'])/total_comparisons*100:.1f}%)")
    print(f"   Steiner wins (ANOMALY): {len(comparison_results['steiner_wins'])} ({len(comparison_results['steiner_wins'])/total_comparisons*100:.1f}%)")

    # Detailed analysis of differences when Dijkstra wins
    if comparison_results['dijkstra_wins']:
        differences = [abs(r['difference']) for r in comparison_results['dijkstra_wins']]
        percentages = [((r['steiner_distance'] - r['dijkstra_distance']) / r['dijkstra_distance'] * 100)
                      for r in comparison_results['dijkstra_wins']]
        ratios = [r['ratio'] for r in comparison_results['dijkstra_wins']]

        print(f"\n📈 WHEN DIJKSTRA WINS (is shorter):")
        print(f"   Average latency increase: {np.mean(differences):.2f} units ({np.mean(percentages):.1f}% worse)")
        print(f"   Max latency increase: {np.max(differences):.2f} units ({np.max(percentages):.1f}% worse)")
        print(f"   Min latency increase: {np.min(differences):.2f} units ({np.min(percentages):.1f}% worse)")
        print(f"   Average ratio (Steiner/Dijkstra): {np.mean(ratios):.3f}")

        # Show top 10 biggest differences with detailed latency info
        sorted_wins = sorted(comparison_results['dijkstra_wins'],
                           key=lambda x: abs(x['difference']), reverse=True)[:10]

        print(f"\n   🔝 TOP 10 WORST LATENCY PENALTIES (Steiner vs Dijkstra):")
        print(f"   {'#':<3} {'Pair':<15} {'Dijkstra':<10} {'Steiner':<10} {'Difference':<12} {'% Worse':<10}")
        print(f"   {'-'*3} {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

        for i, result in enumerate(sorted_wins, 1):
            pair_str = f"{result['pair'][0]}-{result['pair'][1]}"
            diff = result['steiner_distance'] - result['dijkstra_distance']
            percent = (diff / result['dijkstra_distance']) * 100

            print(f"   {i:<3} {pair_str:<15} {result['dijkstra_distance']:<10.2f} "
                  f"{result['steiner_distance']:<10.2f} {diff:<12.2f} {percent:<10.1f}%")

        # Distribution of percentage increases
        print(f"\n   📊 LATENCY PENALTY DISTRIBUTION:")
        # Create buckets for percentage increases
        buckets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, float('inf'))]

        for low, high in buckets:
            count = sum(1 for p in percentages if low <= p < high)
            if count > 0:
                percentage = count / len(percentages) * 100
                if high == float('inf'):
                    print(f"      >{low}% worse: {count} pairs ({percentage:.1f}%)")
                else:
                    print(f"      {low}-{high}% worse: {count} pairs ({percentage:.1f}%)")

        # Examples of different latency penalty ranges
        print(f"\n   💡 EXAMPLE PATHS BY LATENCY PENALTY:")

        # Small penalty (< 10%)
        small_penalty = [r for r in comparison_results['dijkstra_wins']
                        if ((r['steiner_distance'] - r['dijkstra_distance']) / r['dijkstra_distance'] * 100) < 10]
        if small_penalty:
            example = small_penalty[0]
            percent = ((example['steiner_distance'] - example['dijkstra_distance']) / example['dijkstra_distance']) * 100
            print(f"      Small penalty (<10%): {example['pair'][0]}-{example['pair'][1]}")
            print(f"         Dijkstra: {example['dijkstra_distance']:.2f}, Steiner: {example['steiner_distance']:.2f} (+{percent:.1f}%)")

        # Medium penalty (10-50%)
        medium_penalty = [r for r in comparison_results['dijkstra_wins']
                         if 10 <= ((r['steiner_distance'] - r['dijkstra_distance']) / r['dijkstra_distance'] * 100) < 50]
        if medium_penalty:
            example = medium_penalty[0]
            percent = ((example['steiner_distance'] - example['dijkstra_distance']) / example['dijkstra_distance']) * 100
            print(f"      Medium penalty (10-50%): {example['pair'][0]}-{example['pair'][1]}")
            print(f"         Dijkstra: {example['dijkstra_distance']:.2f}, Steiner: {example['steiner_distance']:.2f} (+{percent:.1f}%)")

        # Large penalty (>50%)
        large_penalty = [r for r in comparison_results['dijkstra_wins']
                        if ((r['steiner_distance'] - r['dijkstra_distance']) / r['dijkstra_distance'] * 100) >= 50]
        if large_penalty:
            example = large_penalty[0]
            percent = ((example['steiner_distance'] - example['dijkstra_distance']) / example['dijkstra_distance']) * 100
            print(f"      Large penalty (>50%): {example['pair'][0]}-{example['pair'][1]}")
            print(f"         Dijkstra: {example['dijkstra_distance']:.2f}, Steiner: {example['steiner_distance']:.2f} (+{percent:.1f}%)")

    # Show ties with more detail
    if comparison_results['ties']:
        print(f"\n🤝 TIES (paths with equal distance): {len(comparison_results['ties'])}")
        print("   These pairs have ZERO latency penalty - Steiner uses the optimal path!")

        # Show first 10 ties as examples
        print(f"\n   Examples of optimal paths in Steiner tree:")
        for i, result in enumerate(comparison_results['ties'][:10], 1):
            print(f"   {i}. Nodes {result['pair'][0]}-{result['pair'][1]}: distance = {result['dijkstra_distance']:.2f} (0% penalty)")

        if len(comparison_results['ties']) > 10:
            print(f"   ... and {len(comparison_results['ties']) - 10} more optimal paths")

    # Overall insights
    print(f"\n🔍 LATENCY INSIGHTS:")
    tie_percentage = len(comparison_results['ties']) / total_comparisons * 100

    if comparison_results['dijkstra_wins']:
        avg_penalty = np.mean([((r['steiner_distance'] - r['dijkstra_distance']) / r['dijkstra_distance'] * 100)
                              for r in comparison_results['dijkstra_wins']])

        print(f"   📊 {tie_percentage:.1f}% of paths have ZERO latency penalty (optimal)")
        print(f"   📊 {100-tie_percentage:.1f}% of paths have some latency penalty")
        print(f"   📊 Average latency penalty when suboptimal: {avg_penalty:.1f}%")

        # Calculate total network latency impact
        total_dijkstra = sum(r['dijkstra_distance'] for r in comparison_results['dijkstra_wins']) + \
                        sum(r['dijkstra_distance'] for r in comparison_results['ties'])
        total_steiner = sum(r['steiner_distance'] for r in comparison_results['dijkstra_wins']) + \
                       sum(r['steiner_distance'] for r in comparison_results['ties'])

        overall_penalty = ((total_steiner - total_dijkstra) / total_dijkstra) * 100

        print(f"\n   🌐 OVERALL NETWORK IMPACT:")
        print(f"      Total Dijkstra latency: {total_dijkstra:.2f}")
        print(f"      Total Steiner latency: {total_steiner:.2f}")
        print(f"      Overall latency penalty: {overall_penalty:.1f}%")
    else:
        print(f"   ✅ ALL paths in Steiner tree are optimal (100% ties)!")

def create_visualization(comparison_results, output_path):
    """
    Create visualization of the comparison results
    """
    plt.figure(figsize=(15, 10))

    # Subplot 1: Bar chart of wins/ties
    plt.subplot(2, 2, 1)
    categories = ['Dijkstra\nWins', 'Ties', 'Steiner\nWins']
    counts = [len(comparison_results['dijkstra_wins']),
              len(comparison_results['ties']),
              len(comparison_results['steiner_wins'])]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars = plt.bar(categories, counts, color=colors, edgecolor='black', linewidth=2)
    plt.title('Path Comparison Results', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Node Pairs', fontsize=12)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Subplot 2: Histogram of path length differences
    if comparison_results['dijkstra_wins']:
        plt.subplot(2, 2, 2)
        differences = [abs(r['difference']) for r in comparison_results['dijkstra_wins']]
        plt.hist(differences, bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
        plt.xlabel('Path Length Difference (Steiner - Dijkstra)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Path Length Differences', fontsize=14, fontweight='bold')

        # Add mean line
        mean_diff = np.mean(differences)
        plt.axvline(mean_diff, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_diff:.2f}')
        plt.legend()

    # Subplot 3: Scatter plot of Steiner vs Dijkstra distances
    plt.subplot(2, 2, 3)
    if comparison_results['dijkstra_wins'] or comparison_results['ties']:
        all_results = comparison_results['dijkstra_wins'] + comparison_results['ties']
        dijkstra_dists = [r['dijkstra_distance'] for r in all_results]
        steiner_dists = [r['steiner_distance'] for r in all_results]

        # Separate wins and ties for different colors
        dijkstra_wins_x = [r['dijkstra_distance'] for r in comparison_results['dijkstra_wins']]
        dijkstra_wins_y = [r['steiner_distance'] for r in comparison_results['dijkstra_wins']]
        ties_x = [r['dijkstra_distance'] for r in comparison_results['ties']]
        ties_y = [r['steiner_distance'] for r in comparison_results['ties']]

        plt.scatter(dijkstra_wins_x, dijkstra_wins_y, alpha=0.6, color='#FF6B6B',
                   label='Dijkstra shorter', s=50)
        plt.scatter(ties_x, ties_y, alpha=0.8, color='#4ECDC4',
                   label='Equal distance', s=50)

        # Add y=x line
        max_dist = max(max(dijkstra_dists), max(steiner_dists))
        plt.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5, label='y=x')

        plt.xlabel('Dijkstra Distance', fontsize=12)
        plt.ylabel('Steiner Tree Distance', fontsize=12)
        plt.title('Steiner vs Dijkstra Path Lengths', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Subplot 4: Ratio distribution
    if comparison_results['dijkstra_wins']:
        plt.subplot(2, 2, 4)
        ratios = [r['ratio'] for r in comparison_results['dijkstra_wins'] if r['ratio'] < 10]  # Filter extreme ratios
        plt.hist(ratios, bins=30, color='#45B7D1', edgecolor='black', alpha=0.7)
        plt.xlabel('Ratio (Steiner Distance / Dijkstra Distance)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Distance Ratios', fontsize=14, fontweight='bold')

        # Add mean line
        mean_ratio = np.mean(ratios)
        plt.axvline(mean_ratio, color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_ratio:.2f}')
        plt.axvline(1.0, color='green', linestyle='--', linewidth=2,
                   label='Ratio = 1 (equal)')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Visualization saved: {output_path}")

def save_detailed_report(comparison_results, solution_metadata, output_path):
    """
    Save detailed comparison report to file with latency analysis
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STEINER TREE vs DIJKSTRA SHORTEST PATHS - DETAILED REPORT\n")
        f.write("="*80 + "\n\n")

        # Metadata
        f.write("SOLUTION METADATA:\n")
        f.write("-"*40 + "\n")
        for key, value in solution_metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Summary
        total = (len(comparison_results['dijkstra_wins']) +
                len(comparison_results['steiner_wins']) +
                len(comparison_results['ties']))

        f.write("SUMMARY STATISTICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total pairs compared: {total}\n")
        f.write(f"Dijkstra wins: {len(comparison_results['dijkstra_wins'])} ({len(comparison_results['dijkstra_wins'])/total*100:.1f}%)\n")
        f.write(f"Ties: {len(comparison_results['ties'])} ({len(comparison_results['ties'])/total*100:.1f}%)\n")
        f.write(f"Steiner wins: {len(comparison_results['steiner_wins'])} ({len(comparison_results['steiner_wins'])/total*100:.1f}%)\n\n")

        # Detailed latency statistics
        if comparison_results['dijkstra_wins']:
            f.write("LATENCY ANALYSIS - DIJKSTRA WINS:\n")
            f.write("-"*40 + "\n")
            differences = [abs(r['difference']) for r in comparison_results['dijkstra_wins']]
            percentages = [((r['steiner_distance'] - r['dijkstra_distance']) / r['dijkstra_distance'] * 100)
                          for r in comparison_results['dijkstra_wins']]
            ratios = [r['ratio'] for r in comparison_results['dijkstra_wins']]

            f.write(f"Average absolute difference: {np.mean(differences):.3f}\n")
            f.write(f"Average percentage increase: {np.mean(percentages):.1f}%\n")
            f.write(f"Std deviation of differences: {np.std(differences):.3f}\n")
            f.write(f"Max difference: {np.max(differences):.3f} ({np.max(percentages):.1f}% worse)\n")
            f.write(f"Min difference: {np.min(differences):.3f} ({np.min(percentages):.1f}% worse)\n")
            f.write(f"Average ratio (Steiner/Dijkstra): {np.mean(ratios):.3f}\n\n")

            # Latency distribution
            f.write("LATENCY PENALTY DISTRIBUTION:\n")
            buckets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, float('inf'))]
            for low, high in buckets:
                count = sum(1 for p in percentages if low <= p < high)
                if count > 0:
                    percentage = count / len(percentages) * 100
                    if high == float('inf'):
                        f.write(f"  >{low}% worse: {count} pairs ({percentage:.1f}%)\n")
                    else:
                        f.write(f"  {low}-{high}% worse: {count} pairs ({percentage:.1f}%)\n")
            f.write("\n")

            # List all Dijkstra wins with latency details
            f.write("DETAILED LIST - ALL PATHS WITH LATENCY PENALTY:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Pair':<20} {'Dijkstra':<12} {'Steiner':<12} {'Difference':<12} {'% Penalty':<10}\n")
            f.write("-"*80 + "\n")

            sorted_wins = sorted(comparison_results['dijkstra_wins'],
                               key=lambda x: abs(x['difference']), reverse=True)

            for result in sorted_wins:
                pair_str = f"({result['pair'][0]}, {result['pair'][1]})"
                diff = result['steiner_distance'] - result['dijkstra_distance']
                percent = (diff / result['dijkstra_distance']) * 100

                f.write(f"{pair_str:<20} {result['dijkstra_distance']:<12.3f} "
                       f"{result['steiner_distance']:<12.3f} {diff:<12.3f} {percent:<10.1f}%\n")

            # Overall network impact
            f.write("\n" + "-"*40 + "\n")
            f.write("OVERALL NETWORK LATENCY IMPACT:\n")
            total_dijkstra = sum(r['dijkstra_distance'] for r in comparison_results['dijkstra_wins']) + \
                            sum(r['dijkstra_distance'] for r in comparison_results['ties'])
            total_steiner = sum(r['steiner_distance'] for r in comparison_results['dijkstra_wins']) + \
                           sum(r['steiner_distance'] for r in comparison_results['ties'])

            overall_penalty = ((total_steiner - total_dijkstra) / total_dijkstra) * 100

            f.write(f"Total Dijkstra latency: {total_dijkstra:.2f}\n")
            f.write(f"Total Steiner latency: {total_steiner:.2f}\n")
            f.write(f"Overall latency penalty: {overall_penalty:.1f}%\n")

        # List all ties
        if comparison_results['ties']:
            f.write("\n\nDETAILED LIST - TIES (ZERO LATENCY PENALTY):\n")
            f.write("-"*40 + "\n")
            for i, result in enumerate(comparison_results['ties'], 1):
                f.write(f"{i}. Pair ({result['pair'][0]}, {result['pair'][1]}): distance = {result['dijkstra_distance']:.3f} (0% penalty)\n")

    print(f"📄 Detailed report saved: {output_path}")

def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_path = os.path.join(script_dir, 'graphs/')

    # Ask user for input files
    print("🔍 STEINER TREE vs DIJKSTRA COMPARISON TOOL")
    print("="*50)

    # List available files
    print("\nAvailable graph files:")
    graph_files = [f for f in os.listdir(graphs_path) if f.startswith('grafo') and f.endswith('.pickle')]
    for i, f in enumerate(graph_files):
        print(f"  {i+1}. {f}")

    print("\nAvailable solution files:")
    solution_files = [f for f in os.listdir(graphs_path) if 'solution' in f and f.endswith('.pickle')]
    for i, f in enumerate(solution_files):
        print(f"  {i+1}. {f}")

    # Get file paths
    graph_file = input("\nEnter graph pickle filename (e.g., grafo_3.pickle): ").strip()
    if not graph_file:
        print("❌ No filename entered. Using default: grafo_3.pickle")
        graph_file = "grafo_3.pickle"

    solution_file = input("Enter solution pickle filename (e.g., steiner_GRAPH_3_CUSTOM_COST_solution.pickle): ").strip()
    if not solution_file:
        print("❌ No filename entered. Using default: steiner_GRAPH_3_CUSTOM_COST_solution.pickle")
        solution_file = "steiner_GRAPH_3_CUSTOM_COST_solution.pickle"

    graph_path = os.path.join(graphs_path, graph_file)
    solution_path = os.path.join(graphs_path, solution_file)

    # Check if files exist
    if not os.path.exists(graph_path):
        print(f"❌ Error: Graph file not found: {graph_path}")
        return

    if not os.path.exists(solution_path):
        print(f"❌ Error: Solution file not found: {solution_path}")
        return

    print(f"\n✅ Loading files...")

    # Load data
    graph, steiner_tree, solution_data = load_graph_and_solution(graph_path, solution_path)
    solution_metadata = solution_data.get('solution_metadata', {})

    print(f"✅ Graph loaded: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"✅ Steiner tree loaded: {len(steiner_tree.nodes())} nodes, {len(steiner_tree.edges())} edges")


    start_time = time.time()

    # Calculate all shortest paths
    dijkstra_distances, dijkstra_paths = calculate_all_dijkstra_paths(graph)

    # Calculate Steiner tree distances
    steiner_distances, steiner_paths, steiner_nodes = calculate_steiner_distances(steiner_tree)



    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n⏱️  Calcolo completato in {execution_time:.3f} secondi")


    # Compare paths
    comparison_results = compare_paths(graph, dijkstra_distances, steiner_distances, steiner_nodes)

    # Print results
    print_detailed_results(comparison_results, solution_metadata)

    # Save outputs
    base_name = solution_file.replace('_solution.pickle', '')

    # Create visualization
    viz_path = os.path.join(graphs_path, f"{base_name}_vs_dijkstra_comparison.png")
    create_visualization(comparison_results, viz_path)

    # Save detailed report
    report_path = os.path.join(graphs_path, f"{base_name}_vs_dijkstra_report.txt")
    save_detailed_report(comparison_results, solution_metadata, report_path)

    # Save comparison data as pickle for future analysis
    comparison_data = {
        'comparison_results': comparison_results,
        'solution_metadata': solution_metadata,
        'graph_info': {
            'nodes': len(graph.nodes()),
            'edges': len(graph.edges())
        },
        'steiner_info': {
            'nodes': len(steiner_tree.nodes()),
            'edges': len(steiner_tree.edges())
        }
    }

    comparison_pickle_path = os.path.join(graphs_path, f"{base_name}_vs_dijkstra_data.pickle")
    with open(comparison_pickle_path, 'wb') as f:
        pickle.dump(comparison_data, f)

    print(f"\n✅ All outputs saved:")
    print(f"   📊 Visualization: {viz_path}")
    print(f"   📄 Report: {report_path}")
    print(f"   💾 Data: {comparison_pickle_path}")
    print(f"\n⏱️  Calcolo completato in {execution_time:.3f} secondi")

if __name__ == "__main__":
    main()
