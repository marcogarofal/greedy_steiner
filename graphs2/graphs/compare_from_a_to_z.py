import networkx as nx
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import json
from datetime import datetime
matplotlib.use('Agg')

class SteinerTreeAnalyzer:
    """Analyzer for comparing paths in different Steiner tree solutions"""

    def __init__(self):
        self.trees = {}
        self.metadata = {}
        self.original_graph = None

    def load_solution(self, filepath, algorithm_name):
        """
        Load a Steiner tree solution from pickle file

        Args:
            filepath: Path to the pickle file
            algorithm_name: Name to identify this algorithm (e.g., 'Algorithm1', 'Algorithm2')
        """
        try:
            with open(filepath, 'rb') as f:
                solution_data = pickle.load(f)

            # Debug: Show what's in the pickle file
            print(f"\n🔍 Analyzing pickle file structure for {algorithm_name}:")
            if isinstance(solution_data, dict):
                print(f"   Keys found: {list(solution_data.keys())}")
            elif isinstance(solution_data, nx.Graph):
                print(f"   Direct NetworkX graph found")
                # If it's directly a graph, use it
                self.trees[algorithm_name] = solution_data
                self.metadata[algorithm_name] = {}
                print(f"✅ Loaded {algorithm_name} from {os.path.basename(filepath)}")
                print(f"   - Nodes: {len(self.trees[algorithm_name].nodes())}")
                print(f"   - Edges: {len(self.trees[algorithm_name].edges())}")
                return True
            else:
                print(f"   Type: {type(solution_data)}")

            # Try different possible structures
            tree_found = False

            # Structure 1: Has 'steiner_tree' key
            if isinstance(solution_data, dict) and 'steiner_tree' in solution_data:
                self.trees[algorithm_name] = solution_data['steiner_tree']
                tree_found = True
                print(f"   ✓ Found tree in 'steiner_tree' field")

            # Structure 2: Has 'tree' key
            elif isinstance(solution_data, dict) and 'tree' in solution_data:
                self.trees[algorithm_name] = solution_data['tree']
                tree_found = True
                print(f"   ✓ Found tree in 'tree' field")

            # Structure 3: Has 'graph' key
            elif isinstance(solution_data, dict) and 'graph' in solution_data:
                self.trees[algorithm_name] = solution_data['graph']
                tree_found = True
                print(f"   ✓ Found tree in 'graph' field")

            # Structure 4: Look for NetworkX graph in any key
            elif isinstance(solution_data, dict):
                for key, value in solution_data.items():
                    if isinstance(value, nx.Graph):
                        self.trees[algorithm_name] = value
                        tree_found = True
                        print(f"   ✓ Found NetworkX graph in '{key}' field")
                        break

            if not tree_found:
                # Show available keys to help debug
                if isinstance(solution_data, dict):
                    print(f"   ❌ No graph found. Available keys: {list(solution_data.keys())}")
                    print(f"   💡 Tip: Check the structure of your pickle file")
                raise ValueError("No graph/tree structure found in the pickle file")

            # Store metadata
            if isinstance(solution_data, dict):
                if 'solution_metadata' in solution_data:
                    self.metadata[algorithm_name] = solution_data['solution_metadata']
                elif 'metadata' in solution_data:
                    self.metadata[algorithm_name] = solution_data['metadata']
                else:
                    self.metadata[algorithm_name] = {}

                # Additional info
                if 'capacity_usage' in solution_data:
                    self.metadata[algorithm_name]['capacity_usage'] = solution_data['capacity_usage']
                if 'discretionary_used' in solution_data:
                    self.metadata[algorithm_name]['discretionary_used'] = solution_data['discretionary_used']
                if 'alpha' in solution_data:
                    self.metadata[algorithm_name]['alpha'] = solution_data['alpha']
            else:
                self.metadata[algorithm_name] = {}

            print(f"✅ Loaded {algorithm_name} from {os.path.basename(filepath)}")
            print(f"   - Nodes: {len(self.trees[algorithm_name].nodes())}")
            print(f"   - Edges: {len(self.trees[algorithm_name].edges())}")
            if 'alpha' in self.metadata[algorithm_name]:
                print(f"   - Alpha: {self.metadata[algorithm_name]['alpha']}")

            return True

        except Exception as e:
            print(f"❌ Error loading {filepath}: {str(e)}")
            return False

    def load_original_graph(self, filepath):
        """Load the original graph (optional, for reference)"""
        try:
            with open(filepath, 'rb') as f:
                self.original_graph = pickle.load(f)
            print(f"✅ Loaded original graph")
            return True
        except Exception as e:
            print(f"❌ Error loading original graph: {str(e)}")
            return False

    def find_path_distance(self, tree, start_node, end_node):
        """
        Find the shortest path and distance between two nodes in a tree

        Returns:
            tuple: (distance, path) or (None, None) if no path exists
        """
        try:
            # Find shortest path
            path = nx.shortest_path(tree, start_node, end_node, weight='weight')

            # Calculate total distance
            distance = 0
            for i in range(len(path) - 1):
                distance += tree[path[i]][path[i+1]]['weight']

            return distance, path

        except nx.NetworkXNoPath:
            return None, None
        except nx.NodeNotFound as e:
            print(f"⚠️  Node not found in tree: {e}")
            return None, None

    def analyze_path(self, start_node, end_node, save_results=True):
        """
        Analyze path between two nodes for all loaded algorithms

        Args:
            start_node: Starting node
            end_node: Ending node
            save_results: Whether to save results to file
        """
        print(f"\n{'='*70}")
        print(f"PATH ANALYSIS: {start_node} → {end_node}")
        print(f"{'='*70}")

        results = {}

        for algo_name, tree in self.trees.items():
            print(f"\n📊 {algo_name}:")
            print("-" * 50)

            # Check if nodes exist in tree
            if start_node not in tree.nodes():
                print(f"   ❌ Start node {start_node} not in this tree")
                results[algo_name] = {'exists': False, 'reason': 'start_node_missing'}
                continue

            if end_node not in tree.nodes():
                print(f"   ❌ End node {end_node} not in this tree")
                results[algo_name] = {'exists': False, 'reason': 'end_node_missing'}
                continue

            # Find path
            distance, path = self.find_path_distance(tree, start_node, end_node)

            if distance is not None:
                print(f"   ✅ Path found!")
                print(f"   📏 Distance: {distance}")
                print(f"   🛤️  Path: {' → '.join(map(str, path))}")
                print(f"   🔢 Number of hops: {len(path) - 1}")

                # Show edge weights
                print(f"   📊 Edge weights:")
                edge_details = []
                for i in range(len(path) - 1):
                    edge_weight = tree[path[i]][path[i+1]]['weight']
                    print(f"      {path[i]} → {path[i+1]}: {edge_weight}")
                    edge_details.append((path[i], path[i+1], edge_weight))

                results[algo_name] = {
                    'exists': True,
                    'distance': distance,
                    'path': path,
                    'hops': len(path) - 1,
                    'edge_details': edge_details
                }
            else:
                print(f"   ❌ No path exists between {start_node} and {end_node}")
                results[algo_name] = {'exists': False, 'reason': 'no_path'}

        # Comparison summary
        self._print_comparison_summary(results, start_node, end_node)

        # Save results if requested
        if save_results:
            self.save_path_analysis(results, start_node, end_node)

        return results

    def _print_comparison_summary(self, results, start_node, end_node):
        """Print a comparison summary of the results"""
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*70}")

        # Filter algorithms with valid paths
        valid_results = {k: v for k, v in results.items() if v['exists']}

        if not valid_results:
            print("❌ No algorithm found a valid path between the nodes")
            return

        # Sort by distance
        sorted_algos = sorted(valid_results.items(), key=lambda x: x[1]['distance'])

        print(f"\n📊 Ranking by distance (shortest first):")
        for rank, (algo, result) in enumerate(sorted_algos, 1):
            print(f"{rank}. {algo}: distance = {result['distance']} (hops = {result['hops']})")

        # Best algorithm
        best_algo, best_result = sorted_algos[0]
        print(f"\n🏆 Best: {best_algo} with distance {best_result['distance']}")

        # Comparison with others
        if len(sorted_algos) > 1:
            print(f"\n📊 Relative differences:")
            for algo, result in sorted_algos[1:]:
                diff = result['distance'] - best_result['distance']
                percent = (diff / best_result['distance']) * 100
                print(f"   {algo} is {diff} longer ({percent:.1f}% more)")

    def visualize_paths(self, start_node, end_node, save_path="path_comparison.png"):
        """
        Visualize the paths for all algorithms
        """
        if not self.trees:
            print("❌ No trees loaded to visualize")
            return

        num_algos = len(self.trees)
        fig, axes = plt.subplots(1, num_algos, figsize=(8*num_algos, 8))

        if num_algos == 1:
            axes = [axes]

        for idx, (algo_name, tree) in enumerate(self.trees.items()):
            ax = axes[idx]

            # Spring layout for the tree
            pos = nx.spring_layout(tree, k=2, iterations=50)

            # Draw all edges in gray
            nx.draw_networkx_edges(tree, pos, ax=ax, edge_color='lightgray', width=1)

            # Draw all nodes
            node_colors = []
            for node in tree.nodes():
                if node == start_node:
                    node_colors.append('green')
                elif node == end_node:
                    node_colors.append('red')
                else:
                    node_colors.append('lightblue')

            nx.draw_networkx_nodes(tree, pos, ax=ax, node_color=node_colors,
                                 node_size=500, alpha=0.9)

            # Draw node labels
            nx.draw_networkx_labels(tree, pos, ax=ax, font_size=10)

            # Highlight path if it exists
            distance, path = self.find_path_distance(tree, start_node, end_node)
            if path:
                # Draw path edges in bold blue
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(tree, pos, ax=ax, edgelist=path_edges,
                                     edge_color='blue', width=3)

                # Add edge labels for path
                edge_labels = {}
                for i in range(len(path)-1):
                    edge_labels[(path[i], path[i+1])] = tree[path[i]][path[i+1]]['weight']

                nx.draw_networkx_edge_labels(tree, pos, edge_labels, ax=ax,
                                           font_size=8, font_color='blue')

            # Title
            title = f"{algo_name}\n"
            if distance is not None:
                title += f"Distance: {distance}, Hops: {len(path)-1}"
            else:
                title += "No path found"
            ax.set_title(title, fontsize=14, weight='bold')
            ax.axis('off')

        plt.suptitle(f"Path Comparison: {start_node} → {end_node}",
                    fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📊 Visualization saved: {save_path}")

    def get_tree_stats(self):
        """Get statistics about loaded trees"""
        print(f"\n{'='*70}")
        print("TREE STATISTICS")
        print(f"{'='*70}")

        for algo_name, tree in self.trees.items():
            print(f"\n📊 {algo_name}:")
            print(f"   - Nodes: {len(tree.nodes())}")
            print(f"   - Edges: {len(tree.edges())}")
            print(f"   - Total edge weight: {sum(data['weight'] for _, _, data in tree.edges(data=True))}")

            if algo_name in self.metadata:
                meta = self.metadata[algo_name]
                if 'alpha' in meta:
                    print(f"   - Alpha: {meta['alpha']}")
                if 'acc_cost' in meta:
                    print(f"   - ACC cost: {meta['acc_cost']:.6f}")
                if 'aoc_cost' in meta:
                    print(f"   - AOC cost: {meta['aoc_cost']:.6f}")
                if 'final_score' in meta:
                    print(f"   - Final score: {meta['final_score']:.2f}")
                if 'execution_time_seconds' in meta:
                    print(f"   - Execution time: {meta['execution_time_seconds']:.6f} seconds")

    def save_path_analysis(self, results, start_node, end_node, filename=None):
        """
        Save path analysis results to file

        Args:
            results: Dictionary with analysis results
            start_node: Starting node
            end_node: Ending node
            filename: Custom filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"path_analysis_{start_node}_to_{end_node}_{timestamp}.txt"

        # Ensure we have a path for saving
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create directory if it doesn't exist
        save_dir = script_dir  # Save in the same directory as the script
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, filename)

        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STEINER TREE PATH DISTANCE ANALYSIS\n")
            f.write("="*80 + "\n\n")

            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Start Node: {start_node}\n")
            f.write(f"End Node: {end_node}\n\n")

            # Algorithm details
            f.write("LOADED ALGORITHMS:\n")
            f.write("-"*40 + "\n")
            for algo_name in self.trees.keys():
                f.write(f"- {algo_name}")
                if algo_name in self.metadata and 'alpha' in self.metadata[algo_name]:
                    f.write(f" (α = {self.metadata[algo_name]['alpha']})")
                f.write("\n")
            f.write("\n")

            # Detailed results for each algorithm
            f.write("DETAILED PATH ANALYSIS:\n")
            f.write("="*60 + "\n\n")

            for algo_name, result in results.items():
                f.write(f"{algo_name}:\n")
                f.write("-"*40 + "\n")

                if result['exists']:
                    f.write(f"✓ Path found\n")
                    f.write(f"  Distance: {result['distance']}\n")
                    f.write(f"  Number of hops: {result['hops']}\n")
                    f.write(f"  Path: {' → '.join(map(str, result['path']))}\n")

                    if 'edge_details' in result:
                        f.write(f"  Edge weights:\n")
                        for u, v, weight in result['edge_details']:
                            f.write(f"    {u} → {v}: {weight}\n")
                else:
                    f.write(f"✗ No path found\n")
                    f.write(f"  Reason: {result.get('reason', 'unknown')}\n")

                f.write("\n")

            # Comparison summary
            f.write("COMPARISON SUMMARY:\n")
            f.write("="*60 + "\n\n")

            valid_results = {k: v for k, v in results.items() if v['exists']}

            if not valid_results:
                f.write("No algorithm found a valid path between the nodes.\n")
            else:
                # Sort by distance
                sorted_algos = sorted(valid_results.items(), key=lambda x: x[1]['distance'])

                f.write("Ranking by distance (shortest first):\n")
                for rank, (algo, result) in enumerate(sorted_algos, 1):
                    f.write(f"{rank}. {algo}: distance = {result['distance']} (hops = {result['hops']})\n")

                # Best algorithm
                best_algo, best_result = sorted_algos[0]
                f.write(f"\nBest algorithm: {best_algo} with distance {best_result['distance']}\n")

                # Relative differences
                if len(sorted_algos) > 1:
                    f.write("\nRelative differences:\n")
                    for algo, result in sorted_algos[1:]:
                        diff = result['distance'] - best_result['distance']
                        percent = (diff / best_result['distance']) * 100
                        f.write(f"  {algo} is {diff} longer ({percent:.1f}% more)\n")

            # Additional metadata if available
            f.write("\n" + "="*60 + "\n")
            f.write("ALGORITHM METADATA:\n")
            f.write("-"*40 + "\n")

            for algo_name in self.trees.keys():
                if algo_name in self.metadata:
                    f.write(f"\n{algo_name}:\n")
                    meta = self.metadata[algo_name]
                    if 'alpha' in meta:
                        f.write(f"  Alpha (α): {meta['alpha']}\n")
                    if 'acc_cost' in meta:
                        f.write(f"  ACC cost: {meta['acc_cost']:.6f}\n")
                    if 'aoc_cost' in meta:
                        f.write(f"  AOC cost: {meta['aoc_cost']:.6f}\n")
                    if 'final_score' in meta:
                        f.write(f"  Final score: {meta['final_score']:.2f}\n")
                    if 'execution_time_seconds' in meta:
                        f.write(f"  Execution time: {meta['execution_time_seconds']:.6f} seconds\n")
                    if 'total_edge_cost' in meta:
                        f.write(f"  Total edge cost: {meta['total_edge_cost']}\n")

        print(f"\n💾 Path analysis saved: {filename}")
        print(f"   📁 Full path: {save_path}")

        # Also save as JSON for programmatic access
        json_filename = filename.replace('.txt', '.json')
        json_path = os.path.join(save_dir, json_filename)

        json_data = {
            'analysis_date': datetime.now().isoformat(),
            'start_node': start_node,
            'end_node': end_node,
            'results': results,
            'metadata': self.metadata
        }

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)

        print(f"💾 JSON data saved: {json_filename}")

        return save_path, json_path

    def save_all_distances_matrix(self, weak_nodes, mandatory_nodes, filename=None):
        """
        Save a matrix of all distances between weak and mandatory nodes

        Args:
            weak_nodes: List of weak nodes
            mandatory_nodes: List of mandatory nodes
            filename: Custom filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"distance_matrix_{timestamp}.txt"

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create directory if it doesn't exist
        save_dir = script_dir  # Save in the same directory as the script
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, filename)

        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPLETE DISTANCE MATRIX ANALYSIS\n")
            f.write("="*80 + "\n\n")

            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Weak nodes analyzed: {len(weak_nodes)}\n")
            f.write(f"Mandatory nodes analyzed: {len(mandatory_nodes)}\n\n")

            for algo_name, tree in self.trees.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"{algo_name}:\n")
                f.write(f"{'='*60}\n\n")

                # Create distance matrix
                distances = {}
                total_distance = 0
                valid_paths = 0
                missing_paths = []

                for weak in weak_nodes:
                    distances[weak] = {}
                    for mandatory in mandatory_nodes:
                        if weak in tree.nodes() and mandatory in tree.nodes():
                            dist, path = self.find_path_distance(tree, weak, mandatory)
                            if dist is not None:
                                distances[weak][mandatory] = dist
                                total_distance += dist
                                valid_paths += 1
                            else:
                                distances[weak][mandatory] = None
                                missing_paths.append((weak, mandatory))
                        else:
                            distances[weak][mandatory] = None
                            missing_paths.append((weak, mandatory))

                # Write matrix header
                f.write("Distance Matrix (rows=weak, cols=mandatory):\n")
                f.write("-"*50 + "\n")

                # Header row
                f.write(f"{'Node':<10}")
                for m in mandatory_nodes:
                    f.write(f"{str(m):<10}")
                f.write("\n")

                # Data rows
                for w in weak_nodes:
                    f.write(f"{str(w):<10}")
                    for m in mandatory_nodes:
                        if distances[w][m] is not None:
                            f.write(f"{distances[w][m]:<10}")
                        else:
                            f.write(f"{'--':<10}")
                    f.write("\n")

                # Statistics
                f.write("\nStatistics:\n")
                f.write("-"*30 + "\n")
                f.write(f"Total valid paths: {valid_paths}\n")
                f.write(f"Missing paths: {len(missing_paths)}\n")

                if valid_paths > 0:
                    avg_distance = total_distance / valid_paths
                    f.write(f"Average distance: {avg_distance:.2f}\n")
                    f.write(f"Total distance sum: {total_distance}\n")

                    # Find min and max
                    all_distances = [d for w_dict in distances.values()
                                   for d in w_dict.values() if d is not None]
                    if all_distances:
                        f.write(f"Minimum distance: {min(all_distances)}\n")
                        f.write(f"Maximum distance: {max(all_distances)}\n")

                if missing_paths:
                    f.write(f"\nMissing paths:\n")
                    for w, m in missing_paths[:10]:  # Show first 10
                        f.write(f"  {w} → {m}\n")
                    if len(missing_paths) > 10:
                        f.write(f"  ... and {len(missing_paths) - 10} more\n")

        print(f"\n💾 Distance matrix saved: {filename}")
        print(f"   📁 Full path: {save_path}")

        return save_path


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SteinerTreeAnalyzer()

    # Get path to graph files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(script_dir, 'graphs/')

    print("🔍 Steiner Tree Path Distance Analyzer")
    print("=" * 70)

    # Load solutions - adjust these paths to your actual files
    print("\n📁 Loading Steiner tree solutions...")

    # Example: Load two different algorithm results
    # Replace these with your actual file paths
    algo1_file = input("Enter path to first algorithm's pickle file: ").strip()
    if os.path.exists(algo1_file):
        analyzer.load_solution(algo1_file, "Algorithm 1")
    else:
        print(f"❌ File not found: {algo1_file}")

    algo2_file = input("Enter path to second algorithm's pickle file: ").strip()
    if os.path.exists(algo2_file):
        analyzer.load_solution(algo2_file, "Algorithm 2")
    else:
        print(f"❌ File not found: {algo2_file}")

    # Optional: Load original graph
    load_original = input("\nLoad original graph? (y/n): ").strip().lower()
    if load_original == 'y':
        original_file = input("Enter path to original graph pickle: ").strip()
        if os.path.exists(original_file):
            analyzer.load_original_graph(original_file)

    # Show statistics
    analyzer.get_tree_stats()

    # Interactive path analysis
    while True:
        print("\n" + "="*70)
        print("PATH ANALYSIS OPTIONS:")
        print("1. Analyze specific path")
        print("2. Visualize paths")
        print("3. Show tree statistics")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            start = input("Enter start node: ").strip()
            end = input("Enter end node: ").strip()

            # Convert to int if needed
            try:
                start = int(start)
                end = int(end)
            except ValueError:
                pass  # Keep as string if not numeric

            results = analyzer.analyze_path(start, end)

            # Ask if user wants to save with custom filename
            save_custom = input("\nSave with custom filename? (y/n, default=n): ").strip().lower()
            if save_custom == 'y':
                custom_name = input("Enter filename (without extension): ").strip()
                if custom_name:
                    analyzer.save_path_analysis(results, start, end, f"{custom_name}.txt")

        elif choice == '2':
            start = input("Enter start node: ").strip()
            end = input("Enter end node: ").strip()

            # Convert to int if needed
            try:
                start = int(start)
                end = int(end)
            except ValueError:
                pass

            save_name = input("Enter filename for visualization (default: path_comparison.png): ").strip()
            if not save_name:
                save_name = "path_comparison.png"

            analyzer.visualize_paths(start, end, save_name)

        elif choice == '3':
            analyzer.get_tree_stats()

        elif choice == '4':
            print("\n👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice")
