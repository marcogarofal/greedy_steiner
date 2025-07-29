import networkx as nx
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import pickle
import os
import datetime
import json

# Add matplotlib backend to avoid display errors
import matplotlib
matplotlib.use('Agg')

# Path for plots
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'graphs/')

class ConfigurationLogger:
    """Logger for tracking all configuration results"""

    def __init__(self, base_filename="configuration_results"):
        self.base_filename = base_filename
        self.results = []
        self.current_config = None

    def start_configuration(self, config_params):
        """Start logging a new configuration"""
        self.current_config = {
            'timestamp': datetime.datetime.now().isoformat(),
            'parameters': config_params.copy(),
            'solutions': [],
            'performance_metrics': {},
            'execution_log': []
        }

    def log_message(self, message, level="INFO"):
        """Log a message for current configuration"""
        if self.current_config:
            self.current_config['execution_log'].append({
                'timestamp': datetime.datetime.now().isoformat(),
                'level': level,
                'message': message
            })

    def add_solution(self, solution, solution_type):
        """Add a solution result to current configuration"""
        if self.current_config:
            solution_data = {
                'type': solution_type,
                'score': solution.score,
                'acc_cost': solution.acc_cost,
                'aoc_cost': solution.aoc_cost,
                'total_cost': solution.total_cost,
                'connected_weak': len(solution.connected_weak),
                'failed_connections': len(solution.failed_connections),
                'discretionary_used': solution.discretionary_used,
                'capacity_usage': dict(solution.capacity_usage),
                'edges': list(solution.dijistra_tree.edges()) if solution.dijistra_tree else []
            }
            self.current_config['solutions'].append(solution_data)

    def set_performance_metrics(self, metrics):
        """Set performance metrics for current configuration"""
        if self.current_config:
            self.current_config['performance_metrics'] = metrics

    def finish_configuration(self):
        """Finish current configuration and add to results"""
        if self.current_config:
            self.results.append(self.current_config.copy())
            self.current_config = None

    def save_detailed_log(self, filename=None):
        """Save detailed log with all configurations"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.base_filename}_detailed_{timestamp}.txt"

        filepath = os.path.join(path, filename)

        with open(filepath, 'w') as f:
            f.write("="*100 + "\n")
            f.write("COMPREHENSIVE CONFIGURATION ANALYSIS LOG\n")
            f.write("="*100 + "\n\n")

            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Configurations Tested: {len(self.results)}\n\n")

            # Summary table of all configurations
            f.write("CONFIGURATION SUMMARY TABLE:\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Config':<8} {'Alpha':<7} {'Graph':<7} {'Best Score':<12} {'Connected':<10} {'Discretionary':<15} {'Status':<10}\n")
            f.write("-" * 90 + "\n")

            for i, config in enumerate(self.results):
                best_solution = min(config['solutions'], key=lambda s: s['score']) if config['solutions'] else None
                if best_solution:
                    status = "SUCCESS" if best_solution['failed_connections'] == 0 else f"PARTIAL ({best_solution['failed_connections']} failed)"
                    f.write(f"{i+1:<8} {config['parameters'].get('alpha', 'N/A'):<7} "
                           f"{config['parameters'].get('graph_index', 'N/A'):<7} "
                           f"{best_solution['score']:<12.2f} "
                           f"{best_solution['connected_weak']:<10} "
                           f"{str(best_solution['discretionary_used']):<15} "
                           f"{status:<10}\n")
                else:
                    f.write(f"{i+1:<8} {config['parameters'].get('alpha', 'N/A'):<7} "
                           f"{config['parameters'].get('graph_index', 'N/A'):<7} {'FAILED':<12} "
                           f"{'0':<10} {'[]':<15} {'ERROR':<10}\n")

            f.write("\n\n")

            # Detailed analysis for each configuration
            for i, config in enumerate(self.results):
                f.write("="*100 + "\n")
                f.write(f"CONFIGURATION #{i+1} - DETAILED ANALYSIS\n")
                f.write("="*100 + "\n\n")

                # Parameters
                f.write("CONFIGURATION PARAMETERS:\n")
                f.write("-" * 30 + "\n")
                for key, value in config['parameters'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

                # Solutions comparison
                if config['solutions']:
                    f.write("SOLUTIONS COMPARISON:\n")
                    f.write("-" * 50 + "\n")

                    sorted_solutions = sorted(config['solutions'], key=lambda s: s['score'])

                    for j, sol in enumerate(sorted_solutions):
                        status = "🏆 SELECTED" if j == 0 else f"#{j+1} REJECTED"
                        f.write(f"\n{status}: {sol['type']}\n")
                        f.write(f"  Final Score: {sol['score']:.2f}\n")
                        f.write(f"  Custom Cost Function: {sol['acc_cost'] + sol['aoc_cost']:.6f}\n")
                        f.write(f"    ├─ ACC: {sol['acc_cost']:.6f}\n")
                        f.write(f"    └─ AOC: {sol['aoc_cost']:.6f}\n")
                        f.write(f"  Connection Success: {sol['connected_weak']} nodes\n")
                        f.write(f"  Failed Connections: {sol['failed_connections']}\n")
                        f.write(f"  Edge Cost: {sol['total_cost']}\n")
                        f.write(f"  Discretionary Used: {sol['discretionary_used']}\n")
                        f.write(f"  Capacity Usage: {sol['capacity_usage']}\n")
                        f.write(f"  Solution Edges: {sol['edges']}\n")

                        if j > 0:
                            best_score = sorted_solutions[0]['score']
                            score_diff = sol['score'] - best_score
                            f.write(f"  REJECTION REASON: Score {score_diff:.2f} points higher than best\n")

                # Performance metrics
                if config['performance_metrics']:
                    f.write("\nPERFORMANCE METRICS:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in config['performance_metrics'].items():
                        f.write(f"{key}: {value}\n")

                # Execution log (last 10 messages)
                if config['execution_log']:
                    f.write("\nEXECUTION LOG (Key Events):\n")
                    f.write("-" * 30 + "\n")
                    for log_entry in config['execution_log'][-10:]:  # Last 10 entries
                        f.write(f"[{log_entry['level']}] {log_entry['message']}\n")

                f.write("\n\n")

        print(f"📊 Detailed configuration log saved: {filename}")
        return filepath

    def save_json_export(self, filename=None):
        """Save results in JSON format for external analysis"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.base_filename}_export_{timestamp}.json"

        filepath = os.path.join(path, filename)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"📊 JSON export saved: {filename}")
        return filepath

# Global configuration logger
config_logger = ConfigurationLogger()

class Node:
    def __init__(self, name, node_type, capacity=0, weight=0, operational_cost=1):
        self.name = name
        self.node_type = node_type
        self.capacity = capacity
        self.weight = weight
        self.operational_cost = operational_cost

class Solution:
    def __init__(self, dijistra_tree, capacity_usage, connected_weak, failed_connections,
                 total_cost, capacity_cost, discretionary_used, graph_info="",
                 acc_cost=0, aoc_cost=0, alpha=0.5):
        self.dijistra_tree = dijistra_tree
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

        # Calculate overall score using the new cost function
        self.score = self.calculate_score()





    def calculate_cost_function(self, graph, selected_edges, selected_nodes, alpha=0.5):
        """
        Calculate the custom cost function C(G) = α * ACC + (1-α) * AOC
        WITH BEST NORMALIZED AOC (0-1 range)
        """
        n = len(graph.nodes())

        # Calculate ACC (Average Communication Cost) - già normalizzato
        total_edge_weight = sum(graph[u][v]['weight'] for u, v in selected_edges)
        acc = total_edge_weight / (n * (n - 1)) if n > 1 else 0

        # Calculate AOC (Average Operational Cost) - VERSIONE NORMALIZZATA
        # Approccio: Media pesata delle saturazioni dei nodi
        
        print(f"\n    🔍 AOC NORMALIZED CALCULATION (0-1 range):")
        print(f"    ════════════════════════════════════════════")
        print(f"    📊 Graph info:")
        print(f"       - Total nodes in graph: {n}")
        print(f"       - Selected nodes: {len(selected_nodes)}")
        print(f"       - Selected edges: {len(selected_edges)}")
        print(f"    ────────────────────────────────────────────")

        # Calcola il contributo di ogni nodo power (mandatory + discretionary)
        node_saturations = []
        total_weighted_saturation = 0
        total_weight = 0
        
        for node in selected_nodes:
            # Skip weak nodes (they don't have capacity)
            if node not in power_capacities:
                continue
                
            max_capacity = power_capacities.get(node, float('inf'))
            current_usage = self.capacity_usage.get(node, 0)
            
            # Skip nodes with infinite capacity
            if max_capacity == float('inf'):
                continue
                
            # Calculate saturation level (0 = empty, 1 = at capacity, >1 = overloaded)
            if max_capacity > 0:
                saturation = current_usage / max_capacity
            else:
                # Zero capacity node - treat as fully saturated if used
                saturation = 1.0 if current_usage > 0 else 0.0
            
            # Calculate degree (connectivity importance)
            degree = len([edge for edge in selected_edges if node in edge])
            max_possible_degree = n - 1  # Maximum possible degree
            
            # Normalize degree to [0,1]
            normalized_degree = degree / max_possible_degree if max_possible_degree > 0 else 0
            
            # Weight factor: combination of degree importance and base weight
            # Higher degree = more critical node = higher weight
            weight = 1 + normalized_degree  # Weight between 1 and 2
            
            # Saturation contribution (capped at 2.0 to keep reasonable bounds)
            # 0 = no load, 1 = at capacity, 2 = 100% overloaded
            capped_saturation = min(saturation, 2.0)
            
            # Weighted contribution
            weighted_saturation = capped_saturation * weight
            total_weighted_saturation += weighted_saturation
            total_weight += weight
            
            node_saturations.append({
                'node': node,
                'capacity': max_capacity,
                'usage': current_usage,
                'saturation': saturation,
                'capped_saturation': capped_saturation,
                'degree': degree,
                'weight': weight,
                'contribution': weighted_saturation
            })
            
            print(f"\n    🔸 Node {node}:")
            print(f"       - Capacity: {max_capacity}, Usage: {current_usage}")
            print(f"       - Saturation: {saturation:.2f} ({min(saturation * 100, 200):.0f}%)")
            print(f"       - Degree: {degree} (normalized: {normalized_degree:.2f})")
            print(f"       - Weight: {weight:.2f}")
            print(f"       - Contribution: {weighted_saturation:.3f}")

        # Calculate final normalized AOC
        if total_weight > 0:
            # Weighted average of saturations
            aoc_raw = total_weighted_saturation / total_weight
            
            # Map from [0,2] to [0,1] using smooth function
            # Using tanh-like transformation for smooth transition
            if aoc_raw <= 1:
                # Linear mapping for normal operation (0 to 1)
                aoc = aoc_raw
            else:
                # Asymptotic approach to 1 for overload (1 to 2 maps to 1 to ~1)
                # This gives diminishing returns for extreme overload
                aoc = 1 - 0.5 * (2 - aoc_raw)  # Maps [1,2] to [0.5,1]
        else:
            aoc = 0
        
        # Ensure AOC is strictly in [0,1]
        aoc = max(0.0, min(1.0, aoc))
        
        print(f"\n    ════════════════════════════════════════════")
        print(f"    📊 AOC FINAL CALCULATION:")
        print(f"       - Total weighted saturation: {total_weighted_saturation:.3f}")
        print(f"       - Total weight: {total_weight:.3f}")
        print(f"       - Raw AOC: {aoc_raw if total_weight > 0 else 0:.3f}")
        print(f"       - Normalized AOC: {aoc:.6f}")
        
        # Interpretation guide
        if aoc < 0.3:
            status = "EXCELLENT (low load)"
        elif aoc < 0.6:
            status = "GOOD (moderate load)"
        elif aoc < 0.8:
            status = "WARNING (high load)"
        else:
            status = "CRITICAL (overloaded)"
        
        print(f"       - Status: {status}")
        
        # Combined cost function
        cost = alpha * acc + (1 - alpha) * aoc
        
        print(f"\n    📊 FINAL NORMALIZED COSTS:")
        print(f"       - ACC (normalized): {acc:.6f} ∈ [0,1]")
        print(f"       - AOC (normalized): {aoc:.6f} ∈ [0,1]")
        print(f"       - Combined: {alpha}×{acc:.6f} + {1-alpha}×{aoc:.6f} = {cost:.6f}")
        print(f"    ════════════════════════════════════════════\n")
        
        return cost, acc, aoc


    # VERSIONE SEMPLIFICATA per sostituire solo la parte AOC nel codice esistente
    def calculate_aoc_normalized(selected_nodes, selected_edges, capacity_usage, power_capacities, n):
        """
        Calcola AOC normalizzato in [0,1]
        0 = nessun carico, 1 = sistema completamente sovraccarico
        """
        total_weighted_saturation = 0
        total_weight = 0
        
        for node in selected_nodes:
            # Skip nodes without capacity info
            if node not in power_capacities:
                continue
                
            max_capacity = power_capacities.get(node, float('inf'))
            current_usage = capacity_usage.get(node, 0)
            
            if max_capacity == float('inf'):
                continue
                
            # Saturation level
            if max_capacity > 0:
                saturation = current_usage / max_capacity
            else:
                saturation = 1.0 if current_usage > 0 else 0.0
            
            # Node importance (based on degree)
            degree = len([e for e in selected_edges if node in e])
            weight = 1 + (degree / (n - 1))  # Weight between 1 and 2
            
            # Capped saturation (max 2x overload)
            capped_saturation = min(saturation, 2.0)
            
            total_weighted_saturation += capped_saturation * weight
            total_weight += weight
        
        if total_weight > 0:
            aoc_raw = total_weighted_saturation / total_weight
            # Map [0,2] to [0,1]
            if aoc_raw <= 1:
                aoc = aoc_raw
            else:
                aoc = 1 - 0.5 * (2 - aoc_raw)
        else:
            aoc = 0
        
        return max(0.0, min(1.0, aoc))












    def calculate_score(self):
        """
        Calculate a score to compare solutions using the custom cost function
        FIXED: Remove double penalty for capacity violations since AOC already handles overload
        """
        # Get all nodes that are part of the solution
        selected_nodes = set()
        selected_edges = list(self.dijistra_tree.edges())

        # Add all nodes from selected edges
        for u, v in selected_edges:
            selected_nodes.add(u)
            selected_nodes.add(v)

        # Calculate the custom cost function
        try:
            # Use a reference to the main graph
            cost_func_value, acc, aoc = self.calculate_cost_function(
                main_graph, selected_edges, selected_nodes, self.alpha
            )
            self.acc_cost = acc
            self.aoc_cost = aoc
        except:
            # Fallback to simple calculation if main_graph not available
            cost_func_value = self.total_cost / 1000  # Normalize edge cost
            self.acc_cost = cost_func_value
            self.aoc_cost = 0

        # Add penalties for constraints violations

        # 1. Penalty for unconnected nodes (very high penalty - this is critical)
        connection_penalty = len(self.failed_connections) * 1000

        # 2. REMOVED capacity violation penalty - AOC already handles this!
        # The custom cost function's AOC component already penalizes overload
        # Adding extra penalties would be double-counting
        violation_penalty = 0

        # Alternative: Only penalize EXTREME overload cases (optional)
        # extreme_overload_penalty = 0
        # for node, usage in self.capacity_usage.items():
        #     max_cap = power_capacities.get(node, float('inf'))
        #     if usage > max_cap * 2 and max_cap != float('inf'):  # Only if usage is MORE than double capacity
        #         extreme_overload = usage - (max_cap * 2)
        #         extreme_overload_penalty += extreme_overload * 10
        # violation_penalty = extreme_overload_penalty

        # 3. Connectivity constraint penalty (ensure the graph is connected)
        connectivity_penalty = 0
        if len(selected_edges) > 0:
            temp_graph = nx.Graph()
            temp_graph.add_edges_from(selected_edges)
            if not nx.is_connected(temp_graph):
                connectivity_penalty = 500

        # Total score combining custom cost function with constraint penalties
        total_score = cost_func_value * 1000 + connection_penalty + violation_penalty + connectivity_penalty

        # DEBUG: Print score calculation details
        print(f"    🔍 DEBUG SCORE for {self.graph_info}:")
        print(f"        - Custom Cost Function: {cost_func_value:.6f}")
        print(f"          * ACC (α={self.alpha}): {self.acc_cost:.6f}")
        print(f"          * AOC (1-α={1-self.alpha}): {self.aoc_cost:.6f}")
        print(f"        - Cost function × 1000: {cost_func_value * 1000:.2f}")
        print(f"        - Failed nodes: {len(self.failed_connections)} → Connection penalty: {connection_penalty}")
        print(f"        - Capacity violation penalty: {violation_penalty} (REMOVED - handled by AOC)")
        print(f"        - Connectivity penalty: {connectivity_penalty}")
        print(f"        - TOTAL SCORE: {total_score:.2f}")

        # Show overload details for information only (not penalized)
        overload_info = []
        for node, usage in self.capacity_usage.items():
            max_cap = power_capacities.get(node, float('inf'))
            if usage > max_cap and max_cap != float('inf'):
                overload = usage - max_cap
                overload_info.append((node, usage, max_cap, overload))

        if overload_info:
            print(f"        - OVERLOAD INFO (already in AOC, not penalized again):")
            for node, usage, cap, overload in overload_info:
                print(f"          Node {node}: {usage}/{cap} (overload: +{overload})")

        return total_score























    def __str__(self):
        return (f"Solution {self.graph_info}:\n"
                f"  - Connected nodes: {len(self.connected_weak)} (failed: {len(self.failed_connections)})\n"
                f"  - Custom Cost Function: {self.acc_cost:.6f} + {self.aoc_cost:.6f} = {(self.acc_cost + self.aoc_cost):.6f}\n"
                f"  - Edge cost: {self.total_cost}\n"
                f"  - Capacity cost: {self.capacity_cost:.3f}\n"
                f"  - Discretionary ACTUALLY used: {self.discretionary_used}\n"
                f"  - Score: {self.score:.2f}")

def find_all_paths_to_mandatory(graph, weak_node, mandatory_nodes, discretionary_nodes, max_hops=4):
    """
    Find ALL possible paths from a weak node to any mandatory node,
    directly or through discretionary nodes
    """
    all_paths = []

    # 1. Direct paths (weak -> mandatory)
    for mandatory_node in mandatory_nodes:
        if graph.has_edge(weak_node, mandatory_node):
            cost = graph[weak_node][mandatory_node]['weight']
            all_paths.append({
                'path': [weak_node, mandatory_node],
                'cost': cost,
                'target_mandatory': mandatory_node,
                'discretionary_used': []
            })

    # 2. Paths through 1 discretionary node
    for disc_node in discretionary_nodes:
        if graph.has_edge(weak_node, disc_node):
            cost_to_disc = graph[weak_node][disc_node]['weight']

            for mandatory_node in mandatory_nodes:
                if graph.has_edge(disc_node, mandatory_node):
                    total_cost = cost_to_disc + graph[disc_node][mandatory_node]['weight']
                    all_paths.append({
                        'path': [weak_node, disc_node, mandatory_node],
                        'cost': total_cost,
                        'target_mandatory': mandatory_node,
                        'discretionary_used': [disc_node]
                    })

    # 3. Paths through 2 discretionary nodes
    if max_hops >= 3:
        for disc1 in discretionary_nodes:
            if graph.has_edge(weak_node, disc1):
                cost_to_disc1 = graph[weak_node][disc1]['weight']

                for disc2 in discretionary_nodes:
                    if disc1 != disc2 and graph.has_edge(disc1, disc2):
                        cost_disc1_to_disc2 = graph[disc1][disc2]['weight']

                        for mandatory_node in mandatory_nodes:
                            if graph.has_edge(disc2, mandatory_node):
                                total_cost = cost_to_disc1 + cost_disc1_to_disc2 + graph[disc2][mandatory_node]['weight']
                                all_paths.append({
                                    'path': [weak_node, disc1, disc2, mandatory_node],
                                    'cost': total_cost,
                                    'target_mandatory': mandatory_node,
                                    'discretionary_used': [disc1, disc2]
                                })

    all_paths.sort(key=lambda x: x['cost'])
    return all_paths








def solve_with_discretionary_subset(graph, weak_nodes, mandatory_nodes, discretionary_subset,
                                   power_capacities, graph_info="", alpha=0.5):
    """
    Solve the problem using only a specific subset of discretionary nodes
    with the new cost function - FIXED to include all mandatory nodes
    """
    dijistra_tree = nx.Graph()
    capacity_usage = {node: 0 for node in mandatory_nodes + discretionary_subset}
    connected_weak = set()
    failed_connections = []
    actually_used_discretionary = set()

    # CRITICAL FIX: First, ensure all mandatory nodes are in the dijistra tree
    # We need to connect all mandatory nodes together first
    print(f"    🔧 ENSURING ALL MANDATORY NODES ARE CONNECTED:")
    print(f"    Mandatory nodes to connect: {mandatory_nodes}")

    # If there are multiple mandatory nodes, we need to connect them
    if len(mandatory_nodes) > 1:
        # Create a subgraph with only mandatory nodes to find minimum spanning tree
        mandatory_subgraph = graph.subgraph(mandatory_nodes).copy()

        # Check if mandatory nodes form a connected component
        if nx.is_connected(mandatory_subgraph):
            # Find minimum spanning tree connecting all mandatory nodes
            mandatory_mst = nx.minimum_spanning_tree(mandatory_subgraph, weight='weight')

            # Add all edges from the mandatory MST to the dijistra tree
            for u, v in mandatory_mst.edges():
                dijistra_tree.add_edge(u, v, weight=graph[u][v]['weight'])
                print(f"    ✓ Connected mandatory nodes: {u} -- {v} (weight: {graph[u][v]['weight']})")
        else:
            # If mandatory nodes are not directly connected, we need to find paths through other nodes
            print(f"    ⚠️  Mandatory nodes are not directly connected! Finding paths...")

            # Use dijistra tree approximation to connect all mandatory nodes
            # This is a more complex case - we'll use a simple approach
            mandatory_set = set(mandatory_nodes)
            connected_mandatory = set([mandatory_nodes[0]])  # Start with first mandatory node

            while connected_mandatory != mandatory_set:
                # Find shortest path from any connected mandatory to any unconnected mandatory
                best_path = None
                best_cost = float('inf')

                for connected in connected_mandatory:
                    for unconnected in mandatory_set - connected_mandatory:
                        try:
                            path = nx.shortest_path(graph, connected, unconnected, weight='weight')
                            cost = nx.shortest_path_length(graph, connected, unconnected, weight='weight')

                            if cost < best_cost:
                                best_cost = cost
                                best_path = path
                        except nx.NetworkXNoPath:
                            continue

                if best_path:
                    # Add the path to the dijistra tree
                    for i in range(len(best_path) - 1):
                        dijistra_tree.add_edge(best_path[i], best_path[i+1],
                                            weight=graph[best_path[i]][best_path[i+1]]['weight'])

                        # Update capacity usage for intermediate nodes if they're discretionary
                        if best_path[i] in discretionary_subset:
                            capacity_usage[best_path[i]] = capacity_usage.get(best_path[i], 0) + 1
                            actually_used_discretionary.add(best_path[i])
                        if best_path[i+1] in discretionary_subset:
                            capacity_usage[best_path[i+1]] = capacity_usage.get(best_path[i+1], 0) + 1
                            actually_used_discretionary.add(best_path[i+1])

                    # Mark the target mandatory node as connected
                    for node in best_path:
                        if node in mandatory_set:
                            connected_mandatory.add(node)

                    print(f"    ✓ Connected mandatory path: {best_path} (cost: {best_cost})")
                else:
                    print(f"    ❌ ERROR: Cannot connect all mandatory nodes!")
                    break

    elif len(mandatory_nodes) == 1:
        # If there's only one mandatory node, just add it to the tree (it will be connected when we add weak nodes)
        # The node is implicitly in the tree when edges are added to it
        print(f"    ℹ️  Only one mandatory node: {mandatory_nodes[0]} - will be added when connecting weak nodes")

    print(f"    📊 Mandatory nodes connection phase complete")
    print(f"    Current dijistra tree edges: {list(dijistra_tree.edges())}")

    # Now proceed with connecting weak nodes as before
    # Find all possible paths for each weak node
    all_weak_options = {}
    for weak_node in weak_nodes:
        paths = find_all_paths_to_mandatory(graph, weak_node, mandatory_nodes, discretionary_subset)
        all_weak_options[weak_node] = paths

        # Debug: Show all paths found for this weak node
        print(f"    🛤️  Paths found for weak node {weak_node}:")
        for i, path_info in enumerate(paths):
            print(f"       {i+1}. {path_info['path']} → cost: {path_info['cost']}, target: {path_info['target_mandatory']}")

    # Modified greedy algorithm considering the custom cost function
    all_options = []
    for weak_node, paths in all_weak_options.items():
        for path_info in paths:
            # Calculate REAL incremental cost by simulating the effect on the final cost function
            path_edges = [(path_info['path'][i], path_info['path'][i+1])
                         for i in range(len(path_info['path'])-1)]

            # Simulate adding this path to current state
            simulated_capacity_usage = capacity_usage.copy()
            simulated_tree_edges = list(dijistra_tree.edges())

            # Add the path to simulation
            target_mandatory = path_info['target_mandatory']
            discretionary_used = path_info['discretionary_used']
            path = path_info['path']

            # Update simulated capacity usage
            simulated_capacity_usage[target_mandatory] = simulated_capacity_usage.get(target_mandatory, 0) + 1
            for disc_node in discretionary_used:
                simulated_capacity_usage[disc_node] = simulated_capacity_usage.get(disc_node, 0) + 1

            # Add path edges to simulation
            for i in range(len(path) - 1):
                simulated_tree_edges.append((path[i], path[i+1]))

            # Get all nodes in simulated solution
            simulated_selected_nodes = set()
            for u, v in simulated_tree_edges:
                simulated_selected_nodes.add(u)
                simulated_selected_nodes.add(v)

            # Calculate ACC increment
            edge_weight_sum = sum(graph[u][v]['weight'] for u, v in path_edges)
            n = len(graph.nodes())
            incremental_acc = edge_weight_sum / (n * (n - 1)) if n > 1 else 0

            # Calculate AOC increment by comparing before and after
            # Current AOC (only overload from existing nodes)
            current_aoc_cost = 0
            current_power_nodes = [n for n in capacity_usage.keys() if capacity_usage.get(n, 0) > 0]
            for node in current_power_nodes:
                max_capacity = power_capacities.get(node, float('inf'))
                current_usage = capacity_usage.get(node, 0)

                if max_capacity != float('inf') and max_capacity > 0:
                    overload_j = max(0.0, current_usage - max_capacity)  # Only overload
                    d_j = len([edge for edge in dijistra_tree.edges() if node in edge])
                    current_aoc_cost += overload_j * d_j

            # New AOC (only overload after adding this path)
            new_aoc_cost = 0
            new_power_nodes = [n for n in simulated_selected_nodes if n not in weak_nodes]
            for node in new_power_nodes:
                max_capacity = power_capacities.get(node, float('inf'))
                new_usage = simulated_capacity_usage.get(node, 0)

                if max_capacity != float('inf') and max_capacity > 0:
                    overload_j = max(0.0, new_usage - max_capacity)  # Only overload
                    d_j = len([edge for edge in simulated_tree_edges if node in edge])
                    new_aoc_cost += overload_j * d_j

            incremental_aoc = (new_aoc_cost - current_aoc_cost) / n

            # Combined incremental cost
            incremental_cost = alpha * incremental_acc + (1 - alpha) * incremental_aoc

            all_options.append({
                'weak_node': weak_node,
                'incremental_cost': incremental_cost,
                'incremental_acc': incremental_acc,
                'incremental_aoc': incremental_aoc,
                'edge_cost': edge_weight_sum,
                **path_info
            })

    # Sort by incremental cost instead of simple edge cost
    all_options.sort(key=lambda x: x['incremental_cost'])

    # Debug: Show all options ordered by incremental cost
    print(f"    📊 All path options ordered by incremental cost:")
    for i, option in enumerate(all_options):
        print(f"       {i+1}. {option['weak_node']} via {option['path']} → "
              f"inc_cost: {option['incremental_cost']:.6f} "
              f"(ACC: {option['incremental_acc']:.6f}, AOC: {option['incremental_aoc']:.6f}, "
              f"edge_cost: {option['edge_cost']})")

    selected_connections = []

    # Try to connect using lowest incremental cost paths
    print(f"    🎯 Connecting weak nodes using greedy approach:")
    for option in all_options:
        weak_node = option['weak_node']

        if weak_node in connected_weak:
            print(f"       ⏭️  Skipping {weak_node} (already connected)")
            continue

        path = option['path']
        target_mandatory = option['target_mandatory']
        discretionary_used = option['discretionary_used']

        capacity_usage[target_mandatory] += 1
        for disc_node in discretionary_used:
            capacity_usage[disc_node] += 1
            actually_used_discretionary.add(disc_node)

        for i in range(len(path) - 1):
            dijistra_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

        connected_weak.add(weak_node)
        selected_connections.append(option)

        print(f"    ✓ Connected {weak_node} via {path} (incremental cost: {option['incremental_cost']:.6f})")
        print(f"       └─ Updated capacity_usage: {dict(capacity_usage)}")

    print(f"    📊 After greedy phase: connected {len(connected_weak)}/{len(weak_nodes)} weak nodes")

    # Handle remaining weak nodes
    remaining_weak = set(weak_nodes) - connected_weak

    if remaining_weak:
        print(f"    ⚠️  Warning: Some weak nodes not connected: {remaining_weak}")
        print(f"    This should not happen with capacity constraints removed!")

        # Simple fallback: connect remaining nodes with their best available path
        for weak_node in remaining_weak:
            available_paths = all_weak_options.get(weak_node, [])
            if available_paths:
                # Just take the first (lowest cost) path
                chosen_path = available_paths[0]
                target_mandatory = chosen_path['target_mandatory']
                discretionary_used = chosen_path['discretionary_used']
                path = chosen_path['path']

                capacity_usage[target_mandatory] += 1
                for disc_node in discretionary_used:
                    capacity_usage[disc_node] += 1
                    actually_used_discretionary.add(disc_node)

                for i in range(len(path) - 1):
                    dijistra_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

                connected_weak.add(weak_node)
                selected_connections.append(chosen_path)

                print(f"    ✓ Fallback connected: {weak_node} via {path}")
            else:
                failed_connections.append(weak_node)
                print(f"    ✗ IMPOSSIBLE to connect {weak_node}")

    # FINAL CHECK: Ensure all mandatory nodes are in the final tree
    nodes_in_tree = set()
    for u, v in dijistra_tree.edges():
        nodes_in_tree.add(u)
        nodes_in_tree.add(v)

    missing_mandatory = set(mandatory_nodes) - nodes_in_tree
    if missing_mandatory:
        print(f"    ❌ ERROR: Missing mandatory nodes in final tree: {missing_mandatory}")
        print(f"    🔧 This should not happen after the fix!")
    else:
        print(f"    ✅ All mandatory nodes are included in the final tree")

    # Calculate statistics
    total_cost = sum(graph[u][v]['weight'] for u, v in dijistra_tree.edges())

    # Calculate capacity cost
    capacity_cost = 0
    nodes_actually_used = [n for n in capacity_usage if capacity_usage[n] > 0 and power_capacities.get(n, 0) > 0]

    if nodes_actually_used:
        capacity_ratios = []
        for node in nodes_actually_used:
            if power_capacities[node] > 0:
                ratio = capacity_usage[node] / power_capacities[node]
                capacity_ratios.append(ratio)

        capacity_cost = sum(capacity_ratios) / len(capacity_ratios)

    actually_used_list = sorted(list(actually_used_discretionary))

    return Solution(dijistra_tree, capacity_usage, connected_weak, failed_connections,
                   total_cost, capacity_cost, actually_used_list, graph_info, alpha=alpha)











def solve_dijkstra_all_nodes(graph, weak_nodes, mandatory_nodes, discretionary_nodes,
                            power_capacities, graph_info="", alpha=0.5):
    """
    Solve the problem treating ALL nodes (mandatory + discretionary) as required nodes.
    This is a Dijkstra-like approach where the final tree must include ALL power nodes.
    """
    dijistra_tree = nx.Graph()
    
    # ALL power nodes (mandatory + discretionary) must be in the final tree
    all_power_nodes = mandatory_nodes + discretionary_nodes
    capacity_usage = {node: 0 for node in all_power_nodes}
    connected_weak = set()
    failed_connections = []
    
    print(f"    🔧 DIJKSTRA-LIKE APPROACH: All power nodes must be included")
    print(f"    Mandatory nodes: {mandatory_nodes}")
    print(f"    Discretionary nodes: {discretionary_nodes}")
    print(f"    Total power nodes to connect: {len(all_power_nodes)}")
    
    # Step 1: Connect all power nodes (mandatory + discretionary) together
    print(f"\n    📊 PHASE 1: Connecting ALL power nodes together")
    
    if len(all_power_nodes) > 1:
        # Create a subgraph with all power nodes
        power_subgraph = graph.subgraph(all_power_nodes).copy()
        
        if nx.is_connected(power_subgraph):
            # Find minimum spanning tree connecting all power nodes
            power_mst = nx.minimum_spanning_tree(power_subgraph, weight='weight')
            
            # Add all edges from the MST to the Steiner tree
            for u, v in power_mst.edges():
                dijistra_tree.add_edge(u, v, weight=graph[u][v]['weight'])
                print(f"    ✓ Connected power nodes: {u} -- {v} (weight: {graph[u][v]['weight']})")
        else:
            # If power nodes are not directly connected, find paths through other nodes
            print(f"    ⚠️  Power nodes are not directly connected! Finding indirect paths...")
            
            # Use a more sophisticated approach to connect all power nodes
            power_set = set(all_power_nodes)
            connected_power = set([all_power_nodes[0]])  # Start with first power node
            
            while connected_power != power_set:
                # Find shortest path from any connected to any unconnected power node
                best_path = None
                best_cost = float('inf')
                
                for connected in connected_power:
                    for unconnected in power_set - connected_power:
                        try:
                            path = nx.shortest_path(graph, connected, unconnected, weight='weight')
                            cost = nx.shortest_path_length(graph, connected, unconnected, weight='weight')
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_path = path
                        except nx.NetworkXNoPath:
                            continue
                
                if best_path:
                    # Add the path to the Steiner tree
                    for i in range(len(best_path) - 1):
                        dijistra_tree.add_edge(best_path[i], best_path[i+1], 
                                            weight=graph[best_path[i]][best_path[i+1]]['weight'])
                    
                    # Mark power nodes in the path as connected
                    for node in best_path:
                        if node in power_set:
                            connected_power.add(node)
                    
                    print(f"    ✓ Connected power nodes via path: {best_path} (cost: {best_cost})")
                else:
                    print(f"    ❌ ERROR: Cannot connect all power nodes!")
                    break
    
    print(f"    📊 Power nodes connection complete. Current edges: {len(dijistra_tree.edges())}")
    
    # Step 2: Connect weak nodes to the power network
    print(f"\n    📊 PHASE 2: Connecting weak nodes to power network")
    
    # Find all possible paths for each weak node to ANY power node
    all_weak_options = {}
    for weak_node in weak_nodes:
        paths = find_all_paths_to_power_nodes(graph, weak_node, all_power_nodes)
        all_weak_options[weak_node] = paths
        
        print(f"    🛤️  Paths found for weak node {weak_node}: {len(paths)} options")
        for i, path_info in enumerate(paths[:3]):  # Show first 3 paths
            print(f"       {i+1}. {path_info['path']} → cost: {path_info['cost']}, target: {path_info['target_power']}")

    # Calculate incremental costs for all options
    all_options = []
    for weak_node, paths in all_weak_options.items():
        for path_info in paths:
            # Calculate incremental cost similar to before
            path_edges = [(path_info['path'][i], path_info['path'][i+1])
                         for i in range(len(path_info['path'])-1)]
            
            # Simulate adding this path
            simulated_capacity_usage = capacity_usage.copy()
            simulated_tree_edges = list(dijistra_tree.edges())
            
            target_power = path_info['target_power']
            path = path_info['path']
            
            # Update simulated capacity usage
            simulated_capacity_usage[target_power] = simulated_capacity_usage.get(target_power, 0) + 1
            
            # Add path edges to simulation
            for i in range(len(path) - 1):
                simulated_tree_edges.append((path[i], path[i+1]))
            
            # Get all nodes in simulated solution
            simulated_selected_nodes = set()
            for u, v in simulated_tree_edges:
                simulated_selected_nodes.add(u)
                simulated_selected_nodes.add(v)
            
            # Calculate ACC increment
            edge_weight_sum = sum(graph[u][v]['weight'] for u, v in path_edges 
                                if not dijistra_tree.has_edge(u, v))  # Only count new edges
            n = len(graph.nodes())
            incremental_acc = edge_weight_sum / (n * (n - 1)) if n > 1 else 0
            
            # Calculate AOC increment
            current_aoc_cost = 0
            for node in all_power_nodes:
                max_capacity = power_capacities.get(node, float('inf'))
                current_usage = capacity_usage.get(node, 0)
                
                if max_capacity != float('inf') and max_capacity > 0:
                    overload_j = max(0.0, current_usage - max_capacity)
                    d_j = len([edge for edge in dijistra_tree.edges() if node in edge])
                    current_aoc_cost += overload_j * d_j
            
            new_aoc_cost = 0
            for node in all_power_nodes:
                max_capacity = power_capacities.get(node, float('inf'))
                new_usage = simulated_capacity_usage.get(node, 0)
                
                if max_capacity != float('inf') and max_capacity > 0:
                    overload_j = max(0.0, new_usage - max_capacity)
                    d_j = len([edge for edge in simulated_tree_edges if node in edge])
                    new_aoc_cost += overload_j * d_j
            
            incremental_aoc = (new_aoc_cost - current_aoc_cost) / n
            
            # Combined incremental cost
            incremental_cost = alpha * incremental_acc + (1 - alpha) * incremental_aoc
            
            all_options.append({
                'weak_node': weak_node,
                'incremental_cost': incremental_cost,
                'incremental_acc': incremental_acc,
                'incremental_aoc': incremental_aoc,
                'edge_cost': edge_weight_sum,
                **path_info
            })
    
    # Sort by incremental cost
    all_options.sort(key=lambda x: x['incremental_cost'])
    
    # Connect weak nodes using greedy approach
    print(f"\n    🎯 Connecting weak nodes using greedy approach:")
    for option in all_options:
        weak_node = option['weak_node']
        
        if weak_node in connected_weak:
            continue
        
        path = option['path']
        target_power = option['target_power']
        
        # Update capacity usage
        capacity_usage[target_power] += 1
        
        # Add path edges
        new_edges_added = 0
        for i in range(len(path) - 1):
            if not dijistra_tree.has_edge(path[i], path[i+1]):
                dijistra_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])
                new_edges_added += 1
        
        connected_weak.add(weak_node)
        
        print(f"    ✓ Connected {weak_node} to {target_power} via {path}")
        print(f"       └─ New edges added: {new_edges_added}, Incremental cost: {option['incremental_cost']:.6f}")
        print(f"       └─ Updated capacity at {target_power}: {capacity_usage[target_power]}")
    
    # Handle any remaining weak nodes
    remaining_weak = set(weak_nodes) - connected_weak
    for weak_node in remaining_weak:
        failed_connections.append(weak_node)
        print(f"    ✗ Failed to connect {weak_node}")
    
    # Final verification
    nodes_in_tree = set()
    for u, v in dijistra_tree.edges():
        nodes_in_tree.add(u)
        nodes_in_tree.add(v)
    
    missing_power = set(all_power_nodes) - nodes_in_tree
    if missing_power:
        print(f"\n    ❌ ERROR: Missing power nodes in final tree: {missing_power}")
    else:
        print(f"\n    ✅ All {len(all_power_nodes)} power nodes are included in the final tree")
    
    print(f"    ✅ Connected weak nodes: {len(connected_weak)}/{len(weak_nodes)}")
    
    # Calculate statistics
    total_cost = sum(graph[u][v]['weight'] for u, v in dijistra_tree.edges())
    
    # Calculate capacity cost
    capacity_cost = 0
    nodes_actually_used = [n for n in capacity_usage if capacity_usage[n] > 0 and power_capacities.get(n, 0) > 0]
    
    if nodes_actually_used:
        capacity_ratios = []
        for node in nodes_actually_used:
            if power_capacities[node] > 0:
                ratio = capacity_usage[node] / power_capacities[node]
                capacity_ratios.append(ratio)
        
        capacity_cost = sum(capacity_ratios) / len(capacity_ratios)
    
    # For this approach, all discretionary nodes are "used" by definition
    discretionary_used = sorted(discretionary_nodes)
    
    # IMPORTANTE: Calcola ACC e AOC prima di creare la Solution
    # Ottieni tutti i nodi nell'albero
    selected_nodes = set()
    selected_edges = list(dijistra_tree.edges())
    for u, v in selected_edges:
        selected_nodes.add(u)
        selected_nodes.add(v)
    
    # Calcola ACC (Average Communication Cost)
    n = len(graph.nodes())
    total_edge_weight = sum(graph[u][v]['weight'] for u, v in selected_edges)
    acc = total_edge_weight / (n * (n - 1)) if n > 1 else 0
    
    # Calcola AOC (Average Operational Cost) - OVERLOAD ONLY
    total_operational_cost = 0
    
    print(f"\n    📊 CALCOLO FINALE ACC e AOC per Dijkstra:")
    print(f"    ════════════════════════════════════════════")
    print(f"    Nodi totali nel grafo: {n}")
    print(f"    Archi nella soluzione: {len(selected_edges)}")
    print(f"    Peso totale archi: {total_edge_weight}")
    print(f"    ACC = {total_edge_weight} / ({n} × {n-1}) = {acc:.6f}")
    
    # Calcola AOC per ogni nodo power
    for node in all_power_nodes:
        max_capacity = power_capacities.get(node, float('inf'))
        current_usage = capacity_usage.get(node, 0)
        
        if max_capacity != float('inf') and max_capacity > 0:
            overload_j = max(0.0, current_usage - max_capacity)  # Solo overload
            d_j = len([edge for edge in selected_edges if node in edge])
            contribution = overload_j * d_j
            total_operational_cost += contribution
            
            if overload_j > 0:
                print(f"    Nodo {node}: usage={current_usage}, capacity={max_capacity}, "
                      f"overload={overload_j}, degree={d_j}, contribution={contribution:.3f}")
    
    aoc = total_operational_cost / n if n > 0 else 0
    
    print(f"    AOC = {total_operational_cost:.6f} / {n} = {aoc:.6f}")
    print(f"    ════════════════════════════════════════════")
    
    # Ora crea la Solution con i valori ACC e AOC calcolati
    solution = Solution(dijistra_tree, capacity_usage, connected_weak, failed_connections,
                       total_cost, capacity_cost, discretionary_used, graph_info, 
                       acc_cost=acc, aoc_cost=aoc, alpha=alpha)
    
    # Aggiungi l'attributo dijistra_tree per compatibilità con il salvataggio
    solution.dijistra_tree = dijistra_tree
    
    return solution


def find_all_paths_to_power_nodes(graph, weak_node, power_nodes, max_hops=4):
    """
    Find ALL possible paths from a weak node to any power node (mandatory or discretionary).
    Since all power nodes are required, we can connect to any of them.
    """
    all_paths = []
    
    # Direct paths to any power node
    for power_node in power_nodes:
        if graph.has_edge(weak_node, power_node):
            cost = graph[weak_node][power_node]['weight']
            all_paths.append({
                'path': [weak_node, power_node],
                'cost': cost,
                'target_power': power_node,
                'intermediate_nodes': []
            })
    
    # Also consider paths through other nodes (but limit depth)
    if max_hops >= 2:
        for power_node in power_nodes:
            try:
                # Find shortest path
                path = nx.shortest_path(graph, weak_node, power_node, weight='weight')
                if len(path) <= max_hops + 1:  # path includes start and end
                    cost = nx.shortest_path_length(graph, weak_node, power_node, weight='weight')
                    
                    # Check if we already have this as a direct path
                    if len(path) > 2:  # Only add if it's not a direct path
                        all_paths.append({
                            'path': path,
                            'cost': cost,
                            'target_power': power_node,
                            'intermediate_nodes': path[1:-1]  # Nodes between start and end
                        })
            except nx.NetworkXNoPath:
                continue
    
    # Sort by cost
    all_paths.sort(key=lambda x: x['cost'])
    
    # Remove duplicates (keep only the best path to each target)
    seen_targets = set()
    unique_paths = []
    for path_info in all_paths:
        target = path_info['target_power']
        if target not in seen_targets:
            seen_targets.add(target)
            unique_paths.append(path_info)
    
    return unique_paths


def find_best_solution_dijkstra(graph, weak_nodes, mandatory_nodes, discretionary_nodes,
                               power_capacities, alpha=0.5):
    """
    Find the best solution using Dijkstra-like approach where ALL power nodes are included.
    Returns a tuple (best_solution, all_solutions) to match the original function signature.
    """
    global main_graph
    main_graph = graph  # Store reference for cost function calculation
    
    print(f"\n{'='*60}")
    print(f"DIJKSTRA-LIKE APPROACH (ALL NODES INCLUDED)")
    print(f"Alpha = {alpha}")
    print(f"{'='*60}")
    
    # Single solution with all nodes
    solution = solve_dijkstra_all_nodes(
        graph, weak_nodes, mandatory_nodes, discretionary_nodes,
        power_capacities.copy(), "DIJKSTRA (ALL power nodes)", alpha
    )
    
    print(f"\n📊 DIJKSTRA SOLUTION:")
    print(f"   Score: {solution.score:.2f}")
    print(f"   Connected: {len(solution.connected_weak)}/{len(weak_nodes)}")
    print(f"   Failed: {len(solution.failed_connections)}")
    print(f"   ACC: {solution.acc_cost:.6f}, AOC: {solution.aoc_cost:.6f}")
    print(f"   Total edges: {len(solution.dijistra_tree.edges())}")
    print(f"   Total edge cost: {solution.total_cost}")
    
    # Verify all power nodes are included
    nodes_in_tree = set()
    for u, v in solution.dijistra_tree.edges():
        nodes_in_tree.add(u)
        nodes_in_tree.add(v)
    
    all_power_nodes = set(mandatory_nodes + discretionary_nodes)
    included_power = nodes_in_tree & all_power_nodes
    
    print(f"\n📊 POWER NODES INCLUSION:")
    print(f"   Required power nodes: {len(all_power_nodes)}")
    print(f"   Included power nodes: {len(included_power)}")
    print(f"   Mandatory included: {len(set(mandatory_nodes) & nodes_in_tree)}/{len(mandatory_nodes)}")
    print(f"   Discretionary included: {len(set(discretionary_nodes) & nodes_in_tree)}/{len(discretionary_nodes)}")
    
    # Return as tuple to match original function signature
    # Since Dijkstra approach only produces one solution, we return it as both best and all_solutions
    all_solutions = [solution]
    return solution, all_solutions














def visualize_best_solution(graph, best_solution, weak_nodes, mandatory_nodes, all_discretionary_nodes, save_name="BEST_SOLUTION"):
    """
    Visualize the best solution found (updated for custom cost function)
    """
    global plot_counter
    plt.figure(figsize=(18, 14))

    pos = nx.spring_layout(graph, weight='weight', k=3, iterations=100)

    dijistra_tree = best_solution.dijistra_tree
    capacity_usage = best_solution.capacity_usage
    connected_weak = best_solution.connected_weak
    failed_connections = best_solution.failed_connections
    discretionary_used = best_solution.discretionary_used

    # Node colors (same as before)
    node_colors = []
    node_sizes = []

    for node in graph.nodes():
        if node in failed_connections:
            node_colors.append('black')
            node_sizes.append(2000)
        elif node in connected_weak:
            node_colors.append('lightgreen')
            node_sizes.append(1800)
        elif node in weak_nodes:
            node_colors.append('green')
            node_sizes.append(1600)
        elif node in mandatory_nodes:
            max_cap = power_capacities.get(node, float('inf'))
            used_cap = capacity_usage.get(node, 0)
            if used_cap > max_cap:
                node_colors.append('darkred')
            else:
                node_colors.append('red')
            node_sizes.append(1800)
        elif node in discretionary_used:
            max_cap = power_capacities.get(node, float('inf'))
            used_cap = capacity_usage.get(node, 0)
            if used_cap > max_cap:
                node_colors.append('darkorange')
            else:
                node_colors.append('orange')
            node_sizes.append(1700)
        elif node in all_discretionary_nodes:
            node_colors.append('lightgray')
            node_sizes.append(1400)
        else:
            node_colors.append('gray')
            node_sizes.append(1200)

    # Draw base graph
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=node_sizes,
            font_size=12, font_color="black", alpha=0.5, edge_color='lightgray', width=0.5)

    # Highlight solution edges
    if dijistra_tree.edges():
        nx.draw_networkx_edges(dijistra_tree, pos, edge_color='blue', width=6, alpha=1.0)

    # Edge labels (same as before)
    solution_edges = set(dijistra_tree.edges())

    non_solution_edge_labels = {}
    for (u, v) in graph.edges():
        if (u, v) not in solution_edges and (v, u) not in solution_edges:
            non_solution_edge_labels[(u, v)] = graph[u][v]['weight']

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=non_solution_edge_labels, font_size=9,
                                font_color='gray', bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white", alpha=0.8, edgecolor='none'))

    solution_edge_labels = {}
    solution_pos_adjusted = {}

    for (u, v) in solution_edges:
        if graph.has_edge(u, v):
            solution_edge_labels[(u, v)] = graph[u][v]['weight']

            x1, y1 = pos[u]
            x2, y2 = pos[v]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5

            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
                offset = 0.12
                solution_pos_adjusted[(u, v)] = (center_x + perp_x * offset, center_y + perp_y * offset)
            else:
                solution_pos_adjusted[(u, v)] = (center_x, center_y + 0.08)

    for (u, v), label in solution_edge_labels.items():
        if (u, v) in solution_pos_adjusted:
            plt.text(solution_pos_adjusted[(u, v)][0], solution_pos_adjusted[(u, v)][1],
                    str(label), fontsize=11, ha='center', va='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=1.0, edgecolor='blue', linewidth=2))

    # Capacity labels
    capacity_labels = {}
    for node in mandatory_nodes + all_discretionary_nodes:
        used = capacity_usage.get(node, 0)
        max_cap = power_capacities.get(node, 0)
        capacity_labels[node] = f"{used}/{max_cap}"

    capacity_pos = {node: (pos[node][0], pos[node][1] - 0.18) for node in capacity_labels}

    for node, label in capacity_labels.items():
        color = "yellow" if node in discretionary_used or node in mandatory_nodes else "lightgray"
        plt.text(capacity_pos[node][0], capacity_pos[node][1], label,
                fontsize=10, ha='center', va='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9, edgecolor='black'))

    # Updated title with cost function info
    title = (f"🏆 BEST SOLUTION (Custom Cost Function α={best_solution.alpha})\n"
             f"Score: {best_solution.score:.2f} | ACC: {best_solution.acc_cost:.6f} | AOC: {best_solution.aoc_cost:.6f}\n"
             f"Connected: {len(connected_weak)}/{len(weak_nodes)} | Discretionary: {discretionary_used}")

    plt.title(title, fontsize=14, weight='bold', pad=20)

    # Legend (same as before)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=12, label='Connected weak'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Mandatory'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12, label='Discretionary USED'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Discretionary NOT used'),
        plt.Line2D([0], [0], color='blue', linewidth=4, label='BEST solution connections')
    ]

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{path}{save_name}_{plot_counter:03d}_SCORE_{best_solution.score:.0f}.png', dpi=300, bbox_inches='tight')
    plt.close()
    plot_counter += 1
    print(f"🏆 BEST SOLUTION saved with custom cost function")

def save_solution_summary(best_solution, all_solutions, save_name="solution_summary"):
    """
    Save a text summary of solutions with detailed comparison
    """
    with open(f'{path}{save_name}.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SOLUTION SUMMARY WITH CUSTOM COST FUNCTION\n")
        f.write("="*80 + "\n\n")

        f.write("COST FUNCTION: C(G) = α * ACC + (1-α) * AOC\n")
        f.write(f"α = {best_solution.alpha}\n")
        f.write("ACC = (Σ w_ij * x_ij) / (n(n-1)) - Average Communication Cost\n")
        f.write("AOC = (Σ overload_j * d_j * y_j) / n - Average Operational Cost (OVERLOAD ONLY)\n")
        f.write("Where overload_j = max(0, usage_j - capacity_j) - only overload contributes to cost\n")
        f.write("      d_j = node degree, y_j = node selected\n")
        f.write("This formulation favors load distribution over node count minimization\n\n")

        f.write("🏆 BEST SOLUTION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Final score: {best_solution.score:.2f}\n")
        f.write(f"ACC component: {best_solution.acc_cost:.6f}\n")
        f.write(f"AOC component: {best_solution.aoc_cost:.6f}\n")
        f.write(f"Custom cost function value: {best_solution.acc_cost + best_solution.aoc_cost:.6f}\n")
        f.write(f"Discretionary ACTUALLY used: {best_solution.discretionary_used}\n")
        f.write(f"Connected weak nodes: {len(best_solution.connected_weak)}\n")
        f.write(f"Failed connections: {len(best_solution.failed_connections)}\n")
        f.write(f"Total edge cost: {best_solution.total_cost}\n")
        f.write(f"Capacity efficiency cost: {best_solution.capacity_cost:.3f}\n")
        f.write(f"Solution edges: {list(best_solution.dijistra_tree.edges())}\n")
        f.write(f"Capacity usage: {dict(best_solution.capacity_usage)}\n\n")

        f.write("DETAILED COMPARISON OF TESTED SOLUTIONS:\n")
        f.write("-"*50 + "\n")

        # Sort solutions for detailed comparison
        sorted_solutions = sorted(all_solutions, key=lambda s: s.score)

        for i, solution in enumerate(sorted_solutions):
            status = "SELECTED" if i == 0 else "REJECTED"
            f.write(f"\n#{i+1} {status}: {solution.graph_info}\n")
            f.write(f"  Final Score: {solution.score:.2f}\n")
            f.write(f"  Custom Cost (ACC + AOC): {solution.acc_cost + solution.aoc_cost:.6f}\n")
            f.write(f"    ├─ ACC (α={solution.alpha}): {solution.acc_cost:.6f}\n")
            f.write(f"    └─ AOC (1-α={1-solution.alpha}): {solution.aoc_cost:.6f}\n")
            f.write(f"  Connected: {len(solution.connected_weak)}/{len(solution.connected_weak) + len(solution.failed_connections)}\n")
            f.write(f"  Failed connections: {len(solution.failed_connections)}\n")
            f.write(f"  Edge cost: {solution.total_cost}\n")
            f.write(f"  Discretionary used: {solution.discretionary_used}\n")
            f.write(f"  Capacity usage: {dict(solution.capacity_usage)}\n")

            # Explain why rejected (if not the best)
            if i > 0:
                best_score = sorted_solutions[0].score
                score_difference = solution.score - best_score
                f.write(f"  REJECTION REASON: Score {score_difference:.2f} points higher than best\n")

                # Detailed score breakdown
                best_solution_ref = sorted_solutions[0]
                cost_diff = (solution.acc_cost + solution.aoc_cost) - (best_solution_ref.acc_cost + best_solution_ref.aoc_cost)
                connection_penalty_diff = (len(solution.failed_connections) - len(best_solution_ref.failed_connections)) * 1000

                f.write(f"  SCORE BREAKDOWN vs BEST:\n")
                f.write(f"    ├─ Cost Function Difference: {cost_diff:.6f} * 1000 = {cost_diff * 1000:.2f}\n")
                f.write(f"    ├─ Connection Penalty Difference: {connection_penalty_diff:.2f}\n")
                f.write(f"    └─ Total Difference: {score_difference:.2f}\n")

    print(f"📋 Detailed summary saved with rejection reasons")

def save_graph_for_analysis(graph, graph_index, save_name=None):
    """
    Save the loaded graph in JSON format for analysis
    """
    if save_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"graph_{graph_index}_analysis_{timestamp}.json"

    filepath = os.path.join(path, save_name)

    # Convert graph to JSON-serializable format
    graph_data = {
        'graph_info': {
            'nodes_count': len(graph.nodes()),
            'edges_count': len(graph.edges()),
            'graph_index': graph_index
        },
        'nodes': [],
        'edges': [],
        'node_types': {
            'weak': [],
            'power_mandatory': [],
            'power_discretionary': []
        }
    }

    # Extract nodes
    for node_name, data in graph.nodes(data=True):
        node_info = {
            'id': node_name,
            'type': data.get('node_type', 'unknown'),
            'attributes': dict(data)
        }
        graph_data['nodes'].append(node_info)

        # Categorize by type
        node_type = data.get('node_type', 'unknown')
        if node_type in graph_data['node_types']:
            graph_data['node_types'][node_type].append(node_name)

    # Extract edges
    for u, v, data in graph.edges(data=True):
        edge_info = {
            'from': u,
            'to': v,
            'weight': data.get('weight', 1),
            'attributes': dict(data)
        }
        graph_data['edges'].append(edge_info)

    # Add summary statistics
    graph_data['summary'] = {
        'weak_nodes_count': len(graph_data['node_types']['weak']),
        'mandatory_nodes_count': len(graph_data['node_types']['power_mandatory']),
        'discretionary_nodes_count': len(graph_data['node_types']['power_discretionary']),
        'total_edges': len(graph_data['edges']),
        'average_edge_weight': sum(e['weight'] for e in graph_data['edges']) / len(graph_data['edges']) if graph_data['edges'] else 0
    }

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(graph_data, f, indent=2, default=str)

    print(f"📊 Graph saved for analysis: {save_name}")
    print(f"   📋 Summary: {graph_data['summary']['weak_nodes_count']} weak, "
          f"{graph_data['summary']['mandatory_nodes_count']} mandatory, "
          f"{graph_data['summary']['discretionary_nodes_count']} discretionary nodes")
    print(f"   🔗 {graph_data['summary']['total_edges']} edges, "
          f"avg weight: {graph_data['summary']['average_edge_weight']:.2f}")

    return filepath

def draw_graph(G):
    global plot_counter
    plt.figure(figsize=(12, 10))
    plt.clf()

    pos = nx.spring_layout(G, k=2, iterations=50)
    node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
    colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]

    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold',
            node_size=1500, font_size=10, font_color="black")

    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9,
                                font_color='black', bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white", alpha=0.8, edgecolor='gray'))

    plt.title("Base Graph with All Weights")
    plt.savefig(f'{path}base_graph_{plot_counter:03d}.png', dpi=300, bbox_inches='tight')
    plt.close()
    plot_counter += 1

# Global variables
plot_counter = 0
power_capacities = {}
main_graph = None

if __name__ == "__main__":
    # Choose execution mode
    print("🚀 Algorithm Execution Options:")
    print("1 - Single configuration (saves individual PNG + TXT files)")
    print("2 - Multiple configurations test (comprehensive logs)")
    print("3 - Multiple configurations test + individual files (many files!)")

    execution_mode = input("Enter choice (1, 2, or 3): ").strip()

    # Get alpha parameter from user for ALL modes
    try:
        alpha = float(input(f"Enter Alpha parameter (0.0 to 1.0, default 0.5): ").strip() or "0.5")
        if not (0.0 <= alpha <= 1.0):
            print("⚠️ Alpha should be between 0.0 and 1.0, using default 0.5")
            alpha = 0.5
    except ValueError:
        print("⚠️ Invalid alpha value, using default 0.5")
        alpha = 0.5

    print(f"📊 Using Alpha = {alpha} ({'Only ACC (communication cost)' if alpha == 0.0 else 'Only AOC (operational cost)' if alpha == 1.0 else f'Balanced: {alpha*100:.0f}% ACC, {(1-alpha)*100:.0f}% AOC'})")

    if execution_mode == "1":
        # Single configuration mode (original behavior)
        graph_index = 3  # Change this to load different graphs

        print(f"🚀 Running single configuration test...")

        try:
            # Load graph
            file_name = os.path.join(path, f"grafo_{graph_index}.pickle")

            with open(file_name, "rb") as f:
                graph = pickle.load(f)
            print(f"✅ Loaded {file_name}")
        except FileNotFoundError:
            print(f"❌ Error: File {file_name} not found!")
            print("Please make sure the graph file exists or change the graph_index variable.")
            exit(1)

        print(f"\n{'='*80}")
        print(f"PROCESSING GRAPH {graph_index} WITH CUSTOM COST FUNCTION (α={alpha})")
        print(f"{'='*80}")

        # Extract nodes by type
        weak_nodes_list = []
        mandatory_nodes_list = []
        discretionary_nodes_list = []

        for node_name, data in graph.nodes(data=True):
            if data['node_type'] == "weak":
                weak_nodes_list.append(node_name)
            elif data['node_type'] == "power_mandatory":
                mandatory_nodes_list.append(node_name)
            elif data['node_type'] == "power_discretionary":
                discretionary_nodes_list.append(node_name)

        draw_graph(graph)

        # Save graph for analysis
        graph_analysis_file = save_graph_for_analysis(graph, graph_index)

        print(f"Weak nodes: {weak_nodes_list}")
        print(f"Mandatory nodes: {mandatory_nodes_list}")
        print(f"Discretionary nodes: {discretionary_nodes_list}")

        # Node capacities - customize these based on your graph
        #power_capacities = {1: 1, 2: 1, 3: 1, 4: 3, 5: 3, 6: 5, 7: 0}

        '''
        # Extended power capacities from 1 to 30
        power_capacities = {
            # Original values (1-7)
            1: 1,
            2: 1,
            3: 1,
            4: 3,
            5: 3,
            6: 5,
            7: 0,

            # Extended values (8-30)
            8: 2,
            9: 2,
            10: 4,
            11: 3,
            12: 3,
            13: 5,
            14: 2,
            15: 4,
            16: 6,
            17: 3,
            18: 3,
            19: 5,
            20: 4,
            21: 4,
            22: 6,
            23: 3,
            24: 5,
            25: 8,   # Mandatory node - higher capacity
            26: 6,   # Mandatory node - medium capacity
            27: 7,   # Mandatory node - medium-high capacity
            28: 50,   # Discretionary node
            29: 40,   # Discretionary node
            30: 60    # Discretionary node
        }
        '''




        # SCENARIO 1: Capacità molto basse per forzare overload (AOC alto)
        power_capacities = {
            # Nodi 1-7
            1: 1,    # Capacità minima
            2: 1,
            3: 1,
            4: 1,    # Ridotto da 3 a 1
            5: 1,    # Ridotto da 3 a 1
            6: 2,    # Ridotto da 5 a 2
            7: 0,    # Zero capacità
            
            # Nodi 8-24
            8: 1,
            9: 1,
            10: 1,   # Ridotto da 4 a 1
            11: 1,   # Ridotto da 3 a 1
            12: 1,
            13: 1,   # Ridotto da 5 a 1
            14: 1,
            15: 1,   # Ridotto da 4 a 1
            16: 2,   # Ridotto da 6 a 2
            17: 1,
            18: 1,
            19: 1,   # Ridotto da 5 a 1
            20: 1,   # Ridotto da 4 a 1
            21: 1,
            22: 2,   # Ridotto da 6 a 2
            23: 1,
            24: 1,   # Ridotto da 5 a 1
            
            # Nodi mandatory - capacità molto basse per forzare overload
            25: 2,   # Ridotto da 8 a 2 (mandatory)
            26: 2,   # Ridotto da 6 a 2 (mandatory)
            27: 2,   # Ridotto da 7 a 2 (mandatory)
            
            # Nodi discretionary - anche loro con capacità basse
            28: 3,   # Ridotto da 50 a 3 (discretionary)
            29: 3,   # Ridotto da 40 a 3 (discretionary)
            30: 4    # Ridotto da 60 a 4 (discretionary)
        }



        

        print(f"Node capacities (used for AOC calculation): {power_capacities}")

        # Find best solution with custom cost function
        best_solution, all_solutions = find_best_solution_dijkstra(
            graph, weak_nodes_list, mandatory_nodes_list, discretionary_nodes_list,
            power_capacities, alpha
        )

        # Visualize and save results
        visualize_best_solution(graph, best_solution, weak_nodes_list, mandatory_nodes_list,
                               discretionary_nodes_list, f"dijistra_GRAPH_{graph_index}_CUSTOM_COST")

        save_solution_summary(best_solution, all_solutions, f"dijistra_GRAPH_{graph_index}_custom_cost_summary")

        # SALVATAGGIO PICKLE INLINE - GUARANTEED TO WORK
        print("💾 Saving solution tree as pickle...")
        pickle_filename = f"dijistra_GRAPH_{graph_index}_CUSTOM_COST_solution.pickle"
        pickle_filepath = os.path.join(path, pickle_filename)

        # Create solution data
        solution_data = {
            'dijistra_tree': best_solution.dijistra_tree,
            'solution_metadata': {
                'graph_index': graph_index,
                'alpha': best_solution.alpha,
                'final_score': best_solution.score,
                'acc_cost': best_solution.acc_cost,
                'aoc_cost': best_solution.aoc_cost,
                'total_edge_cost': best_solution.total_cost,
                'timestamp': datetime.datetime.now().isoformat()
            },
            'capacity_usage': dict(best_solution.capacity_usage),
            'connected_weak_nodes': list(best_solution.connected_weak),
            'discretionary_used': best_solution.discretionary_used,
            'solution_edges': list(best_solution.dijistra_tree.edges()),
            'power_capacities_used': power_capacities.copy()
        }

        # Save to pickle
        try:
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(solution_data, f)
            print(f"✅ Pickle saved successfully: {pickle_filename}")
            print(f"   📁 Full path: {pickle_filepath}")

            # Verify file exists
            if os.path.exists(pickle_filepath):
                file_size = os.path.getsize(pickle_filepath)
                print(f"   📊 File size: {file_size} bytes")
            else:
                print("   ❌ File not found after saving!")

        except Exception as e:
            print(f"❌ Error saving pickle: {str(e)}")

        print(f"\n🏆 GRAPH {graph_index} COMPLETED (CUSTOM COST FUNCTION)")
        print(f"BEST SOLUTION:")
        print(f"  - Score: {best_solution.score:.2f}")
        print(f"  - ACC: {best_solution.acc_cost:.6f}")
        print(f"  - AOC: {best_solution.aoc_cost:.6f}")
        print(f"  - Discretionary used: {best_solution.discretionary_used}")
        print(f"  - Connected: {len(best_solution.connected_weak)}/{len(weak_nodes_list)}")

        print(f"\n📁 Output files saved:")
        print(f"  - Graph visualization: dijistra_GRAPH_{graph_index}_CUSTOM_COST_XXX.png")
        print(f"  - Solution summary: dijistra_GRAPH_{graph_index}_custom_cost_summary.txt")
        print(f"  - Solution tree pickle: {pickle_filename}")
        print(f"  - Graph analysis file: {graph_analysis_file}")

        print(f"\n💡 You can send me the graph analysis JSON file for detailed analysis!")
        print(f"💾 You can import the solution tree with:")
        print(f"   import pickle")
        print(f"   solution = pickle.load(open('{pickle_filename}', 'rb'))")

    else:
        print("❌ Multiple configurations mode not implemented in this version")
        print("💡 Use execution mode 1 for single configuration testing")
        print("🔧 The code can be extended to support multiple configurations if needed")
