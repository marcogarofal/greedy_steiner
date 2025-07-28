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
                'edges': list(solution.steiner_tree.edges()) if solution.steiner_tree else []
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
    def __init__(self, steiner_tree, capacity_usage, connected_weak, failed_connections,
                 total_cost, capacity_cost, discretionary_used, graph_info="",
                 acc_cost=0, aoc_cost=0, alpha=0.5):
        self.steiner_tree = steiner_tree
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
        WITH ENHANCED DEBUGGING
        """
        n = len(graph.nodes())

        # Calculate ACC (Average Communication Cost)
        total_edge_weight = sum(graph[u][v]['weight'] for u, v in selected_edges)
        acc = total_edge_weight / (n * (n - 1)) if n > 1 else 0

        # Calculate AOC (Average Operational Cost) - OVERLOAD ONLY
        total_operational_cost = 0

        print(f"\n    🔍 AOC DETAILED CALCULATION DEBUG:")
        print(f"    ════════════════════════════════════════════")
        print(f"    📊 Graph info:")
        print(f"       - Total nodes in graph (n): {n}")
        print(f"       - Selected nodes: {list(selected_nodes)}")
        print(f"       - Number of selected nodes: {len(selected_nodes)}")
        print(f"       - Selected edges: {selected_edges}")
        print(f"       - Number of selected edges: {len(selected_edges)}")
        print(f"    📊 Capacity usage: {dict(self.capacity_usage)}")
        print(f"    📊 Power capacities defined: {power_capacities}")
        print(f"    ────────────────────────────────────────────")

        node_contributions = []

        for node in selected_nodes:
            # Get capacity info
            max_capacity = power_capacities.get(node, float('inf'))
            current_usage = self.capacity_usage.get(node, 0)

            print(f"\n    🔸 Node {node}:")
            print(f"       - Max capacity: {max_capacity}")
            print(f"       - Current usage: {current_usage}")

            if max_capacity == float('inf'):
                overload_j = 0.0
                print(f"       - Status: NO CAPACITY LIMIT (inf) → overload = 0.0")
            elif max_capacity == 0:
                overload_j = 0.0
                print(f"       - Status: ZERO CAPACITY → overload = 0.0")
            else:
                # OVERLOAD ONLY: only excess usage contributes to cost
                overload_j = max(0.0, current_usage - max_capacity)
                if overload_j > 0:
                    print(f"       - Status: OVERLOADED!")
                    print(f"       - Overload amount: {current_usage} - {max_capacity} = {overload_j}")
                else:
                    print(f"       - Status: BALANCED (within capacity)")
                    print(f"       - Spare capacity: {max_capacity - current_usage}")
                    print(f"       - Overload: 0.0 (no contribution to AOC)")

            # Calculate degree
            d_j = len([edge for edge in selected_edges if node in edge])
            print(f"       - Degree (d_j): {d_j} connections")

            # y_j is always 1 for selected nodes
            y_j = 1 if node in selected_nodes else 0

            # Calculate contribution
            contribution = overload_j * d_j * y_j
            total_operational_cost += contribution

            print(f"       - Contribution: {overload_j:.3f} × {d_j} × {y_j} = {contribution:.3f}")

            node_contributions.append({
                'node': node,
                'capacity': max_capacity,
                'usage': current_usage,
                'overload': overload_j,
                'degree': d_j,
                'contribution': contribution
            })

        # Calculate final AOC
        aoc = total_operational_cost / n if n > 0 else 0

        print(f"\n    ════════════════════════════════════════════")
        print(f"    📊 AOC FINAL CALCULATION:")
        print(f"       - Total operational cost: {total_operational_cost:.6f}")
        print(f"       - Divided by n: {total_operational_cost:.6f} / {n} = {aoc:.6f}")

        # Show why AOC might be 0
        if aoc == 0:
            print(f"    ⚠️  AOC IS ZERO! Possible reasons:")

            overloaded_nodes = [nc for nc in node_contributions if nc['overload'] > 0]
            if len(overloaded_nodes) == 0:
                print(f"       1. NO OVERLOADED NODES - all nodes within capacity")
                print(f"       2. Only overload contributes to AOC in this formula")
            else:
                print(f"       1. Overloaded nodes exist but total_cost/n is too small")
                print(f"       2. With n={n}, even overload of {total_operational_cost} gives {aoc}")

            print(f"\n    💡 DETAILED NODE STATUS:")
            for nc in node_contributions:
                if nc['usage'] > 0:
                    status = "OVERLOADED" if nc['overload'] > 0 else "BALANCED"
                    print(f"       Node {nc['node']}: {nc['usage']}/{nc['capacity']} - {status}")

        # Combined cost function
        cost = alpha * acc + (1 - alpha) * aoc

        print(f"\n    📊 FINAL COSTS:")
        print(f"       - ACC: {acc:.6f}")
        print(f"       - AOC: {aoc:.6f}")
        print(f"       - Combined: {alpha}×{acc:.6f} + {1-alpha}×{aoc:.6f} = {cost:.6f}")
        print(f"    ════════════════════════════════════════════\n")

        return cost, acc, aoc














    def calculate_score(self):
        """
        Calculate a score to compare solutions using the custom cost function
        FIXED: Remove double penalty for capacity violations since AOC already handles overload
        """
        # Get all nodes that are part of the solution
        selected_nodes = set()
        selected_edges = list(self.steiner_tree.edges())

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
    with the new cost function
    """
    steiner_tree = nx.Graph()
    capacity_usage = {node: 0 for node in mandatory_nodes + discretionary_subset}
    connected_weak = set()
    failed_connections = []
    actually_used_discretionary = set()

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
            simulated_tree_edges = list(steiner_tree.edges())

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

            # Calculate the REAL cost difference this would make
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
                    d_j = len([edge for edge in steiner_tree.edges() if node in edge])
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

        # REMOVED: capacity feasibility check - let the cost function handle balance
        # Allow all paths and let the AOC component guide the optimization
        path = option['path']
        target_mandatory = option['target_mandatory']
        discretionary_used = option['discretionary_used']

        capacity_usage[target_mandatory] += 1
        for disc_node in discretionary_used:
            capacity_usage[disc_node] += 1
            actually_used_discretionary.add(disc_node)

        for i in range(len(path) - 1):
            steiner_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

        connected_weak.add(weak_node)
        selected_connections.append(option)

        print(f"    ✓ Connected {weak_node} via {path} (incremental cost: {option['incremental_cost']:.6f})")
        print(f"       └─ Updated capacity_usage: {dict(capacity_usage)}")

    print(f"    📊 After greedy phase: connected {len(connected_weak)}/{len(weak_nodes)} weak nodes")

    # No need for balanced overload phase - greedy should connect all nodes
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
                    steiner_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

                connected_weak.add(weak_node)
                selected_connections.append(chosen_path)

                print(f"    ✓ Fallback connected: {weak_node} via {path}")
            else:
                failed_connections.append(weak_node)
                print(f"    ✗ IMPOSSIBLE to connect {weak_node}")

    # Calculate statistics
    total_cost = sum(graph[u][v]['weight'] for u, v in steiner_tree.edges())

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

    return Solution(steiner_tree, capacity_usage, connected_weak, failed_connections,
                   total_cost, capacity_cost, actually_used_list, graph_info, alpha=alpha)

def find_best_solution_simplified(graph, weak_nodes, mandatory_nodes, all_discretionary_nodes,
                                 power_capacities, alpha=0.5):
    """
    Find the best solution by testing ONLY two cases with custom cost function
    """
    global main_graph
    main_graph = graph  # Store reference for cost function calculation

    print(f"\n{'='*60}")
    print(f"SIMPLIFIED APPROACH WITH CUSTOM COST FUNCTION (α={alpha})")
    print(f"{'='*60}")

    all_solutions = []

    # 1. Solution without discretionary nodes
    print("\n--- Testing solution WITHOUT discretionary nodes ---")
    print(f"Available nodes: mandatory={mandatory_nodes}, discretionary=[]")
    solution_no_disc = solve_with_discretionary_subset(
        graph, weak_nodes, mandatory_nodes, [], power_capacities.copy(),
        "WITHOUT discretionary", alpha
    )
    all_solutions.append(solution_no_disc)
    print(f"📊 SOLUTION WITHOUT discretionary:")
    print(f"   Score: {solution_no_disc.score:.2f}")
    print(f"   Connected: {len(solution_no_disc.connected_weak)}/{len(weak_nodes)}")
    print(f"   Failed: {len(solution_no_disc.failed_connections)}")
    print(f"   ACC: {solution_no_disc.acc_cost:.6f}, AOC: {solution_no_disc.aoc_cost:.6f}")

    # 2. Solution with ALL discretionary nodes
    print(f"\n--- Testing solution WITH ALL discretionary nodes ---")
    print(f"Available nodes: mandatory={mandatory_nodes}, discretionary={all_discretionary_nodes}")

    solution_all_disc = solve_with_discretionary_subset(
        graph, weak_nodes, mandatory_nodes, all_discretionary_nodes, power_capacities.copy(),
        f"WITH ALL discretionary {all_discretionary_nodes}", alpha
    )
    all_solutions.append(solution_all_disc)
    print(f"📊 SOLUTION WITH ALL discretionary:")
    print(f"   Score: {solution_all_disc.score:.2f}")
    print(f"   Connected: {len(solution_all_disc.connected_weak)}/{len(weak_nodes)}")
    print(f"   Failed: {len(solution_all_disc.failed_connections)}")
    print(f"   ACC: {solution_all_disc.acc_cost:.6f}, AOC: {solution_all_disc.aoc_cost:.6f}")
    print(f"   Actually used discretionary: {solution_all_disc.discretionary_used}")

    # 3. Compare and decide
    print(f"\n{'='*60}")
    print(f"DETAILED COMPARISON & SELECTION PROCESS")
    print(f"{'='*60}")

    # Sort solutions by score for detailed comparison
    all_solutions.sort(key=lambda s: s.score)

    print(f"📈 RANKING BY SCORE (lower = better):")
    for i, solution in enumerate(all_solutions):
        rank_symbol = "🏆" if i == 0 else f"#{i+1}"
        status = "SELECTED" if i == 0 else "REJECTED"

        print(f"\n{rank_symbol} {status}: {solution.graph_info}")
        print(f"   Final Score: {solution.score:.2f}")
        print(f"   ├─ Custom Cost Function: {solution.acc_cost + solution.aoc_cost:.6f}")
        print(f"   │  ├─ ACC (α={alpha}): {solution.acc_cost:.6f}")
        print(f"   │  └─ AOC (1-α={1-alpha}): {solution.aoc_cost:.6f}")
        print(f"   ├─ Connection Success: {len(solution.connected_weak)}/{len(weak_nodes)} nodes")
        print(f"   ├─ Failed Connections: {len(solution.failed_connections)} (penalty: {len(solution.failed_connections) * 1000})")
        print(f"   ├─ Edge Cost: {solution.total_cost}")
        print(f"   ├─ Discretionary Used: {solution.discretionary_used}")
        print(f"   └─ Capacity Usage: {dict(solution.capacity_usage)}")

        # Show why this solution was rejected (if not the best)
        if i > 0:
            best_score = all_solutions[0].score
            score_difference = solution.score - best_score
            print(f"   ⚠️  REJECTION REASON: Score {score_difference:.2f} points higher than best solution")

            # Detailed breakdown of score difference
            best_solution = all_solutions[0]

            # Compare individual components
            cost_diff = (solution.acc_cost + solution.aoc_cost) - (best_solution.acc_cost + best_solution.aoc_cost)
            connection_penalty_diff = (len(solution.failed_connections) - len(best_solution.failed_connections)) * 1000

            print(f"   📊 SCORE BREAKDOWN vs BEST:")
            print(f"      ├─ Cost Function Difference: {cost_diff:.6f} * 1000 = {cost_diff * 1000:.2f}")
            print(f"      ├─ Connection Penalty Difference: {connection_penalty_diff:.2f}")

            # Check for capacity violations
            violation_diff = 0
            for node, usage in solution.capacity_usage.items():
                max_cap = power_capacities.get(node, float('inf'))
                if usage > max_cap and max_cap != float('inf'):
                    best_usage = best_solution.capacity_usage.get(node, 0)
                    best_violation = max(0, best_usage - max_cap) if max_cap != float('inf') else 0
                    curr_violation = max(0, usage - max_cap)
                    violation_diff += (curr_violation - best_violation) * 50  # Updated penalty calculation

            if violation_diff > 0:
                print(f"      ├─ Capacity Violation Difference: {violation_diff:.2f}")
            print(f"      └─ Total Difference: {score_difference:.2f}")

    best_solution = all_solutions[0]

    print(f"\n🎯 FINAL SELECTION:")
    print(f"   Best solution: {best_solution.graph_info}")
    print(f"   Final score: {best_solution.score:.2f}")
    print(f"   Why it's best: {('Lowest score' if len(all_solutions) > 1 else 'Only feasible solution')}")

    # Show insight about discretionary usage
    if len(all_solutions) >= 2:
        all_disc_solution = [s for s in all_solutions if s.graph_info.startswith("WITH ALL")][0]
        print(f"\n💡 DISCRETIONARY USAGE INSIGHT:")
        print(f"   Available discretionary nodes: {all_discretionary_nodes}")
        print(f"   Actually used by algorithm: {all_disc_solution.discretionary_used}")
        unused = set(all_discretionary_nodes) - set(all_disc_solution.discretionary_used)
        if unused:
            print(f"   Unused discretionary nodes: {sorted(list(unused))}")
            print(f"   → These nodes didn't improve the solution quality")
        else:
            print(f"   → All available discretionary nodes were beneficial")

        # Additional analysis for identical solutions
        if all_solutions[0].score == all_solutions[1].score:
            print(f"\n🔍 IDENTICAL SOLUTIONS ANALYSIS:")
            print(f"   Both solutions found exactly the same result!")
            print(f"   This means the greedy algorithm with discretionary available")
            print(f"   still chose the same paths as without discretionary.")
            print(f"   📊 Possible reasons:")
            print(f"   - Direct paths have better incremental cost than discretionary paths")
            print(f"   - Discretionary paths don't improve the overall cost function")
            print(f"   - The graph topology makes discretionary paths suboptimal")

    return best_solution, all_solutions

def visualize_best_solution(graph, best_solution, weak_nodes, mandatory_nodes, all_discretionary_nodes, save_name="BEST_SOLUTION"):
    """
    Visualize the best solution found (updated for custom cost function)
    """
    global plot_counter
    plt.figure(figsize=(18, 14))

    pos = nx.spring_layout(graph, weight='weight', k=3, iterations=100)

    steiner_tree = best_solution.steiner_tree
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
    if steiner_tree.edges():
        nx.draw_networkx_edges(steiner_tree, pos, edge_color='blue', width=6, alpha=1.0)

    # Edge labels (same as before)
    solution_edges = set(steiner_tree.edges())

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
        f.write(f"Solution edges: {list(best_solution.steiner_tree.edges())}\n")
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


        print(f"Node capacities (used for AOC calculation): {power_capacities}")

        # Find best solution with custom cost function
        best_solution, all_solutions = find_best_solution_simplified(
            graph, weak_nodes_list, mandatory_nodes_list, discretionary_nodes_list,
            power_capacities, alpha
        )

        # Visualize and save results
        visualize_best_solution(graph, best_solution, weak_nodes_list, mandatory_nodes_list,
                               discretionary_nodes_list, f"steiner_GRAPH_{graph_index}_CUSTOM_COST")

        save_solution_summary(best_solution, all_solutions, f"steiner_GRAPH_{graph_index}_custom_cost_summary")

        # SALVATAGGIO PICKLE INLINE - GUARANTEED TO WORK
        print("💾 Saving solution tree as pickle...")
        pickle_filename = f"steiner_GRAPH_{graph_index}_CUSTOM_COST_solution.pickle"
        pickle_filepath = os.path.join(path, pickle_filename)

        # Create solution data
        solution_data = {
            'steiner_tree': best_solution.steiner_tree,
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
            'solution_edges': list(best_solution.steiner_tree.edges()),
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
        print(f"  - Graph visualization: steiner_GRAPH_{graph_index}_CUSTOM_COST_XXX.png")
        print(f"  - Solution summary: steiner_GRAPH_{graph_index}_custom_cost_summary.txt")
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
