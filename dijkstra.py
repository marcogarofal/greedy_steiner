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

# Ensure graphs directory exists
if not os.path.exists(path):
    os.makedirs(path)
    print(f"📁 Created directory: {path}")

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
            f.write("COMPREHENSIVE CONFIGURATION ANALYSIS LOG (DIJKSTRA ALGORITHM)\n")
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
        """
        n = len(graph.nodes())

        # Calculate ACC (Average Communication Cost)
        total_edge_weight = sum(graph[u][v]['weight'] for u, v in selected_edges)
        acc = total_edge_weight / (n * (n - 1)) if n > 1 else 0

        # Calculate AOC (Average Operational Cost) - OVERLOAD ONLY
        total_operational_cost = 0
        print(f"    🔍 AOC Calculation Debug (OVERLOAD ONLY FORMULA):")
        print(f"       Selected nodes: {list(selected_nodes)}")
        print(f"       Selected edges: {selected_edges}")
        print(f"       Current capacity_usage: {dict(self.capacity_usage)}")

        for node in selected_nodes:
            # Calculate overload_j: only overload contributes to cost
            max_capacity = power_capacities.get(node, float('inf'))
            current_usage = self.capacity_usage.get(node, 0)

            if max_capacity == float('inf') or max_capacity == 0:
                overload_j = 0.0  # No capacity constraints
                print(f"       Node {node}: NO CAPACITY LIMIT → overload = 0.0")
            else:
                # OVERLOAD ONLY: only excess usage contributes to cost
                overload_j = max(0.0, current_usage - max_capacity)
                if overload_j > 0:
                    print(f"       Node {node}: capacity={max_capacity}, usage={current_usage} → OVERLOAD = {overload_j}")
                else:
                    print(f"       Node {node}: capacity={max_capacity}, usage={current_usage} → balanced (no cost)")

            # d_j: degree of node j in the solution (number of connections)
            d_j = len([edge for edge in selected_edges if node in edge])
            # y_j: 1 if node is selected, 0 otherwise
            y_j = 1 if node in selected_nodes else 0

            contribution = overload_j * d_j * y_j
            total_operational_cost += contribution

            print(f"                     d_j={d_j}, y_j={y_j}, contribution = {overload_j:.3f} × {d_j} × {y_j} = {contribution:.3f}")

        aoc = total_operational_cost / n if n > 0 else 0

        # Combined cost function
        cost = alpha * acc + (1 - alpha) * aoc

        print(f"    📊 Final AOC: total_overload_cost={total_operational_cost:.3f} / n={n} = {aoc:.6f}")
        print(f"    📊 Final ACC: {acc:.6f}")
        print(f"    📊 Combined cost: {alpha}×{acc:.6f} + {1-alpha}×{aoc:.6f} = {cost:.6f}")

        return cost, acc, aoc

    def calculate_score(self):
        """
        Calculate a score to compare solutions using the custom cost function
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
        except Exception as e:
            # Fallback to simple calculation if main_graph not available
            cost_func_value = self.total_cost / 1000
            self.acc_cost = cost_func_value
            self.aoc_cost = 0

        # Add penalties for constraints violations
        connection_penalty = len(self.failed_connections) * 1000

        # Penalty for capacity violations
        violation_penalty = 0
        max_overload = 0
        total_overload = 0

        for node, usage in self.capacity_usage.items():
            max_cap = power_capacities.get(node, float('inf'))
            if usage > max_cap and max_cap != float('inf'):
                overload = usage - max_cap
                total_overload += overload
                max_overload = max(max_overload, overload)

        violation_penalty = total_overload * 50 + max_overload * 100

        # Connectivity constraint penalty
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
        print(f"        - Capacity violations: total={total_overload}, max={max_overload} → Penalty: {violation_penalty}")
        print(f"        - Connectivity penalty: {connectivity_penalty}")
        print(f"        - TOTAL SCORE: {total_score:.2f}")

        return total_score

    def __str__(self):
        return (f"Solution {self.graph_info}:\n"
                f"  - Connected nodes: {len(self.connected_weak)} (failed: {len(self.failed_connections)})\n"
                f"  - Custom Cost Function: {self.acc_cost:.6f} + {self.aoc_cost:.6f} = {(self.acc_cost + self.aoc_cost):.6f}\n"
                f"  - Edge cost: {self.total_cost}\n"
                f"  - Capacity cost: {self.capacity_cost:.3f}\n"
                f"  - Discretionary ACTUALLY used: {self.discretionary_used}\n"
                f"  - Score: {self.score:.2f}")

def dijkstra_shortest_path(graph, start, end):
    """
    Find shortest path between start and end using Dijkstra's algorithm
    Returns (path, total_cost) or (None, float('inf')) if no path exists
    """
    if start == end:
        return [start], 0

    # Priority queue: (cost, current_node, path)
    pq = [(0, start, [start])]
    visited = set()

    while pq:
        current_cost, current_node, path = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == end:
            return path, current_cost

        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                edge_weight = graph[current_node][neighbor]['weight']
                new_cost = current_cost + edge_weight
                new_path = path + [neighbor]
                heapq.heappush(pq, (new_cost, neighbor, new_path))

    return None, float('inf')

def find_all_paths_dijkstra(graph, weak_node, all_power_nodes, max_paths=5):
    """
    Find the best paths from a weak node to ALL power nodes using Dijkstra's algorithm
    """
    all_paths = []

    print(f"    🛤️  Finding Dijkstra paths from weak node {weak_node} to ALL power nodes:")

    for power_node in all_power_nodes:
        path, cost = dijkstra_shortest_path(graph, weak_node, power_node)

        if path is not None:
            # Determine which discretionary nodes are used in this path
            discretionary_used = []
            for node in path[1:-1]:  # Exclude start (weak) and end (power) nodes
                if node in discretionary_nodes_list:
                    discretionary_used.append(node)

            path_info = {
                'path': path,
                'cost': cost,
                'target_power': power_node,
                'discretionary_used': discretionary_used
            }

            all_paths.append(path_info)
            print(f"       → {power_node}: {path} (cost: {cost}, discretionary used: {discretionary_used})")
        else:
            print(f"       → {power_node}: NO PATH AVAILABLE")

    # Sort paths by cost and return best ones
    all_paths.sort(key=lambda x: x['cost'])
    return all_paths[:max_paths * len(all_power_nodes)]

def solve_with_dijkstra(graph, weak_nodes, mandatory_nodes, discretionary_nodes,
                       power_capacities_config, graph_info="", alpha=0.5):
    """
    Solve the problem using Dijkstra's algorithm
    """
    print(f"\n{'='*60}")
    print(f"DIJKSTRA ALGORITHM APPROACH")
    print(f"Treating discretionary nodes as mandatory power nodes")
    print(f"{'='*60}")

    # Combine mandatory and discretionary nodes - ALL are now power nodes
    all_power_nodes = mandatory_nodes + discretionary_nodes
    print(f"📊 Total power nodes (mandatory + discretionary): {len(all_power_nodes)}")
    print(f"   Mandatory: {mandatory_nodes}")
    print(f"   Discretionary (now treated as mandatory): {discretionary_nodes}")

    steiner_tree = nx.Graph()
    capacity_usage = {node: 0 for node in all_power_nodes}
    connected_weak = set()
    failed_connections = []
    actually_used_discretionary = set()

    # Find all possible paths for each weak node using Dijkstra
    all_weak_options = {}
    for weak_node in weak_nodes:
        paths = find_all_paths_dijkstra(graph, weak_node, all_power_nodes)
        all_weak_options[weak_node] = paths

    # Create all possible connection options
    all_options = []
    for weak_node, paths in all_weak_options.items():
        for path_info in paths:
            # Calculate incremental cost for this path
            path_edges = [(path_info['path'][i], path_info['path'][i+1])
                         for i in range(len(path_info['path'])-1)]

            # Calculate ACC increment
            edge_weight_sum = sum(graph[u][v]['weight'] for u, v in path_edges)
            n = len(graph.nodes())
            incremental_acc = edge_weight_sum / (n * (n - 1)) if n > 1 else 0

            # Simplified AOC for this version
            incremental_aoc = 0

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

    # Debug: Show all options ordered by incremental cost
    print(f"    📊 All Dijkstra path options ordered by incremental cost:")
    for i, option in enumerate(all_options):
        print(f"       {i+1}. {option['weak_node']} via {option['path']} → "
              f"inc_cost: {option['incremental_cost']:.6f} "
              f"(ACC: {option['incremental_acc']:.6f}, edge_cost: {option['edge_cost']})")

    # Connect weak nodes using greedy approach based on incremental cost
    print(f"    🎯 Connecting weak nodes using Dijkstra + greedy approach:")
    for option in all_options:
        weak_node = option['weak_node']

        if weak_node in connected_weak:
            print(f"       ⏭️  Skipping {weak_node} (already connected)")
            continue

        path = option['path']
        target_power = option['target_power']
        discretionary_used = option['discretionary_used']

        # Update capacity usage
        capacity_usage[target_power] += 1
        for disc_node in discretionary_used:
            capacity_usage[disc_node] += 1
            actually_used_discretionary.add(disc_node)

        # Add path edges to solution
        for i in range(len(path) - 1):
            steiner_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

        connected_weak.add(weak_node)

        print(f"    ✓ Connected {weak_node} via {path} (incremental cost: {option['incremental_cost']:.6f})")
        print(f"       └─ Updated capacity_usage: {dict(capacity_usage)}")

    print(f"    📊 Dijkstra connection phase: connected {len(connected_weak)}/{len(weak_nodes)} weak nodes")

    # Check for failed connections
    remaining_weak = set(weak_nodes) - connected_weak
    for weak_node in remaining_weak:
        failed_connections.append(weak_node)
        print(f"    ✗ FAILED to connect {weak_node}")

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

def find_best_solution_dijkstra(graph, weak_nodes, mandatory_nodes, all_discretionary_nodes,
                               power_capacities_config, alpha=0.5):
    """
    Find the best solution using Dijkstra algorithm
    """
    global main_graph
    main_graph = graph

    print(f"\n{'='*60}")
    print(f"DIJKSTRA APPROACH WITH CUSTOM COST FUNCTION (α={alpha})")
    print(f"All discretionary nodes treated as mandatory power nodes")
    print(f"{'='*60}")

    all_solutions = []

    # With Dijkstra, we only have one scenario: all nodes (mandatory + discretionary) are power nodes
    print(f"\n--- Running Dijkstra algorithm ---")
    print(f"Power nodes: mandatory={mandatory_nodes}, discretionary (as mandatory)={all_discretionary_nodes}")

    solution_dijkstra = solve_with_dijkstra(
        graph, weak_nodes, mandatory_nodes, all_discretionary_nodes, power_capacities_config,
        "DIJKSTRA with all power nodes", alpha
    )
    all_solutions.append(solution_dijkstra)

    print(f"📊 DIJKSTRA SOLUTION:")
    print(f"   Score: {solution_dijkstra.score:.2f}")
    print(f"   Connected: {len(solution_dijkstra.connected_weak)}/{len(weak_nodes)}")
    print(f"   Failed: {len(solution_dijkstra.failed_connections)}")
    print(f"   ACC: {solution_dijkstra.acc_cost:.6f}, AOC: {solution_dijkstra.aoc_cost:.6f}")
    print(f"   Actually used discretionary: {solution_dijkstra.discretionary_used}")

    return solution_dijkstra, all_solutions

def visualize_dijkstra_solution(graph, best_solution, weak_nodes, mandatory_nodes, all_discretionary_nodes, save_name="DIJKSTRA_SOLUTION"):
    """
    Visualize the Dijkstra solution with the same quality as the original code
    """
    global plot_counter
    plt.figure(figsize=(18, 14))

    pos = nx.spring_layout(graph, weight='weight', k=3, iterations=100)

    steiner_tree = best_solution.steiner_tree
    capacity_usage = best_solution.capacity_usage
    connected_weak = best_solution.connected_weak
    failed_connections = best_solution.failed_connections
    discretionary_used = best_solution.discretionary_used

    # Node colors (same as original code)
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

    # Edge labels (same as original)
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

    # Capacity labels (same as original)
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

    # Updated title for Dijkstra
    title = (f"🏆 DIJKSTRA SOLUTION (Custom Cost Function α={best_solution.alpha})\n"
             f"Score: {best_solution.score:.2f} | ACC: {best_solution.acc_cost:.6f} | AOC: {best_solution.aoc_cost:.6f}\n"
             f"Connected: {len(connected_weak)}/{len(weak_nodes)} | Discretionary Used: {discretionary_used}")

    plt.title(title, fontsize=14, weight='bold', pad=20)

    # Legend (same as original)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=12, label='Connected weak'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Mandatory'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12, label='Discretionary USED'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Discretionary NOT used'),
        plt.Line2D([0], [0], color='blue', linewidth=4, label='DIJKSTRA solution connections')
    ]

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)

    plt.tight_layout()

    # Create full filepath and save
    filename = f'{save_name}_{plot_counter:03d}_SCORE_{best_solution.score:.0f}.png'
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    plot_counter += 1

    print(f"🏆 DIJKSTRA SOLUTION saved: {filename}")
    print(f"📁 Full path: {filepath}")

    return filepath

def save_dijkstra_summary(best_solution, all_solutions, save_name="dijkstra_summary"):
    """
    Save a text summary of Dijkstra solutions
    """
    filename = f'{save_name}.txt'
    filepath = os.path.join(path, filename)

    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DIJKSTRA SOLUTION SUMMARY WITH CUSTOM COST FUNCTION\n")
        f.write("="*80 + "\n\n")

        f.write("ALGORITHM: DIJKSTRA (All discretionary nodes treated as mandatory)\n")
        f.write("COST FUNCTION: C(G) = α * ACC + (1-α) * AOC\n")
        f.write(f"α = {best_solution.alpha}\n")
        f.write("ACC = (Σ w_ij * x_ij) / (n(n-1)) - Average Communication Cost\n")
        f.write("AOC = (Σ overload_j * d_j * y_j) / n - Average Operational Cost (OVERLOAD ONLY)\n")
        f.write("Where overload_j = max(0, usage_j - capacity_j) - only overload contributes to cost\n")
        f.write("      d_j = node degree, y_j = node selected\n\n")

        f.write("🏆 DIJKSTRA SOLUTION:\n")
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

        f.write("DIJKSTRA ALGORITHM DETAILS:\n")
        f.write("-"*30 + "\n")
        f.write("• All discretionary nodes treated as mandatory power nodes\n")
        f.write("• Used shortest path algorithm (Dijkstra) to connect each weak node\n")
        f.write("• Greedy selection based on incremental cost function\n")
        f.write("• No Steiner tree optimization - direct shortest paths only\n")

    print(f"📋 Dijkstra summary saved: {filename}")
    print(f"📁 Full path: {filepath}")

    return filepath

def save_solution_as_pickle(best_solution, graph_index, alpha, save_name=None):
    """
    Save the solution tree and all related data as a pickle file for later import
    """
    if save_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"dijkstra_solution_graph_{graph_index}_alpha_{alpha}_{timestamp}.pickle"

    # Create comprehensive solution data structure
    solution_data = {
        # Algorithm info
        'algorithm': 'dijkstra',
        'alpha': alpha,
        'graph_index': graph_index,
        'timestamp': datetime.datetime.now().isoformat(),

        # Solution tree (NetworkX graph)
        'solution_tree': best_solution.steiner_tree,

        # Solution metrics
        'score': best_solution.score,
        'acc_cost': best_solution.acc_cost,
        'aoc_cost': best_solution.aoc_cost,
        'total_cost': best_solution.total_cost,
        'capacity_cost': best_solution.capacity_cost,

        # Node and connection data
        'connected_weak': list(best_solution.connected_weak),
        'failed_connections': list(best_solution.failed_connections),
        'discretionary_used': best_solution.discretionary_used,
        'capacity_usage': dict(best_solution.capacity_usage),

        # Tree structure details
        'solution_edges': list(best_solution.steiner_tree.edges()),
        'solution_nodes': list(best_solution.steiner_tree.nodes()),
        'tree_info': {
            'num_edges': len(best_solution.steiner_tree.edges()),
            'num_nodes': len(best_solution.steiner_tree.nodes()),
            'is_connected': nx.is_connected(best_solution.steiner_tree) if len(best_solution.steiner_tree.edges()) > 0 else False
        },

        # Edge weights in the solution
        'edge_weights': {
            (u, v): best_solution.steiner_tree[u][v]['weight']
            for u, v in best_solution.steiner_tree.edges()
        },

        # Additional metadata
        'graph_info': best_solution.graph_info,
        'power_capacities_used': dict(power_capacities)
    }

    filepath = os.path.join(path, save_name)

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(solution_data, f)

        print(f"🔧 Solution saved as pickle: {save_name}")
        print(f"📁 Full path: {filepath}")
        print(f"💾 Contains: solution tree, metrics, and all data for reuse")

        return filepath

    except Exception as e:
        print(f"❌ Error saving solution pickle: {e}")
        return None

def load_solution_from_pickle(pickle_file_path):
    """
    Load a previously saved solution from pickle file

    Usage example:
        solution_data = load_solution_from_pickle('path/to/solution.pickle')
        tree = solution_data['solution_tree']
        score = solution_data['score']
    """
    try:
        with open(pickle_file_path, 'rb') as f:
            solution_data = pickle.load(f)

        print(f"✅ Solution loaded from: {pickle_file_path}")
        print(f"📊 Algorithm: {solution_data.get('algorithm', 'unknown')}")
        print(f"📊 Score: {solution_data.get('score', 'N/A')}")
        print(f"📊 Alpha: {solution_data.get('alpha', 'N/A')}")
        print(f"🌳 Tree edges: {len(solution_data.get('solution_edges', []))}")
        print(f"🔗 Tree nodes: {len(solution_data.get('solution_nodes', []))}")

        return solution_data

    except Exception as e:
        print(f"❌ Error loading solution pickle: {e}")
        return None

def export_tree_to_graphml(best_solution, graph_index, alpha, save_name=None):
    """
    Export the solution tree to GraphML format (alternative to pickle, more universal)
    """
    if save_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"dijkstra_tree_graph_{graph_index}_alpha_{alpha}_{timestamp}.graphml"

    # Create a copy of the tree with additional attributes
    tree_copy = best_solution.steiner_tree.copy()

    # Add solution metadata as graph attributes
    tree_copy.graph['algorithm'] = 'dijkstra'
    tree_copy.graph['alpha'] = alpha
    tree_copy.graph['score'] = best_solution.score
    tree_copy.graph['acc_cost'] = best_solution.acc_cost
    tree_copy.graph['aoc_cost'] = best_solution.aoc_cost
    tree_copy.graph['total_cost'] = best_solution.total_cost
    tree_copy.graph['graph_index'] = graph_index
    tree_copy.graph['timestamp'] = datetime.datetime.now().isoformat()

    # Add node attributes
    for node in tree_copy.nodes():
        tree_copy.nodes[node]['capacity_usage'] = best_solution.capacity_usage.get(node, 0)
        tree_copy.nodes[node]['max_capacity'] = power_capacities.get(node, 0)
        tree_copy.nodes[node]['is_discretionary_used'] = node in best_solution.discretionary_used
        tree_copy.nodes[node]['is_connected_weak'] = node in best_solution.connected_weak

    filepath = os.path.join(path, save_name)

    try:
        nx.write_graphml(tree_copy, filepath)
        print(f"🌳 Tree exported to GraphML: {save_name}")
        print(f"📁 Full path: {filepath}")
        print(f"💡 Can be imported in other graph tools (Gephi, Cytoscape, etc.)")

        return filepath

    except Exception as e:
        print(f"❌ Error exporting to GraphML: {e}")
        return None
    """Draw base graph with Dijkstra labeling"""
    global plot_counter

    plt.figure(figsize=(12, 10))
    plt.clf()

    pos = nx.spring_layout(G, k=2, iterations=50)

    # Updated colors for Dijkstra: discretionary treated as mandatory
    node_colors = []
    for node, data in G.nodes(data=True):
        if data['node_type'] == 'weak':
            node_colors.append('green')
        elif data['node_type'] == 'power_mandatory':
            node_colors.append('red')
        elif data['node_type'] == 'power_discretionary':
            node_colors.append('orange')
        else:
            node_colors.append('gray')

    nx.draw(G, pos, with_labels=True, node_color=node_colors, font_weight='bold',
            node_size=1500, font_size=10, font_color="black")

    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9,
                                font_color='black', bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white", alpha=0.8, edgecolor='gray'))

    plt.title("Base Graph for Dijkstra Algorithm\n(Discretionary nodes treated as mandatory)")

    filename = f'dijkstra_base_graph_{plot_counter:03d}.png'
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    plot_counter += 1

    print(f"📊 Base graph saved: {filename}")
    print(f"📁 Full path: {filepath}")

    return filepath

def run_multiple_configurations_dijkstra(save_individual_files=False, alpha=0.5):
    """
    Run multiple configurations with different parameters using Dijkstra algorithm
    """
    print(f"\n{'='*80}")
    print(f"RUNNING MULTIPLE CONFIGURATION TESTS - DIJKSTRA ALGORITHM")
    print(f"📊 Using fixed Alpha = {alpha} for all configurations")
    print(f"🔄 All discretionary nodes treated as mandatory power nodes")
    if save_individual_files:
        print(f"📁 Individual files will be saved for each configuration")
    else:
        print(f"📊 Only comprehensive logs will be saved (recommended for multiple tests)")
    print(f"{'='*80}")

    # Configuration sets to test
    configurations = [
        {
            'name': 'Dijkstra_Baseline',
            'graph_index': 3,  # CAMBIATO: da 0 a 3
            'alpha': alpha,
            'capacities': {1: 1, 2: 1, 3: 1, 4: 3, 5: 3, 6: 5, 7: 1},
        },
        {
            'name': 'Dijkstra_High_Capacity',
            'graph_index': 3,  # CAMBIATO: da 0 a 3
            'alpha': alpha,
            'capacities': {1: 2, 2: 2, 3: 2, 4: 5, 5: 5, 6: 8, 7: 2},
        },
        {
            'name': 'Dijkstra_Low_Capacity',
            'graph_index': 3,  # CAMBIATO: da 0 a 3
            'alpha': alpha,
            'capacities': {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 1},
        },
    ]

    results_summary = []

    for i, config in enumerate(configurations):
        print(f"\n{'-'*60}")
        print(f"RUNNING DIJKSTRA CONFIGURATION {i+1}/{len(configurations)}: {config['name']}")
        print(f"Alpha: {alpha} | Graph: {config['graph_index']} | Capacities: {config['capacities']}")
        print(f"{'-'*60}")

        best_solution, all_solutions, success = run_configuration_test_dijkstra(
            graph_index=config['graph_index'],
            alpha=config['alpha'],
            power_capacities_config=config['capacities'],
            config_name=config['name'],
            save_individual_files=save_individual_files
        )

        if success and best_solution:
            results_summary.append({
                'name': config['name'],
                'score': best_solution.score,
                'connected': len(best_solution.connected_weak),
                'failed': len(best_solution.failed_connections),
                'discretionary_used': best_solution.discretionary_used,
                'alpha': config['alpha'],
                'total_capacity': sum(config['capacities'].values()),
            })
            print(f"✅ SUCCESS: Score {best_solution.score:.2f}")
        else:
            results_summary.append({
                'name': config['name'],
                'score': float('inf'),
                'connected': 0,
                'failed': 999,
                'discretionary_used': [],
                'alpha': config['alpha'],
                'total_capacity': sum(config['capacities'].values()),
            })
            print(f"❌ FAILED")

    return results_summary

def run_configuration_test_dijkstra(graph_index, alpha, power_capacities_config,
                                   config_name="", save_individual_files=True):
    """
    Run a single configuration test using Dijkstra algorithm
    """
    global config_logger, power_capacities, discretionary_nodes_list

    # Set global variables for this configuration
    power_capacities = power_capacities_config.copy()

    try:
        # Load graph
        file_name = os.path.join(path, f"grafo_{graph_index}.pickle")  # CAMBIATO: da "grafo_" a "graph_"
        print(f"🔍 Trying to load: {file_name}")

        if not os.path.exists(file_name):
            print(f"❌ ERROR: File {file_name} does not exist!")
            return None, [], False

        with open(file_name, "rb") as f:
            graph = pickle.load(f)
        print(f"✅ Successfully loaded {file_name}")

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

        print(f"📊 Graph loaded successfully:")
        print(f"   - Weak nodes: {weak_nodes_list}")
        print(f"   - Mandatory nodes: {mandatory_nodes_list}")
        print(f"   - Discretionary nodes (treated as mandatory): {discretionary_nodes_list}")

        # Draw base graph (if saving individual files)
        if save_individual_files:
            print("🎨 Drawing base graph...")
            draw_graph_dijkstra(graph)

        # Run Dijkstra algorithm
        print("🚀 Starting Dijkstra algorithm...")

        best_solution, all_solutions = find_best_solution_dijkstra(
            graph, weak_nodes_list, mandatory_nodes_list, discretionary_nodes_list,
            power_capacities, alpha
        )

        # Save individual files if requested
        if save_individual_files:
            print("💾 Saving individual files...")
            if config_name:
                base_filename = f"DIJKSTRA_GRAPH_{graph_index}_{config_name.replace(' ', '_')}"
            else:
                base_filename = f"DIJKSTRA_GRAPH_{graph_index}_ALPHA_{alpha}"

            # Visualize and save results
            print("🎨 Creating solution visualization...")
            visualize_dijkstra_solution(graph, best_solution, weak_nodes_list, mandatory_nodes_list,
                                       discretionary_nodes_list, base_filename)

            print("📝 Saving solution summary...")
            save_dijkstra_summary(best_solution, all_solutions, f"{base_filename}_summary")

            print("🔧 Saving solution as pickle...")
            save_solution_as_pickle(best_solution, graph_index, alpha, f"{base_filename}_solution.pickle")

            print("🌳 Exporting tree to GraphML...")
            export_tree_to_graphml(best_solution, graph_index, alpha, f"{base_filename}_tree.graphml")

        print(f"✅ Configuration completed")
        return best_solution, all_solutions, True

    except Exception as e:
        print(f"❌ Configuration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, [], False

def draw_graph_dijkstra(G):
    """Draw base graph with Dijkstra labeling"""
    global plot_counter

    plt.figure(figsize=(12, 10))
    plt.clf()

    pos = nx.spring_layout(G, k=2, iterations=50)

    # Updated colors for Dijkstra: discretionary treated as mandatory
    node_colors = []
    for node, data in G.nodes(data=True):
        if data['node_type'] == 'weak':
            node_colors.append('green')
        elif data['node_type'] == 'power_mandatory':
            node_colors.append('red')
        elif data['node_type'] == 'power_discretionary':
            node_colors.append('orange')
        else:
            node_colors.append('gray')

    nx.draw(G, pos, with_labels=True, node_color=node_colors, font_weight='bold',
            node_size=1500, font_size=10, font_color="black")

    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9,
                                font_color='black', bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white", alpha=0.8, edgecolor='gray'))

    plt.title("Base Graph for Dijkstra Algorithm\n(Discretionary nodes treated as mandatory)")

    filename = f'dijkstra_base_graph_{plot_counter:03d}.png'
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    plot_counter += 1

    print(f"📊 Base graph saved: {filename}")
    print(f"📁 Full path: {filepath}")

    return filepath

# Global variables
plot_counter = 0
power_capacities = {}
main_graph = None
discretionary_nodes_list = []

if __name__ == "__main__":
    # Choose execution mode
    print("🚀 Dijkstra Algorithm Execution Options:")
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

    print(f"📊 Using Alpha = {alpha}")
    print(f"🔄 DIJKSTRA MODE: All discretionary nodes treated as mandatory power nodes")

    if execution_mode == "2":
        # Run multiple configurations without individual files
        print("🚀 Starting comprehensive Dijkstra configuration testing (logs only)...")
        results = run_multiple_configurations_dijkstra(save_individual_files=False, alpha=alpha)

    elif execution_mode == "3":
        # Run multiple configurations WITH individual files
        print("🚀 Starting comprehensive Dijkstra configuration testing with individual files...")
        print("⚠️  Warning: This will create many PNG and TXT files!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            results = run_multiple_configurations_dijkstra(save_individual_files=True, alpha=alpha)
        else:
            print("❌ Cancelled")

    else:
        # Single configuration mode (Dijkstra version)
        graph_index = 3  # CAMBIATO: da 0 a 3 per corrispondere al tuo file

        # Single configuration parameters
        config_params = {
            'config_name': f'Dijkstra_Single_Test_Alpha_{alpha}_Graph_{graph_index}',
            'graph_index': graph_index,
            'alpha': alpha,
            'power_capacities': {1: 1, 2: 1, 3: 1, 4: 3, 5: 3, 6: 5, 7: 1},
        }

        print(f"🚀 Running single Dijkstra configuration test...")
        print(f"📊 Graph index: {graph_index}")

        try:
            # Load graph
            file_name = os.path.join(path, f"grafo_{graph_index}.pickle")  # CAMBIATO: da "grafo_" a "graph_"
            print(f"🔍 Looking for graph file: {file_name}")

            if not os.path.exists(file_name):
                print(f"❌ ERROR: File {file_name} does not exist!")
                print(f"📁 Available files in {path}:")
                try:
                    files = [f for f in os.listdir(path) if f.endswith('.pickle')]
                    for f in files:
                        print(f"   - {f}")
                    # Suggerisci automaticamente l'indice corretto
                    if 'graph_3.pickle' in files:
                        print(f"💡 Found graph_3.pickle - this should work!")
                    elif any(f.startswith('graph_') for f in files):
                        available = [f for f in files if f.startswith('graph_')]
                        print(f"💡 Available graph files: {available}")
                except:
                    print("   - Cannot read directory")
                exit(1)

            with open(file_name, "rb") as f:
                graph = pickle.load(f)
            print(f"✅ Loaded {file_name}")

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

            print("🎨 Drawing base graph...")
            draw_graph_dijkstra(graph)

            print(f"📊 Graph structure:")
            print(f"   Weak nodes: {weak_nodes_list}")
            print(f"   Mandatory nodes: {mandatory_nodes_list}")
            print(f"   Discretionary nodes (treated as mandatory): {discretionary_nodes_list}")

            # Node capacities
            power_capacities = {1: 1, 2: 1, 3: 1, 4: 3, 5: 3, 6: 5, 7: 1}

            print(f"⚡ Node capacities: {power_capacities}")

            # Find best solution with Dijkstra algorithm
            print("🚀 Running Dijkstra algorithm...")
            best_solution, all_solutions = find_best_solution_dijkstra(
                graph, weak_nodes_list, mandatory_nodes_list, discretionary_nodes_list,
                power_capacities, alpha
            )

            # Visualize and save results
            print("🎨 Creating final visualization...")
            visualize_dijkstra_solution(graph, best_solution, weak_nodes_list, mandatory_nodes_list,
                                       discretionary_nodes_list, f"DIJKSTRA_GRAPH_{graph_index}")

            print("📝 Saving final summary...")
            save_dijkstra_summary(best_solution, all_solutions, f"DIJKSTRA_GRAPH_{graph_index}_summary")

            print("🔧 Saving solution as pickle for reuse...")
            pickle_file = save_solution_as_pickle(best_solution, graph_index, alpha)

            print("🌳 Exporting tree to GraphML format...")
            graphml_file = export_tree_to_graphml(best_solution, graph_index, alpha)

            print(f"\n🏆 GRAPH {graph_index} COMPLETED (DIJKSTRA ALGORITHM)")
            print(f"DIJKSTRA SOLUTION:")
            print(f"  - Score: {best_solution.score:.2f}")
            print(f"  - ACC: {best_solution.acc_cost:.6f}")
            print(f"  - AOC: {best_solution.aoc_cost:.6f}")
            print(f"  - Connected: {len(best_solution.connected_weak)}/{len(weak_nodes_list)}")
            print(f"  - Failed connections: {len(best_solution.failed_connections)}")
            print(f"  - Discretionary used: {best_solution.discretionary_used}")

            print(f"\n📁 Output files saved to: {path}")
            print(f"  - Base graph: dijkstra_base_graph_XXX.png")
            print(f"  - Solution visualization: DIJKSTRA_GRAPH_{graph_index}_XXX.png")
            print(f"  - Solution summary: DIJKSTRA_GRAPH_{graph_index}_summary.txt")
            if pickle_file:
                print(f"  - Solution pickle: {os.path.basename(pickle_file)}")
            if graphml_file:
                print(f"  - Tree GraphML: {os.path.basename(graphml_file)}")

            print(f"\n💡 How to reuse the saved solution:")
            print(f"```python")
            print(f"import pickle")
            print(f"import networkx as nx")
            print(f"")
            print(f"# Load the complete solution")
            if pickle_file:
                print(f"with open('{pickle_file}', 'rb') as f:")
                print(f"    solution_data = pickle.load(f)")
                print(f"")
                print(f"# Access the solution tree")
                print(f"tree = solution_data['solution_tree']")
                print(f"score = solution_data['score']")
                print(f"edges = solution_data['solution_edges']")
                print(f"capacity_usage = solution_data['capacity_usage']")
                print(f"")
                print(f"# Or use the helper function:")
                print(f"# solution_data = load_solution_from_pickle('{pickle_file}')")
            print(f"```")

        except FileNotFoundError:
            print(f"❌ Error: File {file_name} not found!")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
