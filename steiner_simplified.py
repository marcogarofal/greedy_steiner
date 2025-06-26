import networkx as nx
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import pickle
import os

# Add matplotlib backend to avoid display errors
import matplotlib
matplotlib.use('Agg')

# Path for plots
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'graphs/')

class Node:
    def __init__(self, name, node_type, capacity=0, weight=0):
        self.name = name
        self.node_type = node_type
        self.capacity = capacity
        self.weight = weight

class Solution:
    def __init__(self, steiner_tree, capacity_usage, connected_weak, failed_connections,
                 total_cost, capacity_cost, discretionary_used, graph_info=""):
        self.steiner_tree = steiner_tree
        self.capacity_usage = capacity_usage
        self.connected_weak = connected_weak
        self.failed_connections = failed_connections
        self.total_cost = total_cost
        self.capacity_cost = capacity_cost
        self.discretionary_used = discretionary_used
        self.graph_info = graph_info

        # Calculate overall score (lower is better)
        self.score = self.calculate_score()

    def calculate_score(self):
        """
        Calculate a score to compare solutions
        Factors considered:
        1. Connected nodes (maximum weight - we want to connect all)
        2. Total edge cost
        3. Capacity violations
        4. Capacity usage efficiency
        """
        # Weight for unconnected nodes (very high penalty)
        connection_penalty = len(self.failed_connections) * 1000

        # Edge cost (medium weight)
        edge_cost = self.total_cost

        # Penalty for capacity violations
        violation_penalty = 0
        for node, usage in self.capacity_usage.items():
            max_cap = power_capacities.get(node, float('inf'))
            if usage > max_cap and max_cap != float('inf'):
                violation_penalty += (usage - max_cap) * 100

        # Capacity inefficiency cost (low weight)
        efficiency_cost = self.capacity_cost * 10

        total_score = connection_penalty + edge_cost + violation_penalty + efficiency_cost

        # DEBUG: Print score calculation details
        print(f"    🔍 DEBUG SCORE for {self.graph_info}:")
        print(f"        - Failed nodes: {len(self.failed_connections)} → Connection penalty: {connection_penalty}")
        print(f"        - Edge cost: {edge_cost}")
        print(f"        - Capacity violations: {violation_penalty}")
        print(f"        - Efficiency capacity: {self.capacity_cost:.3f} × 10 = {efficiency_cost:.2f}")
        print(f"        - TOTAL SCORE: {connection_penalty} + {edge_cost} + {violation_penalty} + {efficiency_cost:.2f} = {total_score:.2f}")

        # Check detailed violations
        if violation_penalty > 0:
            print(f"        - VIOLATION DETAILS:")
            for node, usage in self.capacity_usage.items():
                max_cap = power_capacities.get(node, float('inf'))
                if usage > max_cap and max_cap != float('inf'):
                    print(f"          Node {node}: {usage}/{max_cap} (violation: {usage - max_cap})")

        return total_score

    def __str__(self):
        return (f"Solution {self.graph_info}:\n"
                f"  - Connected nodes: {len(self.connected_weak)} (failed: {len(self.failed_connections)})\n"
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

def check_path_capacity_feasible(path_info, capacity_usage, power_capacities):
    """
    Check if a path is feasible from capacity perspective
    """
    target_mandatory = path_info['target_mandatory']
    discretionary_used = path_info['discretionary_used']

    # Check capacity of target mandatory node
    if capacity_usage[target_mandatory] >= power_capacities.get(target_mandatory, float('inf')):
        return False

    # Check capacity of discretionary nodes in path
    for disc_node in discretionary_used:
        if capacity_usage[disc_node] >= power_capacities.get(disc_node, float('inf')):
            return False

    return True

def solve_with_discretionary_subset(graph, weak_nodes, mandatory_nodes, discretionary_subset, power_capacities, graph_info=""):
    """
    Solve the problem using only a specific subset of discretionary nodes
    """
    steiner_tree = nx.Graph()
    capacity_usage = {node: 0 for node in mandatory_nodes + discretionary_subset}
    connected_weak = set()
    failed_connections = []
    actually_used_discretionary = set()  # Track actually used discretionary nodes

    # Find all possible paths for each weak node
    all_weak_options = {}
    for weak_node in weak_nodes:
        paths = find_all_paths_to_mandatory(graph, weak_node, mandatory_nodes, discretionary_subset)
        all_weak_options[weak_node] = paths

    # Greedy algorithm - connect with cheapest paths
    all_options = []
    for weak_node, paths in all_weak_options.items():
        for path_info in paths:
            all_options.append({
                'weak_node': weak_node,
                **path_info
            })

    all_options.sort(key=lambda x: x['cost'])

    selected_connections = []  # Track selected connections

    # Try to connect using cheapest paths
    for option in all_options:
        weak_node = option['weak_node']

        if weak_node in connected_weak:
            continue

        if check_path_capacity_feasible(option, capacity_usage, power_capacities):
            path = option['path']
            target_mandatory = option['target_mandatory']
            discretionary_used = option['discretionary_used']

            capacity_usage[target_mandatory] += 1
            for disc_node in discretionary_used:
                capacity_usage[disc_node] += 1
                actually_used_discretionary.add(disc_node)  # Register actual usage

            for i in range(len(path) - 1):
                steiner_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

            connected_weak.add(weak_node)
            selected_connections.append(option)

            print(f"    ✓ Connected {weak_node} via {path} (cost: {option['cost']})")

    # Force connection of remaining nodes
    remaining_weak = set(weak_nodes) - connected_weak

    if remaining_weak:
        print(f"    ⚠️  Forcing connection for: {remaining_weak}")

    for weak_node in remaining_weak:
        paths = all_weak_options[weak_node]

        connected = False
        for path_info in paths:
            target_mandatory = path_info['target_mandatory']
            discretionary_used = path_info['discretionary_used']
            path = path_info['path']

            capacity_usage[target_mandatory] += 1
            for disc_node in discretionary_used:
                capacity_usage[disc_node] += 1
                actually_used_discretionary.add(disc_node)  # Register actual usage

            for i in range(len(path) - 1):
                steiner_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

            connected_weak.add(weak_node)
            selected_connections.append(path_info)
            connected = True
            print(f"    ✓ Forced {weak_node} via {path} (cost: {path_info['cost']})")
            break

        if not connected:
            failed_connections.append(weak_node)
            print(f"    ✗ IMPOSSIBLE to connect {weak_node}")

    # Calculate statistics
    total_cost = sum(graph[u][v]['weight'] for u, v in steiner_tree.edges())

    # Calculate capacity cost - ONLY FOR ACTUALLY USED NODES
    capacity_cost = 0
    nodes_actually_used = [n for n in capacity_usage if capacity_usage[n] > 0 and power_capacities.get(n, 0) > 0]

    if nodes_actually_used:
        capacity_ratios = []
        for node in nodes_actually_used:
            if power_capacities[node] > 0:
                ratio = capacity_usage[node] / power_capacities[node]
                capacity_ratios.append(ratio)
                print(f"    📊 Node {node} (USED): {capacity_usage[node]}/{power_capacities[node]} = {ratio:.3f}")

        capacity_cost = sum(capacity_ratios) / len(capacity_ratios)
        print(f"    📊 Capacity cost (used nodes only): {sum(capacity_ratios):.3f} / {len(capacity_ratios)} = {capacity_cost:.3f}")

        # Show unused nodes for clarity
        unused_nodes = [n for n in capacity_usage if capacity_usage[n] == 0]
        if unused_nodes:
            print(f"    📊 Unused nodes (ignored in calculation): {unused_nodes}")
    else:
        print(f"    📊 No power nodes used!")

    # USE ACTUALLY USED DISCRETIONARY NODES, NOT AVAILABLE ONES
    actually_used_list = sorted(list(actually_used_discretionary))

    print(f"    📊 Available: {discretionary_subset} → Actually used: {actually_used_list}")
    print(f"    📊 Final capacity usage: {dict(capacity_usage)}")

    return Solution(steiner_tree, capacity_usage, connected_weak, failed_connections,
                   total_cost, capacity_cost, actually_used_list, graph_info)

def find_best_solution_simplified(graph, weak_nodes, mandatory_nodes, all_discretionary_nodes, power_capacities):
    """
    Find the best solution by testing ONLY two cases:
    1. Without discretionary nodes
    2. With ALL discretionary nodes
    """
    print(f"\n{'='*60}")
    print(f"SIMPLIFIED APPROACH - TESTING ONLY 2 CASES")
    print(f"{'='*60}")

    all_solutions = []

    # 1. Solution without discretionary nodes
    print("\n--- Testing solution WITHOUT discretionary nodes ---")
    solution_no_disc = solve_with_discretionary_subset(
        graph, weak_nodes, mandatory_nodes, [], power_capacities.copy(),
        "WITHOUT discretionary"
    )
    all_solutions.append(solution_no_disc)
    print(solution_no_disc)

    # 2. Solution with ALL discretionary nodes
    print(f"\n--- Testing solution WITH ALL discretionary nodes ---")
    print(f"All discretionary nodes: {all_discretionary_nodes}")

    solution_all_disc = solve_with_discretionary_subset(
        graph, weak_nodes, mandatory_nodes, all_discretionary_nodes, power_capacities.copy(),
        f"WITH ALL discretionary {all_discretionary_nodes}"
    )
    all_solutions.append(solution_all_disc)
    print(f"  Score: {solution_all_disc.score:.2f}, Connected: {len(solution_all_disc.connected_weak)}/{len(weak_nodes)}, Cost: {solution_all_disc.total_cost}")

    # 3. Find best solution between the two
    best_solution = min(all_solutions, key=lambda s: s.score)

    print(f"\n{'='*60}")
    print(f"COMPARING THE 2 SOLUTIONS")
    print(f"{'='*60}")

    # Sort and show both solutions
    all_solutions.sort(key=lambda s: s.score)

    for i, solution in enumerate(all_solutions):
        marker = "🏆 BEST" if solution == best_solution else f"   #{i+1}"
        print(f"{marker} - Score: {solution.score:.2f}")
        print(f"        Discretionary: {solution.discretionary_used}")
        print(f"        Connected: {len(solution.connected_weak)}/{len(weak_nodes)} (failed: {len(solution.failed_connections)})")
        print(f"        Edge cost: {solution.total_cost}, Capacity cost: {solution.capacity_cost:.3f}")

    # Show what discretionary nodes were actually used in the "all discretionary" case
    if len(all_solutions) >= 2:
        all_disc_solution = [s for s in all_solutions if s.graph_info.startswith("WITH ALL")][0]
        print(f"\n💡 INSIGHT: When given ALL discretionary nodes {all_discretionary_nodes},")
        print(f"   the algorithm actually used only: {all_disc_solution.discretionary_used}")
        print(f"   Unused discretionary nodes: {set(all_discretionary_nodes) - set(all_disc_solution.discretionary_used)}")

    return best_solution, all_solutions

def visualize_best_solution(graph, best_solution, weak_nodes, mandatory_nodes, all_discretionary_nodes, save_name="BEST_SOLUTION"):
    """
    Visualize the best solution found
    """
    global plot_counter
    plt.figure(figsize=(18, 14))

    pos = nx.spring_layout(graph, weight='weight', k=3, iterations=100)

    steiner_tree = best_solution.steiner_tree
    capacity_usage = best_solution.capacity_usage
    connected_weak = best_solution.connected_weak
    failed_connections = best_solution.failed_connections
    discretionary_used = best_solution.discretionary_used

    # Node colors
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
            node_colors.append('lightgray')  # Unused discretionary
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

    # Handle weight labels
    solution_edges = set(steiner_tree.edges())

    # Weights for non-solution edges
    non_solution_edge_labels = {}
    for (u, v) in graph.edges():
        if (u, v) not in solution_edges and (v, u) not in solution_edges:
            non_solution_edge_labels[(u, v)] = graph[u][v]['weight']

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=non_solution_edge_labels, font_size=9,
                                font_color='gray', bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white", alpha=0.8, edgecolor='none'))

    # Weights for solution edges (shifted)
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

    # Detailed title
    title = (f"🏆 BEST SOLUTION (SIMPLIFIED APPROACH)\n"
             f"Score: {best_solution.score:.2f} | Connected: {len(connected_weak)}/{len(weak_nodes)} | "
             f"Cost: {best_solution.total_cost} | Discretionary: {discretionary_used}")

    plt.title(title, fontsize=14, weight='bold', pad=20)

    # Improved legend
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
    print(f"🏆 BEST SOLUTION saved as {save_name}_{plot_counter-1:03d}_SCORE_{best_solution.score:.0f}.png")

def save_solution_summary(best_solution, all_solutions, save_name="solution_summary"):
    """
    Save a text summary of solutions
    """
    with open(f'{path}{save_name}.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SIMPLIFIED SOLUTION SUMMARY (2 cases only)\n")
        f.write("="*80 + "\n\n")

        f.write("🏆 BEST SOLUTION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Final score: {best_solution.score:.2f}\n")
        f.write(f"Discretionary ACTUALLY used: {best_solution.discretionary_used}\n")
        f.write(f"Connected weak nodes: {len(best_solution.connected_weak)}\n")
        f.write(f"Failed connections: {len(best_solution.failed_connections)}\n")
        f.write(f"Total edge cost: {best_solution.total_cost}\n")
        f.write(f"Capacity efficiency cost: {best_solution.capacity_cost:.3f}\n")
        f.write(f"Solution edges: {list(best_solution.steiner_tree.edges())}\n")
        f.write(f"Capacity usage: {dict(best_solution.capacity_usage)}\n\n")

        f.write("COMPARISON OF 2 SOLUTIONS:\n")
        f.write("-"*40 + "\n")

        for i, solution in enumerate(all_solutions):
            f.write(f"#{i+1} - Score: {solution.score:7.2f} | ")
            f.write(f"Discretionary USED: {str(solution.discretionary_used):15s} | ")
            f.write(f"Connected: {len(solution.connected_weak):2d} | ")
            f.write(f"Cost: {solution.total_cost:5.0f} | ")
            f.write(f"Info: {solution.graph_info}\n")

    print(f"📋 Summary saved in {save_name}.txt")

plot_counter = 0
power_capacities = {}  # Global variable for score calculation

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

if __name__ == "__main__":
    # Load only the first and last graphs (without and with all discretionary)
    loaded_graphs = []
    graph_indices = [0, 3]  # First (no discretionary) and last (all discretionary)

    for i in graph_indices:
        file_name = os.path.join(path, f"grafo_{i}.pickle")
        with open(file_name, "rb") as f:
            loaded_graphs.append(pickle.load(f))
        print(f"Loaded grafo_{i}.pickle")

    for graph_idx, graph in enumerate(loaded_graphs):
        actual_graph_index = graph_indices[graph_idx]
        print(f"\n{'='*80}")
        print(f"PROCESSING GRAPH {actual_graph_index} ({graph_idx + 1}/2) - SIMPLIFIED APPROACH")
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

        print(f"Weak nodes: {weak_nodes_list}")
        print(f"Mandatory nodes: {mandatory_nodes_list}")
        print(f"Discretionary nodes: {discretionary_nodes_list}")

        # Node capacities (global variable for score calculation)
        power_capacities = {1: 1, 2: 1, 3: 1, 4: 3, 5: 3, 6: 5, 7: 0}

        # FIND BEST SOLUTION WITH SIMPLIFIED APPROACH (only 2 cases)
        best_solution, all_solutions = find_best_solution_simplified(
            graph, weak_nodes_list, mandatory_nodes_list, discretionary_nodes_list, power_capacities
        )

        # VISUALIZE BEST SOLUTION
        visualize_best_solution(graph, best_solution, weak_nodes_list, mandatory_nodes_list,
                               discretionary_nodes_list, f"GRAPH_{actual_graph_index}_SIMPLIFIED")

        # SAVE SUMMARY
        save_solution_summary(best_solution, all_solutions, f"GRAPH_{actual_graph_index}_simplified_summary")

        print(f"\n🏆 GRAPH {actual_graph_index} COMPLETED (SIMPLIFIED)")
        print(f"BEST SOLUTION:")
        print(f"  - Score: {best_solution.score:.2f}")
        print(f"  - Discretionary used: {best_solution.discretionary_used}")
        print(f"  - Connected: {len(best_solution.connected_weak)}/{len(weak_nodes_list)}")
        print(f"  - Cost: {best_solution.total_cost}")

        input("Press ENTER to continue to next graph...")
