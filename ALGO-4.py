import random
import matplotlib.pyplot as plt
import networkx as nx
import logging
import time
from collections import defaultdict
import copy


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("abc_graph_coloring.log", mode='w'),
        logging.StreamHandler()
    ]
)

DEFAULT_NUM_VERTICES = 300
DEFAULT_MIN_DEGREE = 1
DEFAULT_MAX_DEGREE = 30
DEFAULT_EXTRA_EDGES = 500
DEFAULT_TOTAL_BEES = 60
DEFAULT_NUM_SCOUTS = 5
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_ONLOOKER_LIMIT_PER_VERTEX = 50
DEFAULT_MAX_K = DEFAULT_MAX_DEGREE + 1

# --- Graph Generation Function ---
def generate_graph_original(n_vertices, min_degree, max_degree, extra_edges):
    """Generates a random connected graph using the original approach."""
    logging.info("Generating graph (Original Method): Vertices=%d, MinDegree=%d, MaxDegree=%d, ExtraEdges=%d",
                 n_vertices, min_degree, max_degree, extra_edges)
    if n_vertices <= 0: logging.error("Number of vertices must be positive."); return {}, {}
    if max_degree < min_degree: logging.error("max_degree cannot be less than min_degree."); return {}, {}
    if min_degree <= 0: logging.warning("min_degree <= 0, setting to 1."); min_degree = 1

    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    if n_vertices > 1:
        for i in range(n_vertices - 1): G.add_edge(i, i + 1)
    elif n_vertices == 1:
         degrees = {0: 0}; adjacency_list = {0: []}
         if min_degree > 0: degrees[0] = min_degree
         return adjacency_list, degrees

    degrees = dict(G.degree())

    nodes_below_min = [node for node, deg in degrees.items() if deg < min_degree]
    random.shuffle(nodes_below_min)
    for node in nodes_below_min:
        attempts = 0; max_connection_attempts = max(n_vertices * 2, min_degree * 5)
        while degrees.get(node, 0) < min_degree and attempts < max_connection_attempts:
            attempts += 1
            possible_neighbors = [n for n in range(n_vertices) if n != node]
            if not possible_neighbors: continue
            potential_neighbor = random.choice(possible_neighbors)
            if not G.has_edge(node, potential_neighbor) and \
               degrees.get(node, 0) < max_degree and \
               degrees.get(potential_neighbor, 0) < max_degree:
                   G.add_edge(node, potential_neighbor)
                   degrees[node] = degrees.get(node, 0) + 1
                   degrees[potential_neighbor] = degrees.get(potential_neighbor, 0) + 1

    edges_added_successfully = 0; attempts = 0
    max_total_attempts = max(extra_edges * 10, n_vertices * 5)
    while edges_added_successfully < extra_edges and attempts < max_total_attempts:
        attempts += 1
        if n_vertices < 2: break
        u, v = random.sample(range(n_vertices), 2)
        if u != v and not G.has_edge(u, v):
            if degrees.get(u, 0) < max_degree and degrees.get(v, 0) < max_degree:
                G.add_edge(u, v)
                degrees[u] = degrees.get(u, 0) + 1
                degrees[v] = degrees.get(v, 0) + 1
                edges_added_successfully += 1

    if edges_added_successfully < extra_edges: logging.warning(f"Added only {edges_added_successfully}/{extra_edges} extra edges.")
    else: logging.info(f"Added {edges_added_successfully} extra edges.")

    final_degrees = dict(G.degree())
    actual_min = min(final_degrees.values()) if final_degrees else 0
    actual_max = max(final_degrees.values()) if final_degrees else 0
    logging.info(f"Final graph: Min Degree = {actual_min}, Max Degree = {actual_max}")
    adjusted_degrees_for_n_state = final_degrees.copy()
    nodes_corrected_for_n = 0
    for node, deg in adjusted_degrees_for_n_state.items():
        if deg < min_degree:
            adjusted_degrees_for_n_state[node] = min_degree
            nodes_corrected_for_n += 1
    if nodes_corrected_for_n > 0: logging.warning(f"Adjusted n_state for {nodes_corrected_for_n} nodes to meet min_degree={min_degree}.")

    adjacency_list = {node: sorted(list(G.neighbors(node))) for node in G.nodes()}
    logging.info("Graph generation complete. Nodes: %d, Edges: %d", G.number_of_nodes(), G.number_of_edges())
    return adjacency_list, adjusted_degrees_for_n_state

# --- Conflict Counting Function ---
def count_conflicts(vertex_colors, adjacency_list):
    """Counts the number of conflicting edges in the current coloring. Needed for checking validity."""
    conflict_count = 0; checked_edges = set()
    nodes = list(vertex_colors.keys())
    for u in nodes:
        color_u = vertex_colors.get(u, -1)
        if color_u == -1: continue
        if u in adjacency_list:
            for v in adjacency_list[u]:
                 if v in vertex_colors:
                    color_v = vertex_colors.get(v, -1)
                    if color_v == -1: continue
                    edge = tuple(sorted((u, v)))
                    if edge not in checked_edges:
                        if color_u == color_v: conflict_count += 1
                        checked_edges.add(edge)
    return conflict_count

# --- Main ABC Algorithm Function ---
def abc_graph_coloring(adjacency_list, initial_n_state, total_bees, num_scouts, max_iterations, onlooker_limit, max_k):
    """
    Implements the specific ABC variant based on lecture interpretation.
    Includes scout logic: target node with max used color, uncolor, and reset state.
    Tracks and returns the best valid coloring found.
    Goal: Minimize chromatic number of the first complete coloring.
    Termination: When n=0 for all vertices or max_iterations reached.
    Primary reporting metric: Number of used colors.
    """
    num_vertices = len(adjacency_list)
    if num_vertices == 0: return {}, [], 0, 0

    logging.info("Starting ABC Algorithm Variant for Graph Coloring (Tracking Best Solution)")
    logging.info(f"Parameters: Vertices={num_vertices}, Total Bees={total_bees}, Scouts={num_scouts}")
    logging.info(f"Max Iterations={max_iterations}, Onlooker Limit={onlooker_limit}, Max Colors (K)={max_k}")
    logging.info(f"Termination condition: n=0 for all vertices or max_iterations reached.")
    logging.info(f"Reporting metric (Objective Function): Number of used colors.")

    vertex_colors = {node: -1 for node in adjacency_list.keys()}
    vertex_n_state = initial_n_state.copy()
    vertex_process_counter = {node: 0 for node in adjacency_list.keys()}
    all_colors = list(range(max_k))

    best_coloring_so_far = None
    min_colors_found_so_far = float('inf')

    if num_scouts >= total_bees:
        logging.error("Scouts >= Total Bees."); num_scouts = max(0, total_bees - 1)
        logging.warning(f"Adjusted scouts to {num_scouts}")
    num_onlookers = total_bees - num_scouts

    iteration = 0
    report_data = []
    first_pass_complete = False
    first_pass_chromatic_number = float('inf')
    start_time = time.time()

    while iteration < max_iterations:
        iteration += 1

        active_vertices_dict = {v: n for v, n in vertex_n_state.items() if n > 0}
        if not active_vertices_dict:
            logging.info(f"Termination condition met: n=0 for all vertices at iteration {iteration}.")
            break

        total_n_nectar = sum(active_vertices_dict.values())
        if total_n_nectar <= 0:
             logging.warning(f"Iteration {iteration}: No active vertices with n>0 nectar. Breaking.")
             break

        # --- Onlooker Phase Simulation ---
        onlookers_processed_vertices = []
        active_nodes_list = list(active_vertices_dict.keys())
        if active_nodes_list:
            nectar_values = [active_vertices_dict[node] for node in active_nodes_list]
            if len(active_nodes_list) == 1: probs = [1.0]
            elif total_n_nectar > 0: probs = [n / total_n_nectar for n in nectar_values]
            else: probs = [1.0 / len(active_nodes_list)] * len(active_nodes_list)

            for _ in range(num_onlookers):
                try:
                    if len(probs) != len(active_nodes_list): raise ValueError("Prob length mismatch")
                    chosen_vertex = random.choices(active_nodes_list, weights=probs, k=1)[0]
                except ValueError as e:
                    logging.error(f"Choice error (iter {iteration}): {e}. Using random.")
                    if not active_nodes_list: continue
                    chosen_vertex = random.choice(active_nodes_list)

                onlookers_processed_vertices.append(chosen_vertex)

                # Coloring Action Logic (Greedy)
                current_used_colors_set_local = {c for c in vertex_colors.values() if c != -1}
                neighbors = adjacency_list.get(chosen_vertex, [])
                neighbor_colors = {vertex_colors.get(nbr, -1) for nbr in neighbors if vertex_colors.get(nbr, -1) != -1}
                assigned_color = -1; found_color = False
                sorted_used_colors = sorted(list(current_used_colors_set_local))
                for color in sorted_used_colors:
                    if color not in neighbor_colors:
                        vertex_colors[chosen_vertex] = color; found_color = True; break
                if not found_color:
                    for color in all_colors:
                        if color not in neighbor_colors:
                            vertex_colors[chosen_vertex] = color
                            found_color = True; break
                if not found_color:
                    logging.error(f"Could not find valid color for vertex {chosen_vertex} ({max_k} colors). Assigning 0.")
                    vertex_colors[chosen_vertex] = 0
        else:
            logging.warning(f"Iteration {iteration}: No active vertices for onlookers.")

        # --- Update 'n' state and Perform Targeted Scout Action ---
        processed_counts = defaultdict(int)
        for v in onlookers_processed_vertices: processed_counts[v] += 1
        current_used_colors = {c for c in vertex_colors.values() if c != -1}
        for v, count in processed_counts.items():
            if v in vertex_n_state and vertex_n_state[v] > 0:
                 vertex_process_counter[v] = vertex_process_counter.get(v, 0) + count
                 if vertex_process_counter[v] >= onlooker_limit:
                      if vertex_n_state[v] > 0: logging.debug(f"Vertex {v} reached limit {onlooker_limit}. Setting n=0. Bee becomes scout.")
                      vertex_n_state[v] = 0
                      # --- TARGETED SCOUT ACTION ---
                      current_active_nodes_for_scout = [node for node, n_val in vertex_n_state.items() if n_val > 0]
                      target_w = None; scout_action_type = "None"
                      if current_active_nodes_for_scout:
                          if current_used_colors:
                              max_used_color_index = max(current_used_colors)
                              target_candidates = [node for node in current_active_nodes_for_scout if vertex_colors.get(node) == max_used_color_index]
                              if target_candidates: target_w = random.choice(target_candidates); scout_action_type = f"Targeted Max Color ({max_used_color_index})"
                              else: target_w = random.choice(current_active_nodes_for_scout); scout_action_type = "Random Fallback 1"
                          else: target_w = random.choice(current_active_nodes_for_scout); scout_action_type = "Random Fallback 2"
                          if target_w is not None:
                              logging.debug(f"Scout from {v} action ({scout_action_type}): Uncoloring node {target_w} and resetting state.")
                              vertex_colors[target_w] = -1; vertex_n_state[target_w] = initial_n_state.get(target_w, 1); vertex_process_counter[target_w] = 0
                          else: logging.debug(f"Scout from {v}: Could not determine target_w.")
                      else: logging.debug(f"Scout from {v}: No active vertices (n>0) left to target.")
                      # --- End of TARGETED SCOUT ACTION ---

        # --- Check if initial coloring is complete ---
        if not first_pass_complete:
             all_nodes_in_adj = set(adjacency_list.keys())
             if all(vertex_colors.get(node, -1) != -1 for node in all_nodes_in_adj):
                 first_pass_complete = True
                 first_pass_used_colors = {c for c in vertex_colors.values() if c != -1}
                 first_pass_chromatic_number = len(first_pass_used_colors)
                 logging.info(f"--- Initial coloring completed at iteration {iteration} ---")
                 logging.info(f"Chromatic number after first pass: {first_pass_chromatic_number}")
                 # REMOVED Conflict Logging:
                 # conflicts_after_first_pass = count_conflicts(vertex_colors, adjacency_list)
                 # logging.info(f"Conflicts after first pass: {conflicts_after_first_pass}")

        # --- Update Best Solution Found So Far ---
        current_conflicts = count_conflicts(vertex_colors, adjacency_list) # Still needed for check
        current_num_colors = len({c for c in vertex_colors.values() if c != -1})
        is_complete_coloring = all(vertex_colors.get(node, -1) != -1 for node in adjacency_list.keys())

        if is_complete_coloring and current_conflicts == 0 and current_num_colors < min_colors_found_so_far:
            min_colors_found_so_far = current_num_colors
            best_coloring_so_far = copy.deepcopy(vertex_colors)
            logging.info(f"*** New best solution found at iteration {iteration}: {min_colors_found_so_far} colors. ***")


        # --- Reporting ---
        if iteration % 20 == 0 or iteration == 1:
            active_n_count_report = sum(1 for n in vertex_n_state.values() if n > 0)
            conflicts_for_data = current_conflicts
            report_data.append((iteration, current_num_colors, conflicts_for_data, active_n_count_report))
            logging.info(f"Iteration {iteration}: Used Colors={current_num_colors}, Vertices with n>0={active_n_count_report}")

        if iteration >= max_iterations:
            logging.warning(f"Reached max iterations ({max_iterations}) before all n became 0.")
            break

    # --- End of Loop ---
    end_time = time.time(); total_time = end_time - start_time
    logging.info(f"Algorithm finished in {total_time:.2f} seconds at iteration {iteration}.")


    final_conflicts_at_end = count_conflicts(vertex_colors, adjacency_list)
    final_used_colors_at_end = len({c for c in vertex_colors.values() if c != -1})
    logging.info(f"Final loop state: Used Colors={final_used_colors_at_end}")
    logging.info(f"Best valid solution found during run: {min_colors_found_so_far} colors.")

    if not first_pass_complete and iteration > 0 :
        current_used_colors_final = {c for c in vertex_colors.values() if c != -1}
        if current_used_colors_final:
             first_pass_chromatic_number = min_colors_found_so_far if min_colors_found_so_far != float('inf') else len(current_used_colors_final)
             logging.warning("Algorithm stopped before first pass complete; using best/final K.")
        else: first_pass_chromatic_number = 0
    if iteration == 0: first_pass_chromatic_number = 0

    if best_coloring_so_far is None:
        logging.warning("No conflict-free solution was found and saved. Using final state.")
        best_coloring_so_far = vertex_colors

    return best_coloring_so_far, report_data, first_pass_chromatic_number, min_colors_found_so_far


# --- Visualization ---
def visualize_graph_solution(adjacency_list, solution, max_colors_palette=DEFAULT_MAX_K):
    """Visualizes the graph with the coloring solution."""
    # (Implementation remains the same)
    if not solution or not adjacency_list: logging.error("Cannot visualize empty graph or solution."); return
    logging.info("Visualizing the graph coloring solution.")
    G = nx.Graph(); nodes = list(adjacency_list.keys())
    if not nodes: logging.warning("Graph has no nodes."); return
    G.add_nodes_from(nodes)
    for v, neighbors in adjacency_list.items():
        for nbr in neighbors:
             if v in G and nbr in G and v < nbr: G.add_edge(v, nbr)
    node_colors_map = {}; valid_colors = {c for c in solution.values() if c != -1}
    color_indices = {color: idx for idx, color in enumerate(sorted(list(valid_colors)))}
    num_distinct_colors = len(color_indices); uncolored_nodes_exist = False
    for node in G.nodes():
        color = solution.get(node, -1)
        if color == -1: node_colors_map[node] = 'grey'; uncolored_nodes_exist = True
        else:
            cmap = plt.get_cmap('rainbow', max(2, num_distinct_colors))
            norm_index = color_indices[color] / (num_distinct_colors - 1) if num_distinct_colors > 1 else 0.5
            node_colors_map[node] = cmap(norm_index)
    plot_colors = [node_colors_map.get(node, 'grey') for node in G.nodes()]
    if uncolored_nodes_exist: logging.warning("Graph contains uncolored nodes (grey).")
    num_final_conflicts = count_conflicts(solution, adjacency_list)
    num_colors_used_final = len(valid_colors)
    final_conflicts_list = []; final_non_conflicting_edges = []
    edges = list(G.edges())
    for u, v in edges:
        color_u = solution.get(u, -1); color_v = solution.get(v, -1)
        if color_u != -1 and color_v != -1 and color_u == color_v: final_conflicts_list.append((u, v))
        else: final_non_conflicting_edges.append((u,v))

    logging.info("Calculating layout..."); pos = None
    try:
        if len(nodes) > 1000: pos = nx.spring_layout(G, seed=42, iterations=25, k=0.1)
        else: pos = nx.spring_layout(G, seed=42, k=0.1)
        logging.info("Layout complete.")
    except ImportError: logging.warning("Scipy not found, random layout."); pos = nx.random_layout(G, seed=42)
    except Exception as e: logging.error(f"Layout failed: {e}. Using random."); pos = nx.random_layout(G, seed=42)
    plt.figure(figsize=(15, 10)); node_size = max(1.0, 5000 / max(1, len(nodes)))
    nx.draw_networkx_nodes(G, pos, node_color=plot_colors, node_size=node_size, edgecolors='grey', linewidths=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=final_non_conflicting_edges, edge_color='grey', width=0.5, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=final_conflicts_list, edge_color='red', width=1.0, alpha=0.8)
    plt.title(f"ABC Variant - Best Found Coloring ({num_final_conflicts} Conflicts, {num_colors_used_final} Colors Used)")
    plt.axis('off'); plt.show(); logging.info("Graph visualization displayed.")

# --- Plotting Performance ---
def plot_performance(report_data):
    """Plots the evolution of used colors (Primary Objective Function)."""
    if not report_data: logging.warning("No plot data."); return
    iterations, used_colors, _, _ = zip(*report_data)
    fig, ax1 = plt.subplots(figsize=(10, 6)); color = 'tab:blue'
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Number of Used Colors (Objective Function)", color=color)
    ax1.plot(iterations, used_colors, marker='o', linestyle='-', color=color, label='Used Colors')
    ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, linestyle='--', alpha=0.7)
    plt.title("ABC Variant: Objective Function (Used Colors) Over Iterations")
    plt.legend(loc='upper right'); plt.tight_layout(); plt.show()
    logging.info("Performance plot displayed.")

# --- User Input Functions ---
def get_positive_integer_input(prompt, default_value):
    """Gets positive integer input from the user with validation."""
    while True:
        try:
            user_input = input(f"{prompt} (default: {default_value}): ")
            if not user_input: return int(default_value)
            value = int(user_input)
            if value <= 0: print("Please enter a positive integer.")
            else: return value
        except ValueError: print("Invalid input. Please enter an integer.")
        except TypeError: print(f"Error with default type. Using {int(default_value)}"); return int(default_value)

# --- Main Execution ---
if __name__ == "__main__":
    random.seed(42)
    print("--- Algorithm Configuration ---")
    num_vertices = get_positive_integer_input("Enter number of vertices", DEFAULT_NUM_VERTICES)
    total_bees = get_positive_integer_input("Enter total number of bees", DEFAULT_TOTAL_BEES)
    num_scouts = get_positive_integer_input("Enter number of scout bees", DEFAULT_NUM_SCOUTS)
    max_iterations = get_positive_integer_input("Enter max number of iterations", DEFAULT_MAX_ITERATIONS)

    min_degree = DEFAULT_MIN_DEGREE; max_degree = DEFAULT_MAX_DEGREE
    extra_edges_to_pass = DEFAULT_EXTRA_EDGES
    onlooker_limit_to_pass = DEFAULT_ONLOOKER_LIMIT_PER_VERTEX
    max_k = max_degree + 1

    if num_scouts >= total_bees:
        print(f"Warning: Scouts >= Total Bees. Adjusting scouts.")
        num_scouts = max(0, total_bees - 1)

    print("\n--- Starting Execution ---")
    adjacency_list, initial_degrees = generate_graph_original(
        n_vertices=num_vertices, min_degree=min_degree, max_degree=max_degree, extra_edges=extra_edges_to_pass
        )

    if adjacency_list:
        best_coloring, performance_data, first_pass_k, min_colors_achieved = abc_graph_coloring(
            adjacency_list, initial_degrees, total_bees, num_scouts,
            max_iterations, onlooker_limit_to_pass, max_k
        )

        print("\n--- Algorithm Execution Summary ---")
        if first_pass_k != float('inf'):
             print(f"Chromatic number after first complete coloring pass: {first_pass_k}")
        else:
             final_colors_used_count_end = len({c for c in best_coloring.values() if c != -1})
             if final_colors_used_count_end > 0: print(f"Stopped before first pass complete. Colors used at end: {final_colors_used_count_end}")
             else: print("Algorithm did not color any vertices.")

        if min_colors_achieved != float('inf'):
            print(f"Best valid coloring found during run used: {min_colors_achieved} colors")
        else:
            print("No valid (conflict-free) coloring was found during the run.")

        print(f"Log file generated: abc_graph_coloring.log")

        if best_coloring:
            visualize_graph_solution(adjacency_list, best_coloring, max_k)
            plot_performance(performance_data)
        else:
            print("No best solution recorded for visualization.")
    else:
        print("Graph generation failed. Cannot run algorithm.")