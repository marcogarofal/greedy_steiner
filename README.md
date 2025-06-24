# greedy_steiner

greedy approach:

python3 create_graph.py (to create a random graph)

python3 steiner.py (to find the best tree)




exaustive approach:
https://github.com/marcogarofal/exhaustive-algorithm-virtual-network_private




# Steiner Tree Capacitated Problem Solver

## 🇮🇹 Italiano

### Descrizione del Problema

Questo progetto risolve una variante del **Problema dell'Albero di Steiner con Vincoli di Capacità**. L'obiettivo è connettere tutti i nodi "deboli" (weak) ai nodi "obbligatori" (mandatory) attraverso una rete ottimale, utilizzando opzionalmente nodi "discrezionali" (discretionary) come intermediari.

### Tipologie di Nodi

- **Nodi Deboli (Weak)**: Nodi che devono essere connessi alla rete
- **Nodi Obbligatori (Mandatory)**: Nodi sempre disponibili con capacità limitata
- **Nodi Discrezionali (Discretionary)**: Nodi opzionali che possono essere utilizzati come intermediari

### Principio di Funzionamento

#### 1. **Ricerca Percorsi**
Per ogni nodo debole, l'algoritmo trova tutti i possibili percorsi verso i nodi obbligatori:
- Connessioni dirette (weak → mandatory)
- Connessioni attraverso 1 nodo discrezionale (weak → discretionary → mandatory)
- Connessioni attraverso 2 nodi discrezionali (weak → disc1 → disc2 → mandatory)

#### 2. **Strategia di Ottimizzazione**
L'algoritmo testa **tutte le possibili combinazioni** di nodi discrezionali:
- Soluzione senza nodi discrezionali
- Soluzioni con ogni possibile sottoinsiemi di nodi discrezionali
- Per ogni combinazione, applica un algoritmo greedy per scegliere i percorsi più economici

#### 3. **Funzione di Punteggio**
Ogni soluzione viene valutata con una funzione di punteggio multi-obiettivo:
```
Score = Penalità_Connessioni_Fallite + Costo_Archi + Penalità_Violazioni_Capacità + Costo_Efficienza
```

#### 4. **Gestione delle Capacità**
- Ogni nodo obbligatorio e discrezionale ha una capacità massima
- Se la capacità viene superata, si applica una forte penalità
- Si calcola un costo di efficienza basato sull'utilizzo delle capacità

### Algoritmo Principale (Pseudocodice)

```
ALGORITMO SteinerTreeCapacitated:

1. PER ogni grafo caricato:

   2. Estrai nodi per tipologia (weak, mandatory, discretionary)

   3. Inizializza lista_soluzioni = []

   4. // Testa soluzione senza nodi discrezionali
      soluzione_base = RisolviConDiscrezionali([], nodi_weak, nodi_mandatory)
      Aggiungi soluzione_base a lista_soluzioni

   5. // Testa tutte le combinazioni di nodi discrezionali
      PER ogni sottinsieme S di nodi_discretionary:
         soluzione = RisolviConDiscrezionali(S, nodi_weak, nodi_mandatory)
         Aggiungi soluzione a lista_soluzioni

   6. // Trova la migliore
      migliore = ArgMin(lista_soluzioni, key=score)

   7. Visualizza e salva migliore

FUNZIONE RisolviConDiscrezionali(discrezionali, weak, mandatory):

   1. Inizializza utilizzo_capacità, connessi, albero_steiner

   2. // Trova tutti i percorsi possibili
      PER ogni nodo_weak:
         percorsi = TrovaTuttiPercorsi(nodo_weak, mandatory, discrezionali)

   3. // Crea lista di tutte le opzioni di connessione
      opzioni = []
      PER ogni nodo_weak:
         PER ogni percorso in percorsi[nodo_weak]:
            Aggiungi {nodo_weak, percorso, costo} a opzioni

   4. // Ordina per costo crescente
      Ordina opzioni per costo

   5. // Algoritmo greedy
      PER ogni opzione in opzioni:
         SE nodo_weak non è già connesso E capacità_sufficiente:
            Connetti usando questo percorso
            Aggiorna utilizzo_capacità
            Aggiungi archi ad albero_steiner

   6. // Forza connessione dei nodi rimanenti
      PER ogni nodo_weak non connesso:
         Trova percorso più economico disponibile
         Connetti anche se viola capacità

   7. Calcola punteggio e ritorna Soluzione

FUNZIONE TrovaTuttiPercorsi(weak, mandatory_list, discretionary_list):

   percorsi = []

   // Percorsi diretti
   PER ogni mandatory in mandatory_list:
      SE esiste_arco(weak, mandatory):
         Aggiungi percorso [weak, mandatory]

   // Percorsi con 1 intermediario
   PER ogni disc in discretionary_list:
      SE esiste_arco(weak, disc):
         PER ogni mandatory in mandatory_list:
            SE esiste_arco(disc, mandatory):
               Aggiungi percorso [weak, disc, mandatory]

   // Percorsi con 2 intermediari
   PER ogni disc1 in discretionary_list:
      PER ogni disc2 in discretionary_list (disc1 ≠ disc2):
         SE esiste_arco(weak, disc1) E esiste_arco(disc1, disc2):
            PER ogni mandatory in mandatory_list:
               SE esiste_arco(disc2, mandatory):
                  Aggiungi percorso [weak, disc1, disc2, mandatory]

   Ordina percorsi per costo
   Ritorna percorsi
```

### Struttura del Codice

```
steiner_solver.py
├── Classi
│   ├── Node: Rappresenta un nodo con tipo e capacità
│   └── Solution: Memorizza una soluzione completa con punteggio
├── Funzioni Principali
│   ├── find_all_paths_to_mandatory(): Trova tutti i percorsi possibili
│   ├── solve_with_discretionary_subset(): Risolve con un set di nodi discrezionali
│   ├── find_best_solution_overall(): Testa tutte le combinazioni
│   └── visualize_best_solution(): Visualizza la soluzione ottimale
└── Utilità
    ├── check_path_capacity_feasible(): Verifica fattibilità capacità
    ├── save_solution_summary(): Salva riassunto soluzioni
    └── draw_graph(): Disegna il grafo base
```

### Output

Il programma genera:
- **Grafici PNG**: Visualizzazione della soluzione ottimale
- **File di testo**: Riassunto completo di tutte le soluzioni testate
- **Console output**: Debug dettagliato del processo di ottimizzazione

---

## 🇬🇧 English

### Problem Description

This project solves a variant of the **Capacitated Steiner Tree Problem**. The goal is to connect all "weak" nodes to "mandatory" nodes through an optimal network, optionally using "discretionary" nodes as intermediaries.

### Node Types

- **Weak Nodes**: Nodes that must be connected to the network
- **Mandatory Nodes**: Always available nodes with limited capacity
- **Discretionary Nodes**: Optional nodes that can be used as intermediaries

### Working Principle

#### 1. **Path Finding**
For each weak node, the algorithm finds all possible paths to mandatory nodes:
- Direct connections (weak → mandatory)
- Connections through 1 discretionary node (weak → discretionary → mandatory)
- Connections through 2 discretionary nodes (weak → disc1 → disc2 → mandatory)

#### 2. **Optimization Strategy**
The algorithm tests **all possible combinations** of discretionary nodes:
- Solution without discretionary nodes
- Solutions with every possible subset of discretionary nodes
- For each combination, applies a greedy algorithm to choose the cheapest paths

#### 3. **Scoring Function**
Each solution is evaluated with a multi-objective scoring function:
```
Score = Failed_Connections_Penalty + Edge_Cost + Capacity_Violations_Penalty + Efficiency_Cost
```

#### 4. **Capacity Management**
- Each mandatory and discretionary node has a maximum capacity
- If capacity is exceeded, a heavy penalty is applied
- An efficiency cost is calculated based on capacity utilization

### Main Algorithm (Pseudocode)

```
ALGORITHM SteinerTreeCapacitated:

1. FOR each loaded graph:

   2. Extract nodes by type (weak, mandatory, discretionary)

   3. Initialize solutions_list = []

   4. // Test solution without discretionary nodes
      base_solution = SolveWithDiscretionary([], weak_nodes, mandatory_nodes)
      Add base_solution to solutions_list

   5. // Test all combinations of discretionary nodes
      FOR each subset S of discretionary_nodes:
         solution = SolveWithDiscretionary(S, weak_nodes, mandatory_nodes)
         Add solution to solutions_list

   6. // Find the best one
      best = ArgMin(solutions_list, key=score)

   7. Visualize and save best

FUNCTION SolveWithDiscretionary(discretionary, weak, mandatory):

   1. Initialize capacity_usage, connected, steiner_tree

   2. // Find all possible paths
      FOR each weak_node:
         paths = FindAllPaths(weak_node, mandatory, discretionary)

   3. // Create list of all connection options
      options = []
      FOR each weak_node:
         FOR each path in paths[weak_node]:
            Add {weak_node, path, cost} to options

   4. // Sort by increasing cost
      Sort options by cost

   5. // Greedy algorithm
      FOR each option in options:
         IF weak_node not already connected AND sufficient_capacity:
            Connect using this path
            Update capacity_usage
            Add edges to steiner_tree

   6. // Force connection of remaining nodes
      FOR each unconnected weak_node:
         Find cheapest available path
         Connect even if violates capacity

   7. Calculate score and return Solution

FUNCTION FindAllPaths(weak, mandatory_list, discretionary_list):

   paths = []

   // Direct paths
   FOR each mandatory in mandatory_list:
      IF edge_exists(weak, mandatory):
         Add path [weak, mandatory]

   // Paths with 1 intermediary
   FOR each disc in discretionary_list:
      IF edge_exists(weak, disc):
         FOR each mandatory in mandatory_list:
            IF edge_exists(disc, mandatory):
               Add path [weak, disc, mandatory]

   // Paths with 2 intermediaries
   FOR each disc1 in discretionary_list:
      FOR each disc2 in discretionary_list (disc1 ≠ disc2):
         IF edge_exists(weak, disc1) AND edge_exists(disc1, disc2):
            FOR each mandatory in mandatory_list:
               IF edge_exists(disc2, mandatory):
                  Add path [weak, disc1, disc2, mandatory]

   Sort paths by cost
   Return paths
```

### Code Structure

```
steiner_solver.py
├── Classes
│   ├── Node: Represents a node with type and capacity
│   └── Solution: Stores a complete solution with score
├── Main Functions
│   ├── find_all_paths_to_mandatory(): Finds all possible paths
│   ├── solve_with_discretionary_subset(): Solves with a set of discretionary nodes
│   ├── find_best_solution_overall(): Tests all combinations
│   └── visualize_best_solution(): Visualizes the optimal solution
└── Utilities
    ├── check_path_capacity_feasible(): Checks capacity feasibility
    ├── save_solution_summary(): Saves solution summary
    └── draw_graph(): Draws the base graph
```

### Output

The program generates:
- **PNG graphics**: Visualization of the optimal solution
- **Text files**: Complete summary of all tested solutions
- **Console output**: Detailed debug of the optimization process

---

## Requirements

```python
networkx
matplotlib
heapq
pickle
itertools
```

## Usage

```bash
python steiner_solver.py
```

The program expects pickle files named `grafo_0.pickle`, `grafo_1.pickle`, etc. in the `graphs/` directory.

## Key Features

- ✅ **Exhaustive Search**: Tests all possible combinations of discretionary nodes
- ✅ **Multi-objective Optimization**: Balances connection cost, capacity constraints, and efficiency
- ✅ **Capacity Management**: Handles node capacity constraints with penalties
- ✅ **Visualization**: Generates detailed graphs of the optimal solution
- ✅ **Comprehensive Reporting**: Saves detailed analysis of all tested solutions
- ✅ **Flexible Path Finding**: Supports paths up to 3 hops (weak → disc1 → disc2 → mandatory)




















versione futura con max_hops per determinare quanti discretionary node usare al massimo per collegare weak a mandatory (non implementato):


# Steiner Tree Capacitated Problem Solver

## 🇮🇹 Italiano

### Descrizione del Problema

Questo progetto risolve una variante del **Problema dell'Albero di Steiner con Vincoli di Capacità**. L'obiettivo è connettere tutti i nodi "deboli" (weak) ai nodi "obbligatori" (mandatory) attraverso una rete ottimale, utilizzando opzionalmente nodi "discrezionali" (discretionary) come intermediari.

### Tipologie di Nodi

- **Nodi Deboli (Weak)**: Nodi che devono essere connessi alla rete
- **Nodi Obbligatori (Mandatory)**: Nodi sempre disponibili con capacità limitata
- **Nodi Discrezionali (Discretionary)**: Nodi opzionali che possono essere utilizzati come intermediari

### Principio di Funzionamento

#### 1. **Ricerca Percorsi**
Per ogni nodo debole, l'algoritmo trova tutti i possibili percorsi verso i nodi obbligatori:
- Connessioni dirette (weak → mandatory)
- Connessioni attraverso **n** nodi discrezionali (weak → disc1 → disc2 → ... → discN → mandatory)
- La profondità di ricerca è configurabile tramite il parametro `max_hops`

#### 2. **Strategia di Ottimizzazione**
L'algoritmo testa **tutte le possibili combinazioni** di nodi discrezionali:
- Soluzione senza nodi discrezionali
- Soluzioni con ogni possibile sottoinsiemi di nodi discrezionali
- Per ogni combinazione, applica un algoritmo greedy per scegliere i percorsi più economici

#### 3. **Funzione di Punteggio**
Ogni soluzione viene valutata con una funzione di punteggio multi-obiettivo:
```
Score = Penalità_Connessioni_Fallite + Costo_Archi + Penalità_Violazioni_Capacità + Costo_Efficienza
```

#### 4. **Gestione delle Capacità**
- Ogni nodo obbligatorio e discrezionale ha una capacità massima
- Se la capacità viene superata, si applica una forte penalità
- Si calcola un costo di efficienza basato sull'utilizzo delle capacità

### Algoritmo Principale (Pseudocodice)

```
ALGORITMO SteinerTreeCapacitated:

1. PER ogni grafo caricato:

   2. Estrai nodi per tipologia (weak, mandatory, discretionary)

   3. Inizializza lista_soluzioni = []

   4. // Testa soluzione senza nodi discrezionali
      soluzione_base = RisolviConDiscrezionali([], nodi_weak, nodi_mandatory)
      Aggiungi soluzione_base a lista_soluzioni

   5. // Testa tutte le combinazioni di nodi discrezionali
      PER ogni sottinsieme S di nodi_discretionary:
         soluzione = RisolviConDiscrezionali(S, nodi_weak, nodi_mandatory)
         Aggiungi soluzione a lista_soluzioni

   6. // Trova la migliore
      migliore = ArgMin(lista_soluzioni, key=score)

   7. Visualizza e salva migliore

FUNZIONE RisolviConDiscrezionali(discrezionali, weak, mandatory):

   1. Inizializza utilizzo_capacità, connessi, albero_steiner

   2. // Trova tutti i percorsi possibili
      PER ogni nodo_weak:
         percorsi = TrovaTuttiPercorsi(nodo_weak, mandatory, discrezionali)

   3. // Crea lista di tutte le opzioni di connessione
      opzioni = []
      PER ogni nodo_weak:
         PER ogni percorso in percorsi[nodo_weak]:
            Aggiungi {nodo_weak, percorso, costo} a opzioni

   4. // Ordina per costo crescente
      Ordina opzioni per costo

   5. // Algoritmo greedy
      PER ogni opzione in opzioni:
         SE nodo_weak non è già connesso E capacità_sufficiente:
            Connetti usando questo percorso
            Aggiorna utilizzo_capacità
            Aggiungi archi ad albero_steiner

   6. // Forza connessione dei nodi rimanenti
      PER ogni nodo_weak non connesso:
         Trova percorso più economico disponibile
         Connetti anche se viola capacità

   7. Calcola punteggio e ritorna Soluzione

FUNZIONE TrovaTuttiPercorsi(weak, mandatory_list, discretionary_list, max_hops):

   percorsi = []

   // Percorsi diretti (0 intermediari)
   PER ogni mandatory in mandatory_list:
      SE esiste_arco(weak, mandatory):
         Aggiungi percorso [weak, mandatory]

   // Percorsi con intermediari (ricerca DFS fino a max_hops)
   PER hops da 1 a max_hops:
      percorsi_con_hops = TrovaPercorsiConNIntermediari(weak, mandatory_list, discretionary_list, hops)
      Aggiungi tutti percorsi_con_hops a percorsi

   Ordina percorsi per costo
   Ritorna percorsi

FUNZIONE TrovaPercorsiConNIntermediari(weak, mandatory_list, discretionary_list, n_hops):

   SE n_hops == 0:
      // Caso base: connessione diretta
      percorsi = []
      PER ogni mandatory in mandatory_list:
         SE esiste_arco(weak, mandatory):
            Aggiungi [weak, mandatory] a percorsi
      Ritorna percorsi

   ALTRIMENTI:
      // Caso ricorsivo: aggiungi un intermediario
      percorsi = []
      PER ogni disc in discretionary_list:
         SE esiste_arco(weak, disc):
            // Ricerca ricorsiva dal nodo disc
            sotto_percorsi = TrovaPercorsiConNIntermediari(disc, mandatory_list,
                                                          discretionary_list - {disc}, n_hops - 1)
            PER ogni sotto_percorso in sotto_percorsi:
               percorso_completo = [weak] + sotto_percorso
               SE percorso_completo è valido (no cicli):
                  Aggiungi percorso_completo a percorsi

      Ritorna percorsi

ALTERNATIVA CON BFS (Breadth-First Search):

FUNZIONE TrovaTuttiPercorsiBFS(weak, mandatory_list, discretionary_list, max_hops):

   percorsi = []
   coda = [(weak, [weak], 0)]  // (nodo_corrente, percorso, hops_usati)
   visitati_per_percorso = set()

   MENTRE coda non è vuota:
      nodo_corrente, percorso, hops = estrai da coda

      // Se raggiunto un nodo mandatory, salva il percorso
      SE nodo_corrente in mandatory_list:
         Aggiungi percorso a percorsi
         CONTINUA

      // Se raggiunti max_hops, ferma l'esplorazione
      SE hops >= max_hops:
         CONTINUA

      // Esplora tutti i vicini discretionary
      PER ogni vicino in discretionary_list:
         SE esiste_arco(nodo_corrente, vicino) E vicino non in percorso:
            nuovo_percorso = percorso + [vicino]
            stato = (vicino, tuple(nuovo_percorso))
            SE stato non in visitati_per_percorso:
               Aggiungi stato a visitati_per_percorso
               Aggiungi (vicino, nuovo_percorso, hops + 1) a coda

   Ordina percorsi per costo totale
   Ritorna percorsi
```

### Struttura del Codice

```
steiner_solver.py
├── Classi
│   ├── Node: Rappresenta un nodo con tipo e capacità
│   └── Solution: Memorizza una soluzione completa con punteggio
├── Funzioni Principali
│   ├── find_all_paths_to_mandatory(): Trova tutti i percorsi possibili
│   ├── solve_with_discretionary_subset(): Risolve con un set di nodi discrezionali
│   ├── find_best_solution_overall(): Testa tutte le combinazioni
│   └── visualize_best_solution(): Visualizza la soluzione ottimale
└── Utilità
    ├── check_path_capacity_feasible(): Verifica fattibilità capacità
    ├── save_solution_summary(): Salva riassunto soluzioni
    └── draw_graph(): Disegna il grafo base
```

### Output

Il programma genera:
- **Grafici PNG**: Visualizzazione della soluzione ottimale
- **File di testo**: Riassunto completo di tutte le soluzioni testate
- **Console output**: Debug dettagliato del processo di ottimizzazione

---

## 🇬🇧 English

### Problem Description

This project solves a variant of the **Capacitated Steiner Tree Problem**. The goal is to connect all "weak" nodes to "mandatory" nodes through an optimal network, optionally using "discretionary" nodes as intermediaries.

### Node Types

- **Weak Nodes**: Nodes that must be connected to the network
- **Mandatory Nodes**: Always available nodes with limited capacity
- **Discretionary Nodes**: Optional nodes that can be used as intermediaries

### Working Principle

#### 1. **Path Finding**
For each weak node, the algorithm finds all possible paths to mandatory nodes:
- Direct connections (weak → mandatory)
- Connections through **n** discretionary nodes (weak → disc1 → disc2 → ... → discN → mandatory)
- Search depth is configurable via the `max_hops` parameter

#### 2. **Optimization Strategy**
The algorithm tests **all possible combinations** of discretionary nodes:
- Solution without discretionary nodes
- Solutions with every possible subset of discretionary nodes
- For each combination, applies a greedy algorithm to choose the cheapest paths

#### 3. **Scoring Function**
Each solution is evaluated with a multi-objective scoring function:
```
Score = Failed_Connections_Penalty + Edge_Cost + Capacity_Violations_Penalty + Efficiency_Cost
```

#### 4. **Capacity Management**
- Each mandatory and discretionary node has a maximum capacity
- If capacity is exceeded, a heavy penalty is applied
- An efficiency cost is calculated based on capacity utilization

### Main Algorithm (Pseudocode)

```
ALGORITHM SteinerTreeCapacitated:

1. FOR each loaded graph:

   2. Extract nodes by type (weak, mandatory, discretionary)

   3. Initialize solutions_list = []

   4. // Test solution without discretionary nodes
      base_solution = SolveWithDiscretionary([], weak_nodes, mandatory_nodes)
      Add base_solution to solutions_list

   5. // Test all combinations of discretionary nodes
      FOR each subset S of discretionary_nodes:
         solution = SolveWithDiscretionary(S, weak_nodes, mandatory_nodes)
         Add solution to solutions_list

   6. // Find the best one
      best = ArgMin(solutions_list, key=score)

   7. Visualize and save best

FUNCTION SolveWithDiscretionary(discretionary, weak, mandatory):

   1. Initialize capacity_usage, connected, steiner_tree

   2. // Find all possible paths
      FOR each weak_node:
         paths = FindAllPaths(weak_node, mandatory, discretionary)

   3. // Create list of all connection options
      options = []
      FOR each weak_node:
         FOR each path in paths[weak_node]:
            Add {weak_node, path, cost} to options

   4. // Sort by increasing cost
      Sort options by cost

   5. // Greedy algorithm
      FOR each option in options:
         IF weak_node not already connected AND sufficient_capacity:
            Connect using this path
            Update capacity_usage
            Add edges to steiner_tree

   6. // Force connection of remaining nodes
      FOR each unconnected weak_node:
         Find cheapest available path
         Connect even if violates capacity

   7. Calculate score and return Solution

FUNCTION FindAllPaths(weak, mandatory_list, discretionary_list, max_hops):

   paths = []

   // Direct paths (0 intermediaries)
   FOR each mandatory in mandatory_list:
      IF edge_exists(weak, mandatory):
         Add path [weak, mandatory]

   // Paths with intermediaries (DFS search up to max_hops)
   FOR hops from 1 to max_hops:
      paths_with_hops = FindPathsWithNIntermediaries(weak, mandatory_list, discretionary_list, hops)
      Add all paths_with_hops to paths

   Sort paths by cost
   Return paths

FUNCTION FindPathsWithNIntermediaries(weak, mandatory_list, discretionary_list, n_hops):

   IF n_hops == 0:
      // Base case: direct connection
      paths = []
      FOR each mandatory in mandatory_list:
         IF edge_exists(weak, mandatory):
            Add [weak, mandatory] to paths
      Return paths

   ELSE:
      // Recursive case: add one intermediary
      paths = []
      FOR each disc in discretionary_list:
         IF edge_exists(weak, disc):
            // Recursive search from disc node
            sub_paths = FindPathsWithNIntermediaries(disc, mandatory_list,
                                                   discretionary_list - {disc}, n_hops - 1)
            FOR each sub_path in sub_paths:
               complete_path = [weak] + sub_path
               IF complete_path is valid (no cycles):
                  Add complete_path to paths

      Return paths

ALTERNATIVE WITH BFS (Breadth-First Search):

FUNCTION FindAllPathsBFS(weak, mandatory_list, discretionary_list, max_hops):

   paths = []
   queue = [(weak, [weak], 0)]  // (current_node, path, hops_used)
   visited_per_path = set()

   WHILE queue is not empty:
      current_node, path, hops = extract from queue

      // If reached a mandatory node, save the path
      IF current_node in mandatory_list:
         Add path to paths
         CONTINUE

      // If reached max_hops, stop exploration
      IF hops >= max_hops:
         CONTINUE

      // Explore all discretionary neighbors
      FOR each neighbor in discretionary_list:
         IF edge_exists(current_node, neighbor) AND neighbor not in path:
            new_path = path + [neighbor]
            state = (neighbor, tuple(new_path))
            IF state not in visited_per_path:
               Add state to visited_per_path
               Add (neighbor, new_path, hops + 1) to queue

   Sort paths by total cost
   Return paths
```

### Code Structure

```
steiner_solver.py
├── Classes
│   ├── Node: Represents a node with type and capacity
│   └── Solution: Stores a complete solution with score
├── Main Functions
│   ├── find_all_paths_to_mandatory(): Finds all possible paths
│   ├── solve_with_discretionary_subset(): Solves with a set of discretionary nodes
│   ├── find_best_solution_overall(): Tests all combinations
│   └── visualize_best_solution(): Visualizes the optimal solution
└── Utilities
    ├── check_path_capacity_feasible(): Checks capacity feasibility
    ├── save_solution_summary(): Saves solution summary
    └── draw_graph(): Draws the base graph
```

### Output

The program generates:
- **PNG graphics**: Visualization of the optimal solution
- **Text files**: Complete summary of all tested solutions
- **Console output**: Detailed debug of the optimization process

---

## Requirements

```python
networkx
matplotlib
heapq
pickle
itertools
```

## Usage

```bash
python steiner_solver.py
```

The program expects pickle files named `grafo_0.pickle`, `grafo_1.pickle`, etc. in the `graphs/` directory.

## Key Features

- ✅ **Exhaustive Search**: Tests all possible combinations of discretionary nodes
- ✅ **Multi-objective Optimization**: Balances connection cost, capacity constraints, and efficiency
- ✅ **Capacity Management**: Handles node capacity constraints with penalties
- ✅ **Visualization**: Generates detailed graphs of the optimal solution
- ✅ **Comprehensive Reporting**: Saves detailed analysis of all tested solutions
- ✅ **Flexible Path Finding**: Supports paths with configurable number of intermediary hops (weak → disc1 → ... → discN → mandatory)
