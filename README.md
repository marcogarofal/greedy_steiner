# greedy_steiner

greedy approach:

python3 create_graph.py (to create a random graph)

python3 steiner.py (to find the best tree)




exaustive approach:
https://github.com/marcogarofal/exhaustive-algorithm-virtual-network_private




# Steiner Tree Capacitated Problem Solver - Simplified Version

## 🇮🇹 Italiano

### Descrizione del Problema

Questo progetto risolve una variante del **Problema dell'Albero di Steiner con Vincoli di Capacità**. L'obiettivo è connettere tutti i nodi "deboli" (weak) ai nodi "obbligatori" (mandatory) attraverso una rete ottimale, utilizzando opzionalmente nodi "discrezionali" (discretionary) come intermediari.

**Questa versione semplificata** testa solo **2 scenari**:
1. **Senza nodi discrezionali** (solo weak → mandatory)
2. **Con tutti i nodi discrezionali** disponibili

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

#### 2. **Strategia di Ottimizzazione Semplificata**
L'algoritmo testa **solo 2 combinazioni**:
- Soluzione senza nodi discrezionali
- Soluzione con **tutti** i nodi discrezionali disponibili
- Per ogni caso, applica un algoritmo greedy per scegliere i percorsi più economici

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
ALGORITMO SteinerTreeCapacitated_Simplified:

1. Carica SOLO grafo_0.pickle (nessun discretionary) e grafo_3.pickle (tutti i discretionary)

2. PER ogni grafo caricato:

   3. Estrai nodi per tipologia (weak, mandatory, discretionary)

   4. Inizializza lista_soluzioni = []

   5. // Testa soluzione senza nodi discrezionali
      soluzione_senza = RisolviConDiscrezionali([], nodi_weak, nodi_mandatory)
      Aggiungi soluzione_senza a lista_soluzioni

   6. // Testa soluzione con TUTTI i nodi discrezionali
      soluzione_con_tutti = RisolviConDiscrezionali(TUTTI_discretionary, nodi_weak, nodi_mandatory)
      Aggiungi soluzione_con_tutti a lista_soluzioni

   7. // Trova la migliore tra le 2
      migliore = ArgMin(lista_soluzioni, key=score)

   8. Mostra quale subset di discretionary è stato effettivamente utilizzato

   9. Visualizza e salva migliore

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
steiner_solver_simplified.py
├── Classi
│   ├── Node: Rappresenta un nodo con tipo e capacità
│   └── Solution: Memorizza una soluzione completa con punteggio
├── Funzioni Principali
│   ├── find_all_paths_to_mandatory(): Trova tutti i percorsi possibili
│   ├── solve_with_discretionary_subset(): Risolve con un set di nodi discrezionali
│   ├── find_best_solution_simplified(): Testa solo 2 scenari
│   └── visualize_best_solution(): Visualizza la soluzione ottimale
└── Utilità
    ├── check_path_capacity_feasible(): Verifica fattibilità capacità
    ├── save_solution_summary(): Salva riassunto soluzioni
    └── draw_graph(): Disegna il grafo base
```

### Vantaggi della Versione Semplificata

- ⚡ **Molto più veloce**: Testa solo 2 scenari invece di 2^n combinazioni
- 🎯 **Approccio pragmatico**: Confronta gli estremi (nessun vs tutti i discretionary)
- 💡 **Insight utili**: Mostra quali nodi discretionary vengono effettivamente utilizzati
- 🔍 **Facile da analizzare**: Output più semplice da interpretare

### Limitazioni

- ❌ **Non garantisce l'ottimo globale**: Potrebbe esistere una combinazione intermedia migliore
- ⚠️ **Approccio euristico**: Sacrifica la completezza per la velocità

### Output

Il programma genera:
- **Grafici PNG**: Visualizzazione della soluzione ottimale
- **File di testo**: Riassunto delle 2 soluzioni testate
- **Console output**: Debug dettagliato con insight sui nodi discretionary utilizzati

---

## 🇬🇧 English

### Problem Description

This project solves a variant of the **Capacitated Steiner Tree Problem**. The goal is to connect all "weak" nodes to "mandatory" nodes through an optimal network, optionally using "discretionary" nodes as intermediaries.

**This simplified version** tests only **2 scenarios**:
1. **Without discretionary nodes** (only weak → mandatory)
2. **With all discretionary nodes** available

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

#### 2. **Simplified Optimization Strategy**
The algorithm tests **only 2 combinations**:
- Solution without discretionary nodes
- Solution with **all** available discretionary nodes
- For each case, applies a greedy algorithm to choose the cheapest paths

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
ALGORITHM SteinerTreeCapacitated_Simplified:

1. Load ONLY grafo_0.pickle (no discretionary) and grafo_3.pickle (all discretionary)

2. FOR each loaded graph:

   3. Extract nodes by type (weak, mandatory, discretionary)

   4. Initialize solutions_list = []

   5. // Test solution without discretionary nodes
      solution_without = SolveWithDiscretionary([], weak_nodes, mandatory_nodes)
      Add solution_without to solutions_list

   6. // Test solution with ALL discretionary nodes
      solution_with_all = SolveWithDiscretionary(ALL_discretionary, weak_nodes, mandatory_nodes)
      Add solution_with_all to solutions_list

   7. // Find the best between the 2
      best = ArgMin(solutions_list, key=score)

   8. Show which subset of discretionary was actually used

   9. Visualize and save best

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
steiner_solver_simplified.py
├── Classes
│   ├── Node: Represents a node with type and capacity
│   └── Solution: Stores a complete solution with score
├── Main Functions
│   ├── find_all_paths_to_mandatory(): Finds all possible paths
│   ├── solve_with_discretionary_subset(): Solves with a set of discretionary nodes
│   ├── find_best_solution_simplified(): Tests only 2 scenarios
│   └── visualize_best_solution(): Visualizes the optimal solution
└── Utilities
    ├── check_path_capacity_feasible(): Checks capacity feasibility
    ├── save_solution_summary(): Saves solution summary
    └── draw_graph(): Draws the base graph
```

### Advantages of the Simplified Version

- ⚡ **Much faster**: Tests only 2 scenarios instead of 2^n combinations
- 🎯 **Pragmatic approach**: Compares extremes (no vs all discretionary)
- 💡 **Useful insights**: Shows which discretionary nodes are actually used
- 🔍 **Easy to analyze**: Simpler output to interpret

### Limitations

- ❌ **Does not guarantee global optimum**: A better intermediate combination might exist
- ⚠️ **Heuristic approach**: Sacrifices completeness for speed

### Output

The program generates:
- **PNG graphics**: Visualization of the optimal solution
- **Text files**: Summary of the 2 tested solutions
- **Console output**: Detailed debug with insights on used discretionary nodes

---

## Requirements

```python
networkx
matplotlib
pickle
```

## Usage

```bash
python steiner_solver_simplified.py
```

The program expects pickle files `grafo_0.pickle` and `grafo_3.pickle` in the `graphs/` directory.

## Key Features

- ✅ **Simplified Search**: Tests only 2 strategic scenarios
- ✅ **Multi-objective Optimization**: Balances connection cost, capacity constraints, and efficiency
- ✅ **Capacity Management**: Handles node capacity constraints with penalties
- ✅ **Visualization**: Generates detailed graphs of the optimal solution
- ✅ **Insight Generation**: Shows which discretionary nodes are actually used when all are available
- ✅ **Flexible Path Finding**: Supports paths up to 3 hops (weak → disc1 → disc2 → mandatory)

## Comparison with Full Version

| Feature | Full Version | Simplified Version |
|---------|-------------|-------------------|
| **Scenarios Tested** | 2^n combinations | 2 scenarios |
| **Execution Time** | O(2^n) | O(2) |
| **Optimality** | Guaranteed global optimum | Good heuristic solution |
| **Insights** | Complete analysis | Focus on extremes |
| **Use Case** | Research/Complete analysis | Quick decision making |

## When to Use This Version

- ✅ **Quick prototyping**: When you need fast results
- ✅ **Large problems**: When full enumeration is computationally prohibitive
- ✅ **Practical decisions**: When you want to compare "baseline" vs "full resources"
- ✅ **Initial analysis**: To understand if discretionary nodes provide significant benefit

## When to Use the Full Version

- ✅ **Critical applications**: When you need the guaranteed optimal solution
- ✅ **Small problems**: When computational cost is not a concern
- ✅ **Research**: When you need complete analysis of all possibilities
- ✅ **Benchmarking**: To validate the simplified version's results
