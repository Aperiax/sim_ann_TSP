A semestral project implementation of TSP variant allowing revisits. 

# Solving modes: 

## Simulated annealing 
Rudimentary simulated annealing simulation with nested subpath and shortcut optimizations. Results in marginal lowering of the initial path's energy. Works well for bigger (~10000 ish) graphs. 
For smaller graphs the genetic algorithm works much better currently.
## Genetic algorithm 
(primitively) multi-threaded genetic algorithm run for the graph. Spawns num_islands threads
each with their own population, uses channels to send migrants to different threads to maintain diversity, 
optimizations for this are WIP
