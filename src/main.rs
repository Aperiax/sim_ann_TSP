use std::sync::{Arc, Mutex};
use crate::{individual::Individual, population::Population, graph::Graph, genetic::{run_island_model, Summary, GaParams}};

mod graph;
mod sim_ann_impl;
mod individual;
mod population;
mod genetic;

/*
============================ todos ============================

    TODO: don't forget to take out profanities

    TODO: make the crossover operator fitness weighed?

    TODO: implement reservoir sampling instead of full
          sorting

    TODO: make graph gen faster

    TODO: minimize the number of allocations when generating
          starting paths

    TODO: add cli parser to switch between sim and genetic

============================ todos ============================
*/

macro_rules! log_time {
    ($label:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();
        println!("{} took {:?}", $label, duration);
        result
    }};
}
fn main() {

    // WARNING, ANYTHING OVER 10000 crashes your computer
    // ... over 100 for genetic algo
    let graph_test: Graph= match Graph::new(10, 1.){
        Ok(graph) => graph,
        Err(E) => panic!("{E}")
    };
    let mut pop_test = Population::new(200, &graph_test);
    println!("pre new_generation: {}", pop_test.generation.len());
    let mut parents = pop_test.tournament_selection(0.8, 20);
    pop_test.new_generation(&mut parents, 0.5).unwrap();
    println!("post new_generation {}", pop_test.generation.len());



    // this draws like 2GB of memory. That was just hte IDE, it draws like 2MB for 100 slices
    let params = GaParams {
        pop_size: 200,
        max_generation: 300,
        migration_interval: 100,
        num_islands: 5,
        num_migrants: 5,
    };


    let graph:Graph = Graph::new(100, 1.).expect("An unexpected error occured");

    let summary = Arc::new(Mutex::new(Summary::new()));
    let graph_arc = Arc::new(graph.clone());
    println!("Starting genetic algo run");
    run_island_model(graph_arc, params, summary.clone());

    let final_summary = summary.lock().unwrap();
    if let Some(best) = &final_summary.best_indiv {
        println!("Optimal solution found");
        println!("Best fitness: {}", &best.fitness);
        println!("Best chromosome: {:?}", &best.chromosome);
        println!("Found on island id {}, in generaton {}", final_summary.island_id.unwrap_or(0), final_summary.generation.unwrap_or(0))
    } else {
        println!("No solution found!")
    }


}