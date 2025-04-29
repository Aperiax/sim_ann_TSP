use std::sync::{Arc, Mutex};
use crate::{graph::Graph, genetic::{run_island_model, Summary, GaParams}};

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


    // this draws like 2GB of memory. That was just hte IDE, it draws like 2MB for 100 slices
    
    let set_pop_size:usize = 200; 
    let num_migrants:usize = set_pop_size / 40;

    let params = GaParams {
        pop_size: set_pop_size,
        max_generation:300,
        migration_interval: 100,
        num_islands: 5,
        num_migrants: num_migrants,
    };


    let graph:Graph = Graph::new(1000, 1.).expect("An unexpected error occured");

    let summary = Arc::new(Mutex::new(Summary::new()));
    let graph_arc = Arc::new(graph.clone());
    println!("Starting genetic algo run");
    run_island_model(Arc::clone(&graph_arc), params, summary.clone());

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