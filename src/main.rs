use std::sync::{Arc, Mutex};
use crate::graph::Graph;
use crate::sim_ann_impl::{run_island_model, GaParams, Population, Sim, Summary};
use std::thread;
use ndarray::indices;

mod graph;
mod sim_ann_impl;

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





    // this draws like 2GB of memory. That was just hte IDE, it draws like 2MB for 100 slices
    let params = GaParams {
        pop_size: 200,
        max_generation: 100,
        migration_interval: 5,
        num_islands: 4,
        num_migrants: 10,
    };


    let graph:Graph = Graph::new(10, 1.).expect("An unexpected error occured");
    let mut population = Population::new(200, &graph);
    let summary = Arc::new(Mutex::new(Summary::new()));
    let graph_arc = Arc::new(graph.clone());
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