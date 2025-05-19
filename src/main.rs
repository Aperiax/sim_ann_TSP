use std::sync::{Arc, Mutex};
use crate::{population::Population, graph::Graph, genetic::{run_island_model, Summary, GaParams}};
use crate::sim_ann_impl::Sim;

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
    /*let graph_test: Graph= match Graph::new(15, 1.){
        Ok(graph) => graph,
        Err(E) => panic!("{E}")
    };

    let temp = graph_test.clone();
    let mut sim = Sim::new(graph_test, 6000.);
    let opt_res = match sim.run(1000000){
        Ok(res) => res,
        Err(e) => panic!("Error occured: {e}")
    };*/



    
    // just so I can have mutliple runs, I'm gonna yank that shit into pyhton anyway 
    for _ in 0..5{
        let params = GaParams {
            pop_size: 200,
            max_generation: 900,
            migration_interval: 100,
            num_islands: 7,
            num_migrants: 5,
        };
        let graph:Graph = Graph::new(20, 1.).expect("An unexpected error occured");

        let summary = Arc::new(Mutex::new(Summary::new()));
        let graph_arc = Arc::new(graph);
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



}