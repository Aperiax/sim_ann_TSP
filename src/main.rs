use crate::graph::Graph;
use crate::sim_ann_impl::Sim;

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
    let graph = Graph::new(10, 1.).expect("An unexpected error occured");

    //println!("{graph}");
    let _ = log_time!("neighbr retrieval current code", {graph.get_neighbors(0)});
    let _ = log_time!("initial path generation", {
        Sim::generate_tour(&graph)
    });


    let mut sim_test = Sim::new(graph, 10000.);
    let pre_run_enrgy: usize = sim_test.energy.clone();
    let pre_run_path: Vec<usize> = sim_test.path.clone();
    let (sol, en) = sim_test.run(100).expect("Error in simulation");

    println!("Energy pre-run: {pre_run_enrgy}, post-run: {en}");
    println!("Path pre-run: {:?}", pre_run_path);
    println!("Path pst-run: {:?}", sol)
}
