use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use crossbeam::channel::{unbounded, Receiver, Sender};
use crate::{graph::Graph, population::Population, individual::Individual};



#[derive(Clone)]
pub struct GaParams {
    // general parameters for the top level run
    pub pop_size: usize,
    pub max_generation: usize,
    pub migration_interval: usize,
    pub num_migrants: usize,
    pub num_islands: usize, // corresponds to the number of system's threads
}

#[derive(Debug, Clone)]
pub struct Summary{
    pub best_indiv: Option<Individual>,
    pub island_id: Option<usize>,
    pub generation: Option<usize>,
}

impl Summary{
    pub fn new() -> Self{
        Summary{best_indiv: None, island_id: None, generation:None}
    }

    fn update(&mut self, candidate: Individual, island:usize, generation:usize){
        if self.best_indiv.is_none() || candidate.fitness < self.best_indiv.as_ref().unwrap().fitness{
            self.best_indiv = Some(candidate);
            self.island_id = Some(island);
            self.generation = Some(generation);
        }
    }
}

pub fn run_island_model(graph: Arc<Graph>, params: GaParams, summary: Arc<Mutex<Summary>>){

    let mut handles = Vec::with_capacity(params.num_islands);
    let barrier = Arc::new(Barrier::new(params.num_islands));

    let mut senders: Vec<Vec<Sender<Vec<Individual>>>> = vec![Vec::new(); params.num_islands];
    let mut receivers: Vec<Vec<Receiver<Vec<Individual>>>> = vec![Vec::new(); params.num_islands];

    for i in 0..params.num_islands{
        for j in 0..params.num_islands{
            if i != j{
                let (s,r) = unbounded();
                senders[i].push(s);
                receivers[j].push(r)
            }
        }
    }
    println!(" Starting thread spwan ");

    // FIXME: the fucking threads are panicking like headless chicken
    for i in 0..params.num_islands{
        let island_summary = Arc::clone(&summary);
        let island_graph = Arc::clone(&graph);
        let island_params = params.clone();
        let island_senders = senders[i].clone();
        let island_receivers = receivers[i].clone();
        let island_barrier = barrier.clone();
        let handle = thread::spawn(move ||{

            let mut population= Population::new(params.pop_size, &island_graph);

            let best_start = population.generation.iter().min_by_key(|ind| ind.fitness).unwrap();
            println!("Thread {i} best starting: {}", best_start.fitness);
            island_barrier.wait();

            for gen in 0..island_params.max_generation{

                if i == 1 && gen % 100 == 0
                {
                    println!("current gen on core 1: {gen}")
                }
                let mut parents = population.tournament_selection(0.5, 20);

                // this line is the problem
                population.new_generation(&mut parents).unwrap();

                if let Some(candidate) = population.generation.iter()
                    .min_by_key(|ind| ind.fitness)
                    .cloned(){
                    let mut sum = island_summary.lock().unwrap();
                    sum.update(candidate, i, gen)
                }


                if gen % island_params.migration_interval == 0 && gen > 0{
                    island_barrier.wait();
                    let migrants = population.tournament_selection(0.5, island_params.num_migrants);
                    for sender in &island_senders{
                        sender.send(migrants.clone()).unwrap();
                    }
                    island_barrier.wait();
                    let mut all_migrants = Vec::new();

                    island_barrier.wait();
                    for receiver in &island_receivers{
                        let incoming = receiver.recv().unwrap();

                        all_migrants.extend(incoming)
                    }
                    population.generation.extend(all_migrants);
                    population.generation.sort_by(|a, b| a.fitness.cmp(&b.fitness));
                    population.generation.truncate(island_params.pop_size);
                    island_barrier.wait();

                }
            }
        });
        handles.push(handle)
    }
    for handle in handles{
        handle.join().unwrap();
    }
    println!("All islands finished successfully")
}
