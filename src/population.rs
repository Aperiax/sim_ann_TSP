use std::collections::HashSet;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use itertools::Itertools;
use crate::{individual::Individual, graph::Graph};




pub fn fisher_yates_variable<T>(iter: &mut Vec<T>, shuffle_frac:f64) -> &mut Vec<T> {

    // just a fast~ish shuffle that can swap a certain degree of iterator
    let len: usize = iter.len();
    let swaps = (shuffle_frac * len as f64).round() as usize;


    for _ in 0..swaps{
        let a = fastrand::usize(1..iter.len()-1);
        let b = fastrand::usize(1..iter.len()-1);

        iter.swap(a, b)
    }
    iter
}
#[derive(Debug, Clone)]
pub struct Population<'a> {
    city: &'a Graph,
    pop_size: usize,
    pub generation: Vec<Individual>,
    mutation_rate: f64,
    best_result_fitness: usize,
    best_result_chromosome: Vec<usize>
}
impl<'a> Population<'a> {
    pub fn new(pop_size:usize, city: &'a Graph) -> Self{

        // population sizes are going to be the same for each generation,
        // it's mostly about how I splice and mix and mutate the parents



        // sleek way to do it with rayon. but where's the fun in that
        let individuals = match Self::spawn_generation_zero(&city, pop_size, 8){
            Ok(generation) => generation,
            Err(E) => panic!("Generation spawning failed"),
        };

        Population{
            city: city,
            pop_size: pop_size,
            generation: individuals,
            mutation_rate: 0.05, // weigh this by iteration/temperature
            best_result_chromosome: Vec::new(),
            best_result_fitness: 0
        }


    }

    fn spawn_generation_zero(city_map: &'a Graph, pop_size: usize, max_threads: usize) -> Result<Vec<Individual>, String>{

        /*
        It's fast.
         */

        let city_map = Arc::new(city_map.clone());

        //input channel = job queue
        let (task_tx, task_rx) = mpsc::channel::<()>();
        let task_rx = Arc::new(Mutex::new(task_rx));

        //output channel
        let (res_tx, res_rx) = mpsc::channel::<Result<Individual, String>>();

        let mut handles = Vec::with_capacity(max_threads);

        for _ in 0..max_threads{
            let task_rx = Arc::clone(&task_rx);
            let res_tx = res_tx.clone();
            let city_map = Arc::clone(&city_map);

            // create a thread
            let handle = thread::spawn(move ||{
                loop {
                    // worker loop
                    let job = {
                        let rx_lock = task_rx.lock().unwrap();
                        rx_lock.recv()
                    };
                    match job {
                        Ok(()) => {
                            let out = Individual::new(&city_map);
                            res_tx.send(out).expect("res_tx failed to send");
                        },
                        Err(_) => {
                            // no jobs remaining/shit went south => break
                            break;
                        }
                    }
                }
            });
            handles.push(handle)
        };

        // enqueue pop_size jobs
        for _ in 0..pop_size{
            task_tx.send(()).expect("task_tx failed to send");
        }
        drop(task_tx);

        //preallocate a vector
        let mut results = Vec::with_capacity(pop_size);

        for _ in 0..pop_size{
            match res_rx.recv().expect("res_rx failed on receive"){
                Ok(indiv) => results.push(indiv),
                Err(E) => eprintln!("Generation error: {}", E),
            }
        };

        for h in handles{
            h.join().expect("Thread panicked!")
        }

        Ok(results)
    }

    fn calculate_fitness(&self, chromosome_to_calculate:&mut Vec<usize>) -> Result<usize, String>{

        /*
        calculate fitness of an individual
        it receives the idx of the individual and then just converts
        its path into energy
         */

        let city = &self.city;

        if let (Some(&first), Some(&last)) = (chromosome_to_calculate.first(), chromosome_to_calculate.last()){
            if first != last {
                chromosome_to_calculate.push(first)
            }
        }

        let fitness:usize = chromosome_to_calculate
            .windows(2)
            .map(|a| {
                match city.get_edge_weight(a[0], a[1]){
                    Ok(weight) => weight,
                    Err(E) => panic!("Got this: {E}")
                }
            })
            .sum();


        Ok(fitness)

    }

    pub fn tournament_selection(&mut self, elitism_factor: f64, num_chosen: usize) -> Vec<Individual>{

        // elitism factor is a further regularization to take the
        // survival of the fittest to overdrive. Reduces the 200 population to
        // some ~20-ish individuals

        // sort the individuals by fitness in descending order
        // well, the fitness is in ascending order, but the
        // individual's viability is in ascending order, ***the lower, the better***!!!

        // tournament selection also handles saving the best results found so far for the final result

        // i just need to peek the individuals, not clone them, so I'm getting a vector of pointers here
        let sorted_indivs: &Vec<Individual> = &self.generation
            .clone()
            .into_iter()
            .sorted_by(|a, b| a.fitness.cmp(&b.fitness))
            .collect();

        let best_fit = sorted_indivs.first().map(|x| x.fitness).unwrap_or(0) as f64;
        let worst_fit = sorted_indivs.last().map(|x|x.fitness).unwrap_or(0) as f64;

        self.best_result_fitness = best_fit as usize;
        self.best_result_chromosome = sorted_indivs[0].clone().chromosome;


        // the higher the weight, the more fit the individual is in the selection
        // yes, this is just a minmax scaler

        let selection_weights: Vec<f64> = sorted_indivs
            .iter()
            .map(|indiv|{
                (worst_fit - indiv.fitness as f64) / (worst_fit - best_fit)
            }).collect();


        let mut parents:Vec<Individual> = Vec::new();
        let mut last_chosen_parent = 0;
        for (idx, indiv) in sorted_indivs.iter().enumerate(){
            last_chosen_parent = idx;
            let sel_proba = fastrand::f64();
            if selection_weights[idx] > sel_proba && parents.len() != num_chosen{
                parents.push(indiv.clone())
            } else {break;}
        }

        //okay, this crashes the computer
        for indiv in sorted_indivs.iter().skip(last_chosen_parent) {
            let darwin = fastrand::f64();
            if darwin > elitism_factor && parents.len() != num_chosen{
                parents.push(indiv.clone())
            }
            if parents.len() == num_chosen {break;}
        }

        //println!("Selected parents: {:?}", parents);
        parents

    }

    fn make_segments(len:usize, n_segments:usize) -> Vec<(usize, usize)>{
        // returns a vector of indices - the chunks that I'll later draw from chromosomes
        // helper function
        let base = len / n_segments;
        let rem = len - base * n_segments;

        let mut offsets = Vec::with_capacity(n_segments + 1);
        offsets.push(0);

        for i in 0..n_segments{
            let extra = if i < rem {1} else {0};
            let next = offsets[i] + base + extra;

            offsets.push(next);
        };

        let segments = offsets.windows(2).map(|w| (w[0], w[1])).collect::<Vec<(usize, usize)>>();
        segments

    }

    pub fn crossover(&self, parents: &[Individual])-> Vec<Individual>{

        // TODO: figure out something not absolutely brain dead to do the corssovers between parents

        // generate all parent combinations - doesn't take into account the possible equality
        let parent_pairs: Vec<_> = parents
            .iter()
            .combinations(2)
            .filter_map(|x| if x[0] != x[1]{
                Some((x[0].clone(), x[1].clone()))
            }else{
                None
            })
            .collect();

        // each generation lets only 20 parents live, but that doesn't mean we'd get 190 parent pairs always.
        // so populating new generations isn't always as straight forward as it might seem

        let mut new_generation: Vec<Individual> = Vec::with_capacity(self.pop_size);

        for pair in parent_pairs.iter().cloned(){
            let num_segments = 4; // how many segments I'd like to split the chromosome into

            let (longer_chromosome, shorter_chromosome) = if pair.0.chromosome.len() >= pair.1.chromosome.len() {
                (&pair.0.chromosome, &pair.1.chromosome)
            } else {
                (&pair.1.chromosome, &pair.0.chromosome)
            };

            let mut child_chromosome = Vec::with_capacity(longer_chromosome.len().max(shorter_chromosome.len()));
            let segments_longer_parent = Self::make_segments(longer_chromosome.len(), num_segments);
            let segments_shorter_parent = Self::make_segments(shorter_chromosome.len(), num_segments);


            for i in 0..num_segments {
                let use_longer = fastrand::bool();
                let (start, end) = if use_longer {
                    segments_longer_parent[i]
                } else {
                    segments_shorter_parent[i]
                };

                child_chromosome.extend_from_slice(
                    if use_longer {
                        &longer_chromosome[start..end]
                    } else {
                        &shorter_chromosome[start..end]
                    }
                );
            }
            child_chromosome.dedup();
            let closing_and_fitness = match self.calculate_fitness(&mut child_chromosome) {
                Ok(res) => res,
                Err(E) => panic!("Error in child generation: {E}")
            };
            let child = Individual {
                chromosome: child_chromosome,
                fitness: closing_and_fitness
            };
            new_generation.push(child)
        }
        new_generation
    }

    fn mutate(&mut self, mut individual: Individual) -> Individual{

        // chooses a random index in the chromosome and performs one of three actions
        // either swaps two indices, removes ranodm index, or inserts a random index

        let randix = fastrand::usize(0..individual.chromosome.len());
        let choice = fastrand::usize(0..3);

        let mut new_chromosome:Vec<usize> = individual.chromosome.clone();

        match choice{
            0 => {
                let a = fastrand::usize(0..individual.chromosome.len());
                let b = fastrand::usize(0..individual.chromosome.len());

                new_chromosome.swap(a, b)

            },
            1 => {
                //inserts a random city to the path
                let randinsert = fastrand::usize(0..self.city.size);
                if fastrand::f64() < self.mutation_rate + 0.02 {
                    //coin toss for insertion
                    new_chromosome.insert(randix, randinsert)
                } else {
                    //just shuffle the child with very slight shuffle frac
                    fisher_yates_variable(&mut new_chromosome, 0.01);
                }
            },
            2 => {
                if fastrand::f64() < self.mutation_rate {
                    new_chromosome.remove(randix);
                } else {
                    fisher_yates_variable(&mut new_chromosome, 0.01);
                }
            },
            _ => {
                unreachable!()
            },
        }

        new_chromosome.dedup();
        let fitness = match self.calculate_fitness(&mut new_chromosome){
            Ok(res) => res,
            Err(E) => panic!("Error: {E}")
        };

        individual.chromosome = new_chromosome;
        individual.fitness = fitness;

        individual

    }

    pub fn new_generation(&mut self, selected_parents:&mut Vec<Individual>) -> Result<(), String>{
        // calls the crossovers for the parents and starts to populate new generation
        // not the most elegant code I've written, but as long as it works

        // TODO: figure out if it would be logical to have the parents survive onto the next generation
        //       depending on their fitness. Maybe populate the entire generation and then, depending
        //       on the relative fitness of the parents compared to new generation, make them cross over?


        let required_indices: HashSet<usize> = (0..self.city.size).collect();

        // reproduce and mutate children
        let mut new_generation: Vec<Individual> = Vec::with_capacity(self.pop_size);

        let mut offspring = self.crossover(&selected_parents);

        for child in offspring {

                let mut newchild = self.mutate(child);

                let child_indices: HashSet<usize> = newchild.chromosome.clone().into_iter().collect();

                // validation check
                if child_indices != required_indices{

                    let missing: Vec<usize> = required_indices.difference(&child_indices).cloned().collect::<Vec<usize>>();

                    for &val in &missing {
                        let randidx = fastrand::usize(0..newchild.chromosome.len());
                        newchild.chromosome.insert(randidx, val);
                    }

                }

                // very unsightly way of returning the closed path chromosome along with it's fitness
                let mut closed_and_calcd = match self.calculate_fitness(&mut newchild.chromosome){
                    Ok(res) => res,
                    Err(E) => panic!("err: {E}")
                };

                // removing illegal self-loops, dedup maintains the validity of the path by only
                // removing one of the repeating indices, as long as they repeat right after each other
                newchild.chromosome.dedup();

                // due to some Rust internals, I need to construct a whole ass new individual object
                let final_child = Individual{
                    chromosome: newchild.chromosome,
                    fitness: closed_and_calcd
                };

                new_generation.push(final_child);

        }

        if &new_generation.len() <= &self.pop_size{

            let pop_diff = &self.pop_size - &new_generation.len();
            if pop_diff < selected_parents.len(){

                fisher_yates_variable(selected_parents, self.mutation_rate);

                for parent in selected_parents.iter().take(pop_diff){
                    new_generation.push(parent.clone())
                }

            } else {

                let diff = pop_diff - &selected_parents.len();
                fisher_yates_variable(selected_parents, self.mutation_rate);

                for parent in selected_parents.iter().cloned(){
                    new_generation.push(parent);
                }

                for _ in 0..diff{
                    let new_indiv = Individual::new(&self.city)?;
                    new_generation.push(new_indiv);
                }

            }

        } else {
            new_generation.truncate(self.pop_size);
        }

        // resetting the generation field to valid children
        self.generation = new_generation;

        Ok(())
    }
}
