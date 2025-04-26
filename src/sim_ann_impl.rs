use crate::graph::Graph;
use std::collections::{HashSet, VecDeque};
use std::hash::Hash;
use fastrand;
use rayon::prelude::*;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use itertools::Itertools;
/*
========================= todos =========================

TODO: make the city insertions/removals not retarded
TODO: Add subpath optimizations
TODO: don't forget to take out profanities
TODO: pralelize the genetic algo by hand, no rayon.

==========================================================
*/

const KB: f64 = 1.;
const T_MAX: f64 = 1000.;
const T_MIN: f64 = 10.;

pub struct Sim {
    // the general simulation struct
    pub path: Vec<usize>,
    temp: f64,
    pub energy: usize,
    plateau_count: u8,
    city_map: Graph,
    city_size: usize,
    energy_monitor: EnergyMonitor,
    best_path: Vec<usize>,
    best_en: usize
}

pub struct EnergyMonitor {

    // an energy management struct for Sim. Sim remembers only the current iteration's energy.
    // Energy monitor on the other hand remembers last N iterations to detect plateaus and
    // signals Sim to do stuff if needed via MonitorAction

    memory: VecDeque<usize>,
    best_seen: f64,
    worst_seen: f64,
    its_since_best: usize,
    plateau_count: usize,
    memory_window_size: usize,
    relative_change_iter: f64,
    memory_range: f64,
}


#[derive(Debug)]
pub struct Population {
    // I'll most likely not need the pop_size?
    city: Graph,
    pop_size: usize,
    pub generation: Vec<Individual>,
    mutation_rate: f64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Individual {
    chromosome: Vec<usize>,
    fitness: usize,
}

pub enum MonitorAction{

    // action enum that the run_check() of energy monitor emits,
    // signalling all guarding actions that the monitor tells the sim to do in case of
    // if getting stuck
    Nothing,
    PertrubAndIncrease {
        shuffle_frac: f64,
        delta: f64,
    },
    Update_best_path {
        // if this pops out a true, it is going to make the simulation run automatically
        // update the self.best_path and best_en to the current iteration's energy and path
        update: bool
    }
}

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

pub fn mean(iterable: &VecDeque<usize>) -> f64{

    let sum: f64 = iterable.into_iter().map(|&val| val as f64).sum();

    let mean = sum / iterable.len() as f64;
    mean
}

pub fn stdev(iterable: &VecDeque<usize>, mean: f64) -> f64 {
    let sum_diff: f64 = iterable.into_iter().map(|&val| (val as f64 - mean).powf(2.)).sum();
    let result: f64 = (sum_diff / ((iterable.len() - 1) as f64)).sqrt();
    result
}

pub fn zscore(en_delta:f64, stdev:f64, mean:f64) -> f64 {

    let z_t: f64 = (en_delta - mean) / (stdev + 1e-6).sqrt();
    z_t

}


impl Sim {

    // simulation constructor
    pub fn new(city_map: Graph, starting_temp: f64) -> Self{

        //let initial_path:Vec<(usize, usize)> = Self::generate_tour(city_map);
        let city_size = city_map.size.clone();
        let init_path = Self::generate_tour(&city_map).expect("An error occurred in pathfinding");

        Sim
        {
            path:init_path.0.clone(),
            temp:starting_temp,
            energy:init_path.1.clone(),
            plateau_count:0,
            city_map,
            city_size,
            energy_monitor: EnergyMonitor::new(init_path.1, 5),
            best_path: init_path.0,
            best_en: init_path.1
        }
    }

    pub fn generate_tour(city_map: &Graph) -> Result<(Vec<usize>, usize), String>{
        //Generates the initial solution - helper for init_sim()
        // pathfinding setup
        println!("Starting greedy tour generation");
        let mut idx = 0;
        let mut path_energy: usize = 0;
        let size = city_map.size.clone();

        let indices: HashSet<usize> = (0..size).collect();
        let mut seen_vertices: HashSet<usize> = HashSet::new();

        let mut backtrack_stack = Vec::new();
        let mut path:Vec<usize> = Vec::new();

        seen_vertices.insert(idx);
        path.push(idx);


        while indices != seen_vertices {

            let mut neighbor_list = city_map.get_neighbors(idx)?;

            let mut next_index = None;

            if neighbor_list.len() > 1 {
                for (neighbor, _) in neighbor_list {
                    if !seen_vertices.contains(&neighbor.1) {
                        // I'll handle constructing the path into edge coordinates later
                        // just get the index for now
                        next_index = Some(neighbor.1);
                        break;
                    }
                }
            } else if neighbor_list.len() == 1{
                next_index = Some(neighbor_list[0].0.1);
            }

            match next_index {
                Some(next) => {
                    path_energy += city_map.get_edge_weight(idx, next)?;
                    seen_vertices.insert(next);
                    backtrack_stack.push(next);
                    path.push(next);
                    idx = next;
                }
                None => {
                    // the problem isn't the backtracking, but rather the graph generation
                    // in and of itself
                    if let Some(previous) = backtrack_stack.pop(){
                        idx = previous;
                    } else {
                        break;
                    }
                }
            }

        }

        if path.last() != path.first() {
            path.push(match path.first(){
                Some(&first) => first,
                None => 0
            })
        }

        Ok((path, path_energy))
    }

    fn temp_scheduler(&mut self, t_prev:f64, nsteps: usize) -> Result<(), String>{
        // Gradually cools the system
        // lundy & mees cooling scheme


        let beta = (T_MAX-T_MIN) / (nsteps as f64*T_MAX*T_MIN);
        let t_new = t_prev / (1. + beta * t_prev);

        self.temp = t_new;
        Ok(())
    }

    pub fn path_to_edges(&self) -> Vec<&[usize]>{
        //yank the path out

        let mut edges = Vec::new();
        for window in self.path.windows(2){
            edges.push(window);
        }
        edges

    }

    fn close_path(&mut self) -> Result<(), String>{

        let first = match self.path.first(){
            Some(& first) => first,
            None => panic!("Something went horribly wrong with the path vector")
        };

        let last = match self.path.last(){
            Some(&last) => last,
            None => panic!("Something went horribly wrong with the path vector")
        };

        if first != last {
            self.path.push(first);
            self.energy = self.calculate_energy()?;
        }
        Ok(())
    }

    fn perturb_config(&mut self, shuffle_param:f64, mode:&str) -> Result<(), String>{

        let size = self.path.len().to_owned();

        match mode {
            "randmut" => {

                let decision = fastrand::f64();
                // 1% chance to add a vertex onto the path
                if decision < 0.01
                {
                    let randinsert = fastrand::usize(0..size);
                    self.path.insert(randinsert, fastrand::usize(0..size));
                    fisher_yates_variable(&mut self.path, shuffle_param);
                }
                else {
                    fisher_yates_variable(&mut self.path, shuffle_param);
                }

            }
            _ => {
                   fisher_yates_variable(&mut self.path, shuffle_param);
            }
        }


        Ok(())
    }

    fn validate_config(&self) -> Result<bool, String>{
    //Checks if all vertices are represented in the path

        // iter() - iterates references, into_iter() - consumes the references into owned items
        let represented:HashSet<_> = self.path.clone().into_iter().collect();
        let indices: HashSet<_> = (0..self.city_size).collect();

        if represented == indices{
            Ok(true)
        } else {
            Ok(false)
        }

    }

    pub fn calculate_energy(&self) -> Result<usize, &str>{

        //Calculates the path energy, literally sum edge weights for every road represented
        // path is the order of vertices that have been added into the list, I just need to look up
        // the vertices

        let mut en_total:usize = 0;
        let path_edges = self.path_to_edges();

        for edge in path_edges{

            let a = *edge.get(0).ok_or("Edge missing or invalid")?;
            let b = *edge.get(1).ok_or("Edge missing or invalid")?;

            let weight = self.city_map.get_edge_weight(a, b)
                .unwrap_or(10000);
            en_total += weight
        }
        Ok(en_total)

    }

    pub fn run(&mut self, nsteps: usize) -> Result<(Vec<usize>, usize), String>{
    //Core sim runner
        let mut iter = 1;
        // the current solution is going to get updated, it's currently a pointer undercover to
        // the path field of sim struct
        let mut retries: usize = 0;
        let mut en = self.energy.clone();
        let size_init = self.path.len().clone();
        let mut max_iter = nsteps;

        println!("********WELCOME********");
        println!("Initiating simulated annealing run");
        println!("Initial solution \"energy\": {en}");
        println!("Simulation was set up to run for: {nsteps} steps");


        while retries < 3{

            while iter < max_iter {


                /*

                 ==== GENERAL SETUP ====

                */

                let mut sim_temp = self.temp.clone();
                self.close_path()?;
                // okay, start off with cloning whatever I would need to use for later rejecting/accepting

                let pre_perturb_config = self.path.clone();
                let pre_perturb_energy = self.energy.clone() as f64;


                // perturb config at the start of each iteration step - has the chance to append new cities
                self.perturb_config(0.1, "normal")?;

                // balances out the possibilites of randmut randomly slapping a new city in, or magicking out
                // a vertex out of nowhere
                if self.path.len() > size_init {
                    let proba_rem = fastrand::f64();
                    if proba_rem > 0.1 {
                        let randix = fastrand::usize(1..self.path.len()-1);

                        let vertex_to_remove = self.path[randix];
                        let count = self.path.iter().filter(|&&c|c==vertex_to_remove).count();

                        if count > 1 {
                           self.path.remove(randix);
                        };

                    };
                };

                /*

                ==== NESTED OPTIMIZATIONS ====

                */

                if iter % 20 == 0 && self.path.len() > 5 {
                    let segment_size = (self.path.len() / 5).max(3).min(10);
                    let start_idx = fastrand::usize(0..self.path.len() - segment_size);

                    let end_idx = start_idx + segment_size;

                    self.optimize_subpath(start_idx, end_idx)?;
                }

                if iter % 50 == 0 {
                    self.optimize_shortcuts()?;
                }

                // save the best path encountered, since it can happen literally anywhere

                let mut iter_energy = self.calculate_energy()? as f64;
                let mut iter_config = self.path.clone();

                // set decision variables
                match self.energy_monitor.update(iter_energy as usize, iter)? {
                    MonitorAction::PertrubAndIncrease {shuffle_frac, delta} => {
                        self.perturb_config(shuffle_frac, "normal")?;
                        self.temp += delta;
                    },
                    MonitorAction::Update_best_path{update} => {
                        println!("New optimim discovered!");
                        self.best_path = iter_config.clone();
                        self.best_en = iter_energy as usize;
                    }
                    MonitorAction::Nothing => {
                        // just does nothing
                    }
                }



                // ==== probability related vars ====
                let delta_en = iter_energy - pre_perturb_energy;

                let mmry_mean = mean(&self.energy_monitor.memory);
                let stdev = stdev(&self.energy_monitor.memory, mmry_mean);
                let z_t = zscore(delta_en, stdev, mmry_mean);

                // this is rather weird, but it's the only thing that actually gives me results that are anywhere
                // near normal results. Essentially if autoaccept doesn't happen, it has ~ 50% chance of being accepted
                let sigmoid_output_proba: f64 = 1. / (1. + (-z_t/self.energy_monitor.memory_range).exp());
                let acceptance: f64 = fastrand::f64();




                /*

                 ==== UPDATE ====

                */


                if delta_en > 0. && self.energy_monitor.relative_change_iter > 1. {
                    // this would mean we've most likely taken an illegal turn, since this condition
                    // is satisfied only when the change of energy is massive.
                    self.path = pre_perturb_config;
                    self.energy = pre_perturb_energy as usize;

                    if !self.energy_monitor.memory.is_empty() {
                        // forget the rejected config's energy as well as the config.
                        // the simulation is dependent on the memory window, but trying to keep
                        // even the horrendous energies encountered would then steer the simulation
                        // to very weird rabbit holes and destabilize it
                        self.energy_monitor.memory.pop_front();
                    }

                } else {

                    if delta_en < 0. || sigmoid_output_proba > acceptance {
                        self.path = iter_config;
                        self.energy = iter_energy as usize;


                    } else {

                        self.path = pre_perturb_config;
                        self.energy = pre_perturb_energy as usize;

                        if !self.energy_monitor.memory.is_empty() {
                            // forget the rejected config's energy as well as the config.
                            self.energy_monitor.memory.pop_front();
                        }

                    }

                }

                self.temp_scheduler(sim_temp, nsteps)?;

                // ==== SUMMARY PRINTS ====

                if iter % 10000 == 0{
                    println!("===== ITERATION {iter} =====");
                    println!("Iteration energy: {iter_energy}, iteration delta_e {delta_en}");
                    println!("Iteration proba: {sigmoid_output_proba}, acceptance rolled: {acceptance}");
                    println!("Best seen energy: {}, Worst seen energy: {}", self.energy_monitor.best_seen, self.energy_monitor.worst_seen);
                    println!("Current temperature: {}", self.temp)
                }

                iter += 1;
            }

            println!("====MAIN RUN OVER====");
            if self.best_en >= en {

                let restart_steps:usize = (max_iter + 100) / 5;

                println!("Simulation failed to discover an optimum, restarting run");

                self.temp += 1000.;
                self.perturb_config(1., "randmut")?;

                if retries < 3 {
                    println!("==========Setting temperature to: {}==========", self.temp);
                    println!("==========Rerunning with {restart_steps} steps==========");
                } else {
                    println!("Max retries reached, stopping run.")
                }
                retries += 1;
                iter = 0;
            } else {
                break;
            }


        }

        if en < self.best_en {
            println!("Simulation failed to reduce path energy")
        } else {
            println!("Optimization succesful!")
        }
        println!("{:?}", self.best_path);
        Ok((self.best_path.clone(), self.best_en.clone()))
    }

    fn nested_sim_ann(&self, start_city:usize, end_city:usize, must_visit: &HashSet<usize>) -> Result<Vec<usize>, String>{

        // miniature simulation annealing running for the subpaths

        let subproblem_size = must_visit.len();
        let iterations = (subproblem_size * 20).max(100);

        // not using the temp scheduler here, since that is tied to the sim struct and i don't want to
        // mess the main sim ann run
        let cooling_rate = 0.95;

        let mut current_path = vec![start_city];
        current_path.extend(must_visit.iter().cloned());
        current_path.push(end_city);

        let mut current_energy = self.calculate_path_energy_on_slices(&current_path)?;

        let mut init_temp = 1000.;
        let mut temp = init_temp;

        let mut best_path = current_path.clone();
        let mut best_energy = current_energy;

        for _ in 0..iterations{

            let mut neighbor_path = current_path.clone();
            self.perturb_subpath(&mut neighbor_path, must_visit)?;

            let neighbor_energy = self.calculate_path_energy_on_slices(&neighbor_path)?;

            let accept = if neighbor_energy < current_energy {
                true
            } else {
                let delta_e = (neighbor_energy - current_energy) as f64;
                let proba = (-delta_e / temp).exp();

                fastrand::f64() < proba
            };

            if accept{
                current_path = neighbor_path;
                current_energy = neighbor_energy;

                if current_energy < best_energy {
                    best_energy = current_energy;
                    best_path = current_path.clone();
                }
            }


            temp *= cooling_rate;
        }
        Ok(best_path)
    }

    fn optimize_shortcuts(&mut self) -> Result<(), String>{

        let mut improved = true;
        let mut total_savings = 0;


        while improved {
            improved = false;
            for i in 0..self.path.len() - 2{

                // calculate for triples
                let city_1 = self.path[i];
                let city_2 = self.path[i+1];
                let city_3 = self.path[i+2];

                if city_1 == city_2 || city_2 == city_3 {
                    continue;
                }

                let current_cost = match (self.city_map.get_edge_weight(city_1, city_2), self.city_map.get_edge_weight(city_2, city_3)){
                    (Ok(cost1), Ok(cost2)) => cost1 + cost2,
                    _ => continue
                };


                let shortcut_cost = match self.city_map.get_edge_weight(city_1, city_3){
                    Ok(cost) => cost,
                    _ => continue
                };

                if shortcut_cost < current_cost {
                    let count_2:usize = self.path.iter().filter(|&&c| c == city_2).count();

                    // take the shortcut only if all cities were already visited, or we've already
                    // passed the middle city more than once
                    let all_cities_visited = match self.validate_config() {
                        Ok(valid) => valid && count_2 > 1,
                        _ => false
                    };

                    if all_cities_visited{
                        self.path.remove(i+1);
                        total_savings += current_cost - shortcut_cost;
                        improved = true;
                        break;
                    }

                }


            }

        }
        if total_savings > 0 {
            self.energy = self.calculate_energy()?;
        }

        Ok(())
    }

    fn optimize_subpath (&mut self, start_idx:usize, end_idx:usize) -> Result<(), String>{

        if start_idx  >= self.path.len() || end_idx >= self.path.len() || start_idx >= end_idx{
            // silent return check to ignore invalid indices
            return Ok(())
        }

        let start_city = self.path[start_idx];
        let end_city = self.path[end_idx];

        // hashset of indices
        let cities_to_visit: HashSet<usize> = self.path[start_idx+1..end_idx].iter().cloned().collect();
        if cities_to_visit.len() <= 1 {
            return  Ok(())
        }

        let mut relevant_cities = cities_to_visit.clone();
        relevant_cities.insert(start_city);
        relevant_cities.insert(end_city);

        let optimised_path = self.nested_sim_ann(start_city, end_city, &cities_to_visit)?;
        let opt_len = optimised_path.len().clone();
        self.path.splice(start_idx+1..end_idx, optimised_path.into_iter().skip(1).take(opt_len-2));

        self.energy = self.calculate_energy()?;



        Ok(())
    }

    fn calculate_path_energy_on_slices(&self, path: &[usize]) -> Result<usize, String>{

        let mut en_total:usize = 0;

        for edge in path.windows(2){

            let a = *edge.get(0).ok_or("Edge missing or invalid")?;
            let b = *edge.get(1).ok_or("Edge missing or invalid")?;

            let weight = self.city_map.get_edge_weight(a, b)
                .unwrap_or(10000);
            en_total += weight
        }
        Ok(en_total)

    }

    fn perturb_subpath(&self, path: &mut Vec<usize>, must_visit: &HashSet<usize>) -> Result<(), String>{

        let mut visited = HashSet::new();
        visited.insert(path[0]);

        let operation = fastrand::usize(0..3);
        match operation {
            0 => {
                let idx1 = fastrand::usize(1..path.len()-1);
                let idx2= fastrand::usize(1..path.len()-1);
                path.swap(idx1, idx2);
            },
            1 => {
                let insrt_pos = fastrand::usize(1..path.len()-1);
                let new_city = if fastrand::f64() < 0.7 && !must_visit.is_empty(){
                    let missing: Vec<_> = must_visit.iter().cloned().collect();
                    missing[fastrand::usize(0..missing.len())]
                } else {
                    fastrand::usize(0..self.city_size)
                };

                path.insert(insrt_pos, new_city);
            },
            2 => {
                // rem random if it's not the only occurence
                if path.len() > 3 {
                    let remove_pos = fastrand::usize(1..path.len()-1);
                    let city_to_remove = path[remove_pos];

                    //occurences of the city to remove
                    let occurences = path.iter().filter(|&&c| c == city_to_remove).count();
                    if !must_visit.contains(&city_to_remove) || occurences > 1 {
                        path.remove(remove_pos);
                    }
                }
            },
            _ => unreachable!()
        }

        for &required in must_visit {
            if !path.contains(&required){
                let inser_pos = fastrand::usize(1..path.len());
                path.insert(inser_pos, required)
            }
        }

        Ok(())
    }
}

impl EnergyMonitor {

    //Monitors the system's energy; if a plateau is reached, it calls permute path since it's most
    //likely stuck in a local minimum and bumps up the temperature a bit to let it explore around
    // the problem is that when I'd call this stuff, it would not remember the past energy
    // meaning I'd have to implement a separate energy monitor struct
    // matches and modifies the field of Energy_monitor

    fn new(starting_energy: usize, memory_window: usize) -> Self{

        // initialize the starting memory with the first energy we've seen
        let mut energy:VecDeque<usize> = VecDeque::with_capacity(memory_window);

        energy.push_back(starting_energy);

        EnergyMonitor
        {
            best_seen:starting_energy as f64,
            worst_seen:starting_energy as f64,
            memory: energy,
            its_since_best:0,
            plateau_count:0,
            memory_window_size: memory_window,
            relative_change_iter: 0.,
            memory_range: 0.
        }
    }

    fn push_energy(&mut self, iteration_energy: usize) -> Result<(), String>{
        // just updates the memory queue
        self.memory.push_back(iteration_energy);
        Ok(())
    }

    fn detect_plateaus(&mut self, iteration: usize) -> bool{

        /// checks the memory vecdeque, if at least three values haven't really changed much,
        /// it calls signal_perturb adn signal_temp_increase, which, in turn, give a go ahead
        /// to the Sim struct to modify its fields

        // some arbitrary minimal change, tweakable. Doing relative ranges due to
        // the fact that energy can change wildly iteration to iteration, so sliding minmax regularization
        // seemed like a good idea.

        const THRESHOLD: f64 = 0.1;

        // the memory range should take in the worst and best seen values in the current memory window,
        // otherwise it's not really useful

        let memory_max = match self.memory.iter().max(){
            Some(&max) => max as f64,
            None => 0.,
        };

        let memory_min = match self.memory.iter().min(){
            Some(&min) => min as f64,
            None => 0.
        };


        let memory_range: f64 = memory_max - memory_min;
        let mmry_mean: f64 = mean(&self.memory);

        // I need to somehow calculate the plateau check from the relative change

        let relative_change: f64 = memory_range / mmry_mean;
        self.relative_change_iter = relative_change;
        self.memory_range = memory_range;


        if relative_change < THRESHOLD{
            self.plateau_count += 1
        }

        if self.plateau_count > 5 && iteration > 10 {
            // that means we're most likely in a plateau
            self.plateau_count = 0;
            true
        } else {

            false
        }


    }

    fn calc_shuffle_factor(&self,  plateau_check: bool) -> Result<f64, String>{

        // this is probably a bit misleading, but should act as a guard on top of the normal perturbations
        // that are going to be happening during the simulation. If the system is heading to some
        // stagnant minimum, this should just add another layer of perturbations on top to give a bigger
        // bump


        // dynamically rescale the factor depending on the seen energies
        let factor = (self.memory[0] as f64 - self.best_seen) / (self.worst_seen - self.best_seen);

        if plateau_check {

             // if a plateau was detected, immediately permute as many times as possible -> probably
             // too drastic, tweak later
             // this way we can do a temperature-dependent fraction for the shuffling function

             Ok(factor)

        } else {

            // and if the plateau wasn't flagged at all and the counter is zero, then do nothing
            // the zero is just a dummy that isn't going to be used

            Ok(0.)
        }

    }

    fn update(&mut self, iter_energy: usize, iteration: usize) -> Result<MonitorAction, String>{
        // handles the comprehensive plateau handling and detection. Probably overengineering,
        // but then again, maybe not.

        // push the current iteration's energy into the memory deque
        let iter_energy_flt: f64 = iter_energy as f64;
        let mut update_path: bool = false;

        if iter_energy_flt < self.best_seen{
            self.best_seen = iter_energy_flt;
            update_path = true;
        };

        if iter_energy_flt > self.worst_seen{
            self.worst_seen = iter_energy_flt
        }

        // drop the thingies out of range for the init mem_window => keeping the memory a real rolling
        // window
        self.memory.push_front(iter_energy);

        if self.memory.len() > self.memory_window_size{
            self.memory.pop_back();
        };


        let plateau_check = self.detect_plateaus(iteration);

        let action:MonitorAction = if plateau_check {
            MonitorAction::PertrubAndIncrease {
                shuffle_frac: self.calc_shuffle_factor(plateau_check)?,
                delta: 100.,
            }
        } else if update_path{
            MonitorAction::Update_best_path {
                update: true
            }
        } else{
            MonitorAction::Nothing
        };
        Ok(action)
    }
}

impl Individual {
    fn new(city: &Graph) -> Result<Self, String>{

        //wraping the individual in a result so that I can paralelize generating the population

        let generation = match Self::generate_individual(city){
            Ok(generation) => generation,
            Err(E) => panic!("Error: {E}"),
        };

        let individual = Individual{
            chromosome: generation.0,
            fitness: generation.1
        };

        Ok(individual)
    }

    pub fn generate_individual(city_map: &Graph) -> Result<(Vec<usize>, usize), String> {
        let mut fitness: usize = 0;
        let size = city_map.size.clone();
        let mut idx = fastrand::usize(0..size);

        let indices: HashSet<usize> = (0..size).collect();
        let mut seen_vertices: HashSet<usize> = HashSet::new();

        let mut backtrack_stack = Vec::new();
        let mut chromosome: Vec<usize> = Vec::new();

        seen_vertices.insert(idx);
        chromosome.push(idx);

        // TODO: make this tad better by sorting the neighbor list in ascending order
        // that way it should technically preferably take the shortest path always


        while indices != seen_vertices {
            let mut neighbor_list = city_map.get_neighbors(idx)?;

            let mut next_index = None;

            if neighbor_list.len() > 1 {
                for (neighbor, _) in neighbor_list {
                    if !seen_vertices.contains(&neighbor.1) {
                        next_index = Some(neighbor.1);
                        break;
                    }
                }
            } else if neighbor_list.len() == 1 {
                next_index = Some(neighbor_list[0].0.1);
            }

            match next_index {
                Some(next) => {
                    fitness += city_map.get_edge_weight(idx, next)?;
                    seen_vertices.insert(next);
                    backtrack_stack.push(next);
                    chromosome.push(next);
                    idx = next;
                }
                None => {
                    if let Some(previous) = backtrack_stack.pop() {
                        idx = previous;
                    } else {
                        break;
                    }
                }
            }
        };

        if chromosome.last() != chromosome.first() {

            let first  = match chromosome.first().clone(){
                Some(&first) => first,
                None => panic!("Something's very wrong")
            };

            let last  = match chromosome.last().clone(){
                Some(&last) => last,
                None => panic!("Something's very wrong")
            };

            let fitness_delta: usize = city_map.get_edge_weight(first, last)?;
            // close the path
            chromosome.push(first);
            fitness += fitness_delta

        }

        Ok((chromosome, fitness))
    }
}

impl Population {
    pub fn new(pop_size:usize, city: Graph) -> Self{

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
            mutation_rate: 0.05
        }


    }

    fn spawn_generation_zero(city_map: &Graph, pop_size: usize, max_threads: usize) -> Result<Vec<Individual>, String>{

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

    fn calculate_fitness(&mut self, chromosome:Vec<usize>) -> Result<(Vec<usize>, usize), String>{

        /*
        calculate fitness of an individual
        it receives the idx of the individual and then just converts
        its path into energy
         */

        let city = &self.city;
        let mut chromosome_to_calculate = chromosome.clone();

        //closes the path if it was not so at the start
        if chromosome_to_calculate.last() != chromosome_to_calculate.first(){
            chromosome_to_calculate.push(*chromosome_to_calculate.first().unwrap());
        }

        let fitness:usize = chromosome_to_calculate
            .windows(2)
            .par_bridge()
            .map(|a| {
                let weight = match city.get_edge_weight(*a.get(0).unwrap(), *a.get(1).unwrap()){
                    Ok(weight) => weight,
                    Err(E) => panic!("Something's wrong, I can feel it")
                };
                weight
            })
            .sum();


        Ok((chromosome_to_calculate, fitness))

    }

    pub fn tournament_selection(&mut self, elitism_factor: f64) -> Vec<Individual>{

        // elitism factor is a further regularization to take the
        // survival of the fittest to overdrive. Reduces the 200 population to
        // some ~20-ish individuals

        // sort the individuals by fitness in descending order
        // well, the fitness is in ascending order, but the
        // individual's viability is in descening

        let sorted_indivs: Vec<Individual> = self.generation
            .clone()
            .into_iter()
            .sorted_by(|a, b| a.fitness.cmp(&b.fitness))
            .collect();


        let best_fit = sorted_indivs.first().map(|x| x.fitness).unwrap_or(0) as f64;
        let worst_fit = sorted_indivs.last().map(|x|x.fitness).unwrap_or(0) as f64;
        println!("worst fitness: {worst_fit}");

        // the higher the weight, the more fit the individual is in the selection
        // yes, this is just a minmax scaler

        let selection_weights: Vec<f64> = sorted_indivs
            .iter()
            .map(|indiv|{
                (worst_fit - indiv.fitness as f64) / (worst_fit - best_fit)
            }).collect();


        let mut parents:Vec<Individual> = Vec::new();

        for (idx, indiv) in sorted_indivs.iter().enumerate(){

            let sel_proba = fastrand::f64() + elitism_factor;
            if selection_weights[idx] > sel_proba{
                parents.push(indiv.clone())
            }
        }

        //println!("Selected parents: {:?}", parents);
        println!("Num selected: {}", parents.len());
        parents


    }

    fn crossover(&self, parents: Vec<Individual>){

        // TODO: figure out something not absolutely brain dead to do the corssovers between parents

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
                if fastrand::f64() < 0.5 {
                    //coin toss for insertion
                    new_chromosome.insert(randix, randinsert)
                } else {
                    //just shuffle the child with very slight shuffle frac
                    fisher_yates_variable(&mut new_chromosome, 0.01);
                }
            },
            2 => {
                if fastrand::f64() < 0.5 {
                    new_chromosome.remove(randix);
                } else {
                    fisher_yates_variable(&mut new_chromosome, 0.01);
                }
            },
            _ => {
                unreachable!()
            },
        }

        let (valid_chromosome, fitness) = match self.calculate_fitness(new_chromosome){
            Ok(res) => res,
            Err(E) => panic!("Error: {E}")
        };

        individual.chromosome = valid_chromosome;
        individual.fitness = fitness;

        individual

    }

    fn reproduce_parents(){
        // calls the crossovers for the parents and starts to populate new generation
        // TODO: figure out if it would be logical to have the parents survive onto the next generation
        //       depending on their fitness. Maybe populate the entire generation and then, depending
        //       on the relative fitness of the parents compared to new generation, make them cross over?

    }

    fn run(){

    }

}
