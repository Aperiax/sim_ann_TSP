use std::collections::HashSet;
use crate::graph::Graph;
#[derive(Debug, Clone)]
pub struct Individual {
    pub chromosome: Vec<usize>,
    pub fitness: usize,
}
impl Ord for Individual {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fitness.cmp(&other.fitness)
    }
}
impl PartialOrd for Individual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Individual {
    fn eq(&self, other: &Self) -> bool {
        self.chromosome == other.chromosome
    }
}
impl Eq for Individual {}

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

impl Individual {
    pub fn new(city: &Graph) -> Result<Self, String>{

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

            // shuffle the neighbor lists - make it random, very random
            fisher_yates_variable(&mut neighbor_list, fastrand::f64());

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