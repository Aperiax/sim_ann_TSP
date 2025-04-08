use crate::graph::Graph;
use rand::prelude::*;
use std::collections::HashSet;

pub struct Sim {
    path: Vec<usize>,
    temp: f64,
    energy: usize,
    plateau_count: u8,
    city_size: usize
}

impl Sim {

    // simulation constructor
     pub fn new(starting_temp: f64) -> Self{

        const MAX:i32 = 100; //any other number would be an absolute overkill

        let mut rng = rand::rng();
        let nums: Vec<i32> = (1..MAX).collect();
        let number= nums.choose(&mut rng);
        let mut cities: Vec<String> = Vec::new();

        //set the cities (vertices)
        for i in 0..*number.unwrap(){
            cities.push(format!("City{}", i))
        }

        // set up an initial graph
        let city_map:Graph = Graph::new(cities, 0.6).expect("An error occurred
        in graph generation");

        //let initial_path:Vec<(usize, usize)> = Self::generate_tour(city_map);
        // placeholder
        let city_size = city_map.size.clone();
        let init_path = Self::generate_tour(city_map).expect("An error occurred in pathfinding");

        Sim
        {
            path:init_path.0,
            temp:starting_temp,
            energy:init_path.1,
            plateau_count:0,
            city_size
        }

    }

    fn generate_tour(city_map: Graph) -> Result<(Vec<usize>, usize), String>{

        //Generates the initial solution - helper for init_sim()
        // pathfinding setup

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
            neighbor_list.sort_by(|a,b| a.1.cmp(&b.1));

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
                    path.push(next);
                    idx = next;
                    backtrack_stack.push(next);
                }
                None => {
                    if let Some(previous) = backtrack_stack.pop(){
                        idx = previous;
                    } else {
                        break;
                    }
                }
            }

        }

        Ok((path, path_energy))
    }

    fn temp_scheduler(){
    //Gradually cools the system

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

    fn calculate_energy(){
    //Calculates the path energy, literally sum edge weights for every road represented
    }

    fn energy_monitor(){
    //Monitors the system's energy; if a plateau is reached, it calls permute path since it's most
    //likely stuck in a local minimum
    }

    fn permute_path(){
    //Shuffles the path/subset of it depending on a percentage passed,
    //increases the systems temperature marginally
    }

    pub fn run(){
    //Core sim runner
    }
}
