use ndarray::{Array2, Axis};
use rand::prelude::*;
use std::collections::HashMap;
use std::fmt;
use fastrand;
use itertools::Itertools;
use log::Level::Warn;
use ndarray::parallel::prelude::*;
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use serde_json;

#[derive(Debug, Clone)]
pub struct Vertex
{
    // equivalent to the column labels in pandas
    pub id: usize,
    pub name: String,
}
/*
======================== TODOS =========================

TODO: allow for non-usize edge weights => that was a
      needless rabbit hole, I'll jsut go with casting
      back and forth, since it's O(1) anyway.
========================================================
*/

#[derive(Debug, Clone)]
pub struct Graph
{
    pub vertices: Vec<Vertex>,
    pub name_to_id: HashMap<String, usize>,
    pub adjacency_matrix: Array2<usize>,
    pub size: usize,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug)]
struct GraphData {
    number_of_cities: usize,
    edge_list: Vec<[usize; 3]>
}

impl Vertex {
    pub fn new(id: usize, name:&String) -> Self{
        
        let name_unwrapped = name.clone();

        Vertex
        {
            id: id,
            name: name_unwrapped,
        }

    }
}

impl Graph{
    // equivalent of __init__ in python
    // Core constructor for the graph class
    pub fn new(num_vertices: usize, density:f32) -> Result<Self, String>{

        const MAX: usize = 800;
        //preallocations and scoping
        let size: usize= num_vertices;
        let num_connections_max = (size * (size - 1)) / 2;
        let cons_to_use = (density * num_connections_max as f32) as usize;

        let adjacency_matrix:Array2<usize> = Array2::zeros((size, size));

        let mut possible_edges:Vec<(usize, usize)> = Self::precalculate_all_possible_edges(size);
        let vertex_names: Vec<String> = Self::generate_vertex_names(num_vertices);

        let vertices: Vec<Vertex> = vertex_names.iter()
            .enumerate()
            .map(|(id, name)| Vertex::new(id, name))
            .collect();

        let name_to_id = vertices.iter()
            .map(|v| (v.name.clone(), v.id))
            .collect();

        // initialize an empty graph first
        let mut graph =  Graph {
            vertices,
            name_to_id,
            adjacency_matrix,
            size
        };

        // assign edges based on the number of connections to use and weigh them

        //parital fisher-yates shuffle

        let len_of_edges = possible_edges.len();
        for idx in (1..len_of_edges-1){
            let j = fastrand::usize(0..len_of_edges);
            possible_edges.swap(idx, j)
        }


        let edge_list: Vec<(usize, usize, usize)> = possible_edges
            .par_iter()
            .take(cons_to_use)
            .map(|&(i,j)| {
                let weight = fastrand::usize(0..MAX);
                (i, j, weight)
            })
            .collect();

        for (i, j, weight) in edge_list{
            graph.make_an_edge_weighed(i,j,weight)?
        }

        graph.kill_orphans()?;
        Ok(graph)
    }

    pub fn from_json(json_path:&str) -> Result<Self, String>{
        
        // a method to make the graph from pre-defined data

        let mut file = File::open(json_path)
            .map_err(|e| format!("Failed to open the json file: {}",e))?;
        
        let mut json_string: String = String::new();
        file.read_to_string(&mut json_string)
        .map_err(|e| format!("Failed to read the JSON file: {}", e))?;
        
        let graph_data: GraphData = serde_json::from_str(&json_string)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;
        
        let num_vertices = graph_data.number_of_cities;
        let adjacency_matrix = Array2::zeros((num_vertices, num_vertices));
        let vertex_names = Self::generate_vertex_names(num_vertices);

        let vertices: Vec<Vertex> = vertex_names.iter()
            .enumerate()
            .map(|(id, name)| Vertex::new(id, name))
            .collect();

        let name_to_id: HashMap<String, usize> = vertices.iter()
            .map(|v| (v.name.clone(), v.id))
            .collect();

        let mut graph = Graph{
            vertices,
            name_to_id,
            adjacency_matrix,
            size: num_vertices,
        }; 

        for edge in graph_data.edge_list{
            let from = edge[0];
            let to = edge[1];
            let weight = edge[2];

            graph.make_an_edge_weighed(from, to, weight);
        };
        
        Ok(graph)
    }


    fn generate_vertex_names(num_vertices: usize) -> Vec<String>{

        let mut vertex_names: Vec<String> = Vec::new();
        for i in 0..num_vertices{
            vertex_names.push(String::from(format!("city{}", i)))
        };
        vertex_names
    }

    fn precalculate_all_possible_edges(size:usize) -> Vec<(usize, usize)>{

        // okay, this is *very* fast now
        let mut possible_edges: Vec<(usize, usize)> = (0..size)
            .tuple_combinations()
            .par_bridge()
            .collect();

        possible_edges
    }

    fn kill_orphans(&mut self) -> Result<(), String>{

        // check the adjacency matrix for completely isolated nodes
        // it iterates over all columns, if any column's checksum is zero, it means
        // that the vertex is an orphan
        // if you hit an orphan, out a new edge on a random index of the column
        // currently parallelized using the rayon + ndarra::parallel combo

        let orphans: Vec<usize> = self
            .adjacency_matrix
            .axis_iter(Axis(1))
            .into_par_iter()
            .enumerate()
            .filter_map(|(i, col)| if col.sum() == 0 {Some(i)} else {None})
            .collect();


        for i in orphans{
            let mut rand_index:usize = fastrand::usize(0..self.size-1);
            if rand_index >= i {
                rand_index += 1;
            }
            self.make_an_edge(i, rand_index)?;
        }

        Ok(())
    }


    pub fn make_an_edge(&mut self, a:usize, b:usize) -> Result<(), String>{
        if a == b
        {
            return Err(String::from("Location indices cannot equal each other"))
        }
        if a >= self.size || b >= self.size
        {
            return Err(String::from("Indices are out of bounds of the graph shape"))
        }

        //symmetrically update the adjacency matrix
        self.adjacency_matrix[[a,b]] = 1;
        self.adjacency_matrix[[b,a]] = 1;

        Ok(())
    }

    pub fn make_an_edge_weighed(&mut self, a:usize, b:usize, weight:usize) -> Result<(), String>{
        if a == b
        {
            return Err(String::from("Location indices cannot equal each other"))
        }
        if a >= self.size || b >= self.size
        {
            return Err(String::from("Indices are out of bounds of the graph shape"))
        }

        //symmetrically update the adjacency matrix
        self.adjacency_matrix[[a,b]] = weight;
        self.adjacency_matrix[[b,a]] = weight;

        Ok(())
    }

    pub fn remove_an_edge(&mut self, a:usize, b:usize) -> Result<(), String>{
    // remove the specified edge, if there's a need
        if a == b
        {
            return Err(String::from("Location indices cannot equal each other"))
        }
        if a >= self.size || b >= self.size
        {
            return Err(String::from("Indices are out of bounds of the graph shape"))
        }

        self.adjacency_matrix[[a,b]] = 0;
        self.adjacency_matrix[[b,a]] = 0;

        Ok(())
    }

    pub fn set_edge_weight(&mut self, a:usize, b:usize, weight:usize) -> Result<(), String>{

        if a == b
        {
            return Err(String::from("Location indices cannot equal each other"))
        }
        if a >= self.size || b >= self.size
        {
            return Err(String::from("Indices are out of bounds of the graph shape"))
        }

        self.adjacency_matrix[[a,b]] = weight;
        self.adjacency_matrix[[b,a]] = weight;

        Ok(())


    }

    pub fn get_edge_weight(&self, a:usize, b:usize) -> Result<usize, String>{

        if a == b
        {
            return Err(String::from("Location indices cannot equal each other"))
        }
        if a >= self.size || b >= self.size
        {
            return Err(String::from("Indices are out of bounds of the graph shape"))
        }

        // also figure out how to allow an "invalid" edge as long as there are the correct indices,
        // just add a massive penalty

        let weight = self.adjacency_matrix.column(a)[b];
        match weight {
            // this is kind of redundant considering that I use a valid path generation. Oh, well
            0 => {
                //println!("TELEPORT");
                Ok(10000)
            },
            _ => {
                //println!("ok");
                Ok(weight)
            }
        }
    }
    pub fn get_neighbors(&self, vertex_id: usize) -> Result<Vec<((usize, usize), usize)>, String>{

      //generate a list of neighbors for the vertex
      //takes the column of the matrix (vertex_id corresponding to the column) and checks
      //all the edges (position in the column from top) - if there is a nonzero number at that
      //position, then the neighbor pair is (vertex_id, column_id)
      //it also return the weight along with the edge that is outgoing from the particular
      //vertex

        // currently costly asf, since it allocates a fresh vec for each call. That was probably also why it was crashing the program

        if vertex_id > self.vertices.len()
        {
            return  Err(String::from(format!("The vertex id is out of bounds for a graph of size{}", self.vertices.len())))
        }


        let column = self.adjacency_matrix.column(vertex_id);
        // Vec<((vertex_id, neighbor_id)), edge_weight)>
        let neighboring_vertices: Vec<((usize, usize), usize)> = column
            .iter()
            .enumerate()
            .par_bridge()
            .filter_map(|(j, &weight)|{
                if weight != 0 {
                    Some(((vertex_id, j), weight))
                } else {
                    None
                }
            })
            .collect();

        Ok(neighboring_vertices)
    }

    pub fn wipe_config(&mut self) {
        // completely wipe the adjacency matrix, just in case
        let adjacency_matrix = Array2::zeros((self.size, self.size));
        self.adjacency_matrix = adjacency_matrix
    }

}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result{
        let header = self.vertices.iter()
            .map(|v| format!("{:8}", v.name))
            .collect::<Vec<String>>()
            .join(" ");

        writeln!(f, "{:8} {}", "", header)?;

        for (i, row) in self.adjacency_matrix.rows().into_iter().enumerate(){
            let row_label = &self.vertices[i].name;
            let cells = row.iter()
                .map(|&val| format!("{:8}", val))
                .collect::<Vec<String>>()
                .join(" ");

            writeln!(f, "{:8} {}", row_label, cells)?;

        }
        Ok(())
    }
}