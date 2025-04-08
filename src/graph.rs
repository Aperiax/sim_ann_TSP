use ndarray::Array2;
use rand::prelude::*;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Vertex
{
    // equivalent to the column labels in pandas
    pub id: usize,
    pub name: String,
}

// TODO: allow for non-usize edge weights
pub struct Graph
{
    pub vertices: Vec<Vertex>,
    pub name_to_id: HashMap<String, usize>,
    pub adjacency_matrix: Array2<usize>,
    pub size: usize,
}

impl Vertex {
    pub fn new(id: usize, name:String) -> Self{

        Vertex
        {
            id,
            name,
        }

    }
}

impl Graph{
    // equivalent of __init__ in python
    // Core constructor for the graph class
    pub fn new(vertex_names:Vec<String>, density:f32) -> Result<Self, String>{

        let mut rng = rand::rng();
        let size: usize= vertex_names.len();
        let num_connections_max = (size * (size - 1)) / 2;
        let cons_to_use = (density * num_connections_max as f32) as i32;
        let adjacency_matrix:Array2<usize> = Array2::zeros((size, size));

        let mut possible_edges:Vec<(usize, usize)> = Self::precalculate_all_possible_edges(size);


        let vertices: Vec<Vertex> = vertex_names.iter()
            .enumerate()
            .map(|(id, name)| Vertex{
                id,
                name: name.clone(),
            })
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
        // pre-shuffle the edges
        possible_edges.shuffle(&mut rng);

        for edge in possible_edges.into_iter().take(cons_to_use as usize){

            let edge_weight = rng.random_range(0..200);
            graph.make_an_edge(edge.0, edge.1)?;
            graph.kill_orphans()?;
            graph.set_edge_weight(edge.0, edge.1, edge_weight)?;

        }

        Ok(graph)
    }

    fn precalculate_all_possible_edges(size:usize) -> Vec<(usize, usize)>{

        let mut possible_edges: Vec<(usize, usize)> = Vec::new();

        for i in 0..size{
            for j in (i+1)..size{
                possible_edges.push((i, j));
            }
        }

        // fun fact, borrow checker and compiler check the last thing that was executed
        // in the function for type inference, so it spazes out without the possible_edges at the
        // end

        possible_edges
    }

    fn kill_orphans(&mut self) -> Result<(), String>{

        // check the adjacency matrix for completely isolated nodes
        // it iterates over all columns, if any column's checksum is zero, it means
        // that the vertex is an orphan
        // if you hit an orphan, out a new edge on a random index of the column
        let mut rng = rand::rng();

        let orphans: Vec<usize> = self
            .adjacency_matrix
            .columns()
            .into_iter()
            .enumerate()
            .filter_map(|(i, col)| if col.sum() == 0 {Some(i)} else {None})
            .collect();

        for i in orphans{
            let mut rand_index:usize = rng.random_range(0..self.size - 1);
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

        let weight = self.adjacency_matrix.column(a)[b];

        Ok(weight)
    }
    pub fn get_neighbors(&self, vertex_id: usize) -> Result<Vec<((usize, usize), usize)>, String>{

      //generate a list of neighbors for the vertex
      //takes the column of the matrix (vertex_id corresponding to the column) and checks
      //all the edges (position in the column from top) - if there is a nonzero number at that
      //position, then the neighbor pair is (vertex_id, column_id)
      //it also return the weight along with the edge that is outgoing from the particular
      //vertex

        if vertex_id > self.vertices.len()
        {
            return  Err(String::from(format!("The vertex id is out of bounds for a graph of size{}", self.vertices.len())))
        }


        let column_view = self.adjacency_matrix.column(vertex_id);
        // Vec<((vertex_id, neighbor_id)), edge_weight)>
        let neighboring_vertices: Vec<((usize, usize), usize)> = column_view.into_iter()
            .enumerate()
            .filter(|&x| *x.1 != 0)
            .map(|x| ((vertex_id, x.0), *x.1))
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