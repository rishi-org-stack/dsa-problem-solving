use std::cmp::{min, Ordering};
use std::collections::{BTreeSet, HashMap};
use std::io::stdin;

#[derive(Debug, Eq)]
pub struct EdgeWeight(u32, u32);
impl EdgeWeight {
    pub fn new(a: u32, b: u32) -> Self {
        EdgeWeight(a, b)
    }
}
impl Ord for EdgeWeight {
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.cmp(&other.1)
    }
}

impl PartialOrd for EdgeWeight {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for EdgeWeight {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}
impl Clone for EdgeWeight {
    fn clone(&self) -> Self {
        EdgeWeight(self.0, self.1)
    }
}

pub fn dijskatra_shortest_path(
    matrix: &HashMap<u32, Vec<EdgeWeight>>,
    size: u32,
    ending: u32,
    starting: EdgeWeight,
) {
    let starting_clone = starting.clone();
    let mut q: BTreeSet<EdgeWeight> = BTreeSet::new();
    q.insert(starting);

    let mut dist: Vec<u32> = vec![u32::MAX; (size + 1) as usize];

    dist[starting_clone.0 as usize] = 0;
    loop {
        if q.is_empty() {
            break;
        }

        let node: EdgeWeight = q.pop_first().unwrap();
        if !matrix[&node.0].is_empty() {
            for child in &matrix[&node.0] {
                if dist[child.0 as usize] > (dist[node.0 as usize] + child.1) {
                    let child_clone: EdgeWeight = child.clone();
                    q.insert(child_clone);
                    dist[child.0 as usize] = dist[node.0 as usize] + child.1;
                }
            }
        }
    }

    for i in 1..=size {
        println!("{}", dist[i as usize])
    }
}

fn main() {
    let mut nm_str = String::new();
    stdin().read_line(&mut nm_str).unwrap();

    let n_m: Vec<u32> = nm_str
        .split_whitespace()
        .map(|f| f.parse::<u32>().unwrap())
        .collect();

    let mut matrix: HashMap<u32, Vec<EdgeWeight>> = HashMap::new();

    let n: u32 = n_m[0];
    let m: u32 = n_m[1];

    for _ in 0..m {
        let mut ab_str = String::new();
        stdin().read_line(&mut ab_str).unwrap();

        let a_b: Vec<u32> = ab_str
            .split_whitespace()
            .map(|f| f.parse::<u32>().unwrap())
            .collect();

        matrix
            .entry(a_b[0])
            .and_modify(|v| v.push(EdgeWeight::new(a_b[1], a_b[2])))
            .or_insert(vec![EdgeWeight::new(a_b[1], a_b[2])]);
        matrix
            .entry(a_b[1])
            .and_modify(|v| v.push(EdgeWeight::new(a_b[0], a_b[2])))
            .or_insert(vec![EdgeWeight::new(a_b[0], a_b[2])]);
    }

    dijskatra_shortest_path(&matrix, n, n, EdgeWeight::new(1, 0));
}
