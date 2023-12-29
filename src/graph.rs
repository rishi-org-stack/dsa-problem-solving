use std::{
    cmp::Ordering,
    cmp::{self, min},
    collections::{BTreeSet, HashMap, VecDeque},
    process::Child,
    vec,
};

#[derive(Debug)]
pub struct Graph {
    node_count: u32,
    edge_count: u32,
    matix: HashMap<u32, Vec<u32>>,
    directed: bool,
}

impl Graph {
    pub fn new(list: Vec<Vec<u32>>, directed: bool) -> Graph {
        let mut matrix: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut node_count = 0;
        for node in list.iter() {
            matrix
                .entry(node[0])
                .and_modify(|v| {
                    node_count += 1;
                    v.push(node[1])
                })
                .or_insert(vec![node[1]]);
            if !directed {
                matrix
                    .entry(node[1])
                    .and_modify(|v| {
                        node_count += 1;
                        v.push(node[0])
                    })
                    .or_insert(vec![node[0]]);
            }
        }
        Graph {
            edge_count: 2 * list.len() as u32,
            node_count,
            matix: matrix,
            directed,
        }
    }

    pub fn traverse_bfs(self) {
        let mut visited_map: HashMap<u32, bool> = HashMap::new();
        for node_edge in self.matix {
            if visited_map.get(&node_edge.0) != Some(&true) {
                visited_map.insert(node_edge.0, true);
                let mut q: VecDeque<&u32> = VecDeque::new();
                q.push_back(&node_edge.0);

                loop {
                    if q.is_empty() {
                        break;
                    }
                    let data = q.pop_front().unwrap();
                    println!("node: {}", *data);
                    for node in &node_edge.1 {
                        if visited_map.get(node) != Some(&true) {
                            q.push_back(node);
                            visited_map.insert(*node, true);
                        }
                    }
                }
            }
        }
    }

    pub fn traverse_dfs(self) {
        let mut visited_map: HashMap<u32, bool> = HashMap::new();
        for node_edge in self.matix {
            if visited_map.get(&node_edge.0) != Some(&true) {
                visited_map.insert(node_edge.0, true);
                let mut q: VecDeque<&u32> = VecDeque::new();
                q.push_back(&node_edge.0);

                loop {
                    if q.is_empty() {
                        break;
                    }
                    let data = q.pop_back().unwrap();
                    println!("node: {}", *data);

                    for node in &node_edge.1 {
                        if visited_map.get(node) != Some(&true) {
                            q.push_back(node);
                            visited_map.insert(*node, true);
                        }
                    }
                }
            }
        }
    }
}

pub fn shortest_path(
    matrix: &HashMap<u32, Vec<u32>>,
    size: u32,
    starting: u32,
    ending: u32,
) -> u32 {
    let mut q: VecDeque<u32> = VecDeque::new();
    q.push_back(starting);

    let mut dist: Vec<u32> = vec![u32::MAX; (size + 1) as usize];
    dist[starting as usize] = 1;
    loop {
        if q.is_empty() {
            break;
        }

        let node: u32 = q.pop_front().unwrap();
        println!("node {}", node);
        if !matrix[&node].is_empty() {
            for child in &matrix[&node] {
                if dist[*child as usize] >= u32::MAX {
                    q.push_back(*child);
                    dist[*child as usize] = dist[node as usize] + 1;
                }
            }
        }
    }
    println!("{:?}", dist);

    let mut c = dist[ending as usize] - 1;
    let mut dir: Vec<u32> = vec![ending];
    let mut prev = ending;
    loop {
        if c == 0 {
            break;
        }

        for child in &matrix[&prev] {
            if dist[*child as usize] == dist[prev as usize] - 1 {
                c -= 1;
                prev = *child;
                dir.push(prev);
            }
        }
    }
    println!("{:?}", dir);
    dist[ending as usize]
}

pub fn detect_cycle(matrix: &HashMap<u32, Vec<u32>>, size: u32, starting: u32) -> bool {
    let mut q: VecDeque<Vec<u32>> = VecDeque::new();
    q.push_back(vec![starting, 0]);

    let mut visited: Vec<bool> = vec![false; (size + 1) as usize];

    loop {
        if q.is_empty() {
            break;
        }

        let node_parent: Vec<u32> = q.pop_front().unwrap();
        for child in &matrix[&node_parent[0]] {
            if *child != node_parent[1] && visited[*child as usize] {
                return true;
            }
            if *child != node_parent[1] {
                visited[*child as usize] = true;
                q.push_back(vec![*child, node_parent[0]]);
            }
        }
    }

    false
}
pub fn can_complete(matrix: &HashMap<u32, Vec<u32>>, size: u32, starting: u32) -> bool {
    let mut q: VecDeque<u32> = VecDeque::new();
    q.push_back(starting);

    let mut visited: Vec<bool> = vec![false; (size + 1) as usize];

    loop {
        if q.is_empty() {
            return true;
        }

        let current_node: u32 = q.pop_front().unwrap();

        if matrix.contains_key(&current_node) {
            for child_node in &matrix[&current_node] {
                if visited[*child_node as usize] {
                    return false;
                }

                visited[*child_node as usize] = true;
                q.push_back(*child_node);
            }
        }
    }
}
//TODO: fix it broken
pub fn can_complete_p(matrix: &HashMap<u32, Vec<u32>>, size: u32, starting: u32) -> Vec<u32> {
    let mut q: VecDeque<u32> = VecDeque::new();
    q.push_back(starting);

    let mut visited: Vec<bool> = vec![false; (size + 1) as usize];
    let mut ans: Vec<u32> = vec![];
    loop {
        if q.is_empty() {
            return ans;
        }

        let current_node: u32 = q.pop_front().unwrap();
        ans.push(current_node);
        if matrix.contains_key(&current_node) {
            for child_node in &matrix[&current_node] {
                if visited[*child_node as usize] {
                    return vec![];
                }

                visited[*child_node as usize] = true;
                q.push_back(*child_node);
            }
        }
    }
}

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
pub fn dijskatra(
    matrix: &HashMap<u32, Vec<EdgeWeight>>,
    size: u32,
    ending: u32,
    starting: EdgeWeight,
) -> u32 {
    let starting_clone = starting.clone();
    let mut q: BTreeSet<EdgeWeight> = BTreeSet::new();
    q.insert(starting);

    let mut dist: Vec<u32> = vec![u32::MAX; (size + 1) as usize];

    dist[starting_clone.0 as usize] = 0;
    loop {
        if q.is_empty() {
            break;
        }

        println!("{:?}", dist);
        let node: EdgeWeight = q.pop_first().unwrap();
        println!("start : node {:?}", node);
        if !matrix[&node.0].is_empty() {
            for child in &matrix[&node.0] {
                if dist[child.0 as usize] > (dist[node.0 as usize] + child.1) {
                    let child_clone: EdgeWeight = child.clone();
                    q.insert(child_clone);
                    dist[child.0 as usize] =
                        min(dist[node.0 as usize] + child.1, dist[child.0 as usize]);
                }
            }
        }
    }
    dist[ending as usize]
}

pub fn detect_cycle_directed_dfs(
    matrix: &HashMap<u32, Vec<u32>>,
    visted: &mut Vec<bool>,
    starting: u32,
    size: u32,
) -> bool {
    let mut s: VecDeque<u32> = VecDeque::new();
    s.push_back(starting);
    let mut pv: Vec<bool> = vec![false; (size + 1) as usize];
    loop {
        if s.is_empty() {
            break;
        }

        let node: u32 = s.pop_back().unwrap();
        visted[node as usize] = true;
        pv[node as usize] = true;

        if matrix.contains_key(&node) {
            for child in &matrix[&node] {
                if visted[*child as usize] && pv[*child as usize] {
                    return true;
                }

                if !visted[*child as usize] {
                    visted[*child as usize] = true;
                    pv[*child as usize] = true;
                    s.push_back(*child);
                }
            }
        }
    }
    false
}

#[derive(Clone, Copy)]
pub struct position {
    pub row: i32,
    pub col: i32,
}

impl position {
    pub fn normalize(&self, colsCount: i32) -> i32 {
        let result = (self.row * colsCount) + self.col + 1;
        result
    }
    pub fn from_normal(normal: i32, colsCount: i32) -> position {
        let new_normal: i32 = normal - 1;
        let x = new_normal / colsCount;
        let y = new_normal - (x * colsCount);
        position { row: x, col: y }
    }
}

pub fn labyrinth(
    matrix: &Vec<Vec<&str>>,
    start: position,
    end: position,
    lengthX: i32,
    lengthY: i32,
    visited: &mut HashMap<i32, bool>,
) -> i32 {
    let mut q: VecDeque<position> = VecDeque::new();
    q.push_back(start);

    let mut dist: Vec<i32> = vec![i32::MAX; ((lengthX * lengthY) + 1) as usize];
    dist[start.normalize(lengthY) as usize] = 0;

    loop {
        if q.is_empty() {
            break;
        }

        let lastPos: position = q.pop_front().unwrap();

        if lastPos.row + 1 < lengthX
            && matrix[(lastPos.row + 1) as usize][(lastPos.col) as usize] != "#"
        {
            let newPos: position = position {
                row: lastPos.row + 1,
                col: lastPos.col,
            };
            if !visited.contains_key(&newPos.normalize(lengthY)) {
                q.push_back(newPos);
                let normal = newPos.normalize(lengthY);
                visited.insert(normal, true);
                let lastNormal = lastPos.normalize(lengthY);
                if dist[normal as usize] > dist[lastNormal as usize] + 1 {
                    dist[normal as usize] = dist[lastNormal as usize] + 1;
                }
            }
        }

        if lastPos.col + 1 < lengthY
            && matrix[(lastPos.row) as usize][(lastPos.col + 1) as usize] != "#"
        {
            let newPos: position = position {
                row: lastPos.row,
                col: lastPos.col + 1,
            };
            if !visited.contains_key(&newPos.normalize(lengthY)) {
                q.push_back(newPos);
                let normal = newPos.normalize(lengthY);
                visited.insert(normal, true);
                let lastNormal = lastPos.normalize(lengthY);
                if dist[normal as usize] > dist[lastNormal as usize] + 1 {
                    dist[normal as usize] = dist[lastNormal as usize] + 1;
                }
            }
        }

        if lastPos.row - 1 >= 0 && matrix[(lastPos.row - 1) as usize][(lastPos.col) as usize] != "#"
        {
            let newPos: position = position {
                row: lastPos.row - 1,
                col: lastPos.col,
            };
            if !visited.contains_key(&newPos.normalize(lengthY)) {
                q.push_back(newPos);
                let normal = newPos.normalize(lengthY);
                visited.insert(normal, true);
                let lastNormal = lastPos.normalize(lengthY);
                if dist[normal as usize] > dist[lastNormal as usize] + 1 {
                    dist[normal as usize] = dist[lastNormal as usize] + 1;
                }
            }
        }

        if lastPos.col - 1 >= 0 && matrix[(lastPos.row) as usize][(lastPos.col - 1) as usize] != "#"
        {
            let newPos: position = position {
                row: lastPos.row,
                col: lastPos.col - 1,
            };
            if !visited.contains_key(&newPos.normalize(lengthY)) {
                q.push_back(newPos);
                let normal = newPos.normalize(lengthY);
                visited.insert(normal, true);
                let lastNormal = lastPos.normalize(lengthY);
                if dist[normal as usize] > dist[lastNormal as usize] + 1 {
                    dist[normal as usize] = dist[lastNormal as usize] + 1;
                }
            }
        }
    }
    println!("dist {:?}", dist);
    println!("end {:?}", end.normalize(lengthY));
    return dist[end.normalize(lengthY) as usize];
}

pub fn building_road(matrix: &HashMap<u32, Vec<u32>>, visted: &mut Vec<bool>, starting: u32) {
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(starting);

    loop {
        if queue.is_empty() {
            break;
        }
        let last: u32 = queue.pop_front().unwrap();
        visted[last as usize] = true;

        if matrix.contains_key(&last) {
            for child in &matrix[&last] {
                if !visted[*child as usize] {
                    visted[*child as usize] = true;
                    queue.push_back(*child);
                }
            }
        }
    }
}

pub fn message_route(
    matrix: &HashMap<u32, Vec<u32>>,
    starting: u32,
    target: u32,
    size: u32,
) -> u32 {
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(starting);
    let mut dist: Vec<u32> = vec![u32::MAX; (size + 1) as usize];
    dist[starting as usize] = 1;
    loop {
        if queue.is_empty() {
            break;
        }
        let last: u32 = queue.pop_front().unwrap();

        if matrix.contains_key(&last) {
            for child in &matrix[&last] {
                if dist[*child as usize] > dist[last as usize] + 1 {
                    dist[*child as usize] = dist[last as usize] + 1;
                    queue.push_back(*child);
                }
            }
        }
    }

    dist[target as usize]
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
                    dist[child.0 as usize] =
                        min(dist[node.0 as usize] + child.1, dist[child.0 as usize]);
                }
            }
        }
    }

    for i in 1..=size {
        println!("{}", dist[i as usize])
    }
}
