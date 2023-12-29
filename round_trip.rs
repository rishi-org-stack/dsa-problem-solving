use std::collections::{HashMap, VecDeque};
use std::io::stdin;
fn round_trip(
    matrix: &HashMap<u32, Vec<u32>>,
    start: u32,
    stack: &mut VecDeque<Vec<u32>>,
    path: &mut Vec<u32>,
    size: u32,
) -> (u32, bool) {
    let mut c: u32 = 1;
    let mut possible: bool = false;
    let mut local_visited: Vec<bool> = vec![false; (size + 1) as usize];
    loop {
        if stack.is_empty() {
            break;
        }

        let parent_node: Vec<u32> = stack.pop_back().unwrap();
        local_visited[parent_node[1] as usize] = true;

        if matrix.contains_key(&parent_node[1]) {
            for child in &matrix[&parent_node[1]] {
                if *child != parent_node[0] && local_visited[*child as usize] && *child == start {
                    c += 1;
                    possible = true;
                    path.push(*child);
                    return (c, possible);
                } else {
                    if *child != parent_node[0] && !local_visited[*child as usize] {
                        local_visited[*child as usize] = true;
                        stack.push_back(vec![parent_node[1], *child]);
                        path.push(*child);
                        c += 1;
                        break;
                    }
                }
            }
        }
    }
    (c, possible)
}
fn main() {
    let mut nm_str = String::new();
    stdin().read_line(&mut nm_str).unwrap();

    let n_m: Vec<u32> = nm_str
        .split_whitespace()
        .map(|f| f.parse::<u32>().unwrap())
        .collect();

    let mut matrix: HashMap<u32, Vec<u32>> = HashMap::new();

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
            .and_modify(|v| v.push(a_b[1]))
            .or_insert(vec![a_b[1]]);
        matrix
            .entry(a_b[1])
            .and_modify(|v| v.push(a_b[0]))
            .or_insert(vec![a_b[0]]);
    }
    // println!("matrix: {:?}", matrix);

    for i in 1..=n {
        let mut stack: VecDeque<Vec<u32>> = VecDeque::new();
        stack.push_back(vec![0, i]);
        let mut path: Vec<u32> = vec![i];
        let possible = round_trip(&matrix, i, &mut stack, &mut path, n);
        if possible.1 {
            println!("{}", possible.0);
            for idx in 0..possible.0 {
                println!("{}", path[idx as usize]);
            }
            return;
        }
    }

    println!("IMPOSSIBLE");
}
