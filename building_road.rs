use std::collections::{HashMap, VecDeque};
use std::io::stdin;
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

    let mut visited: Vec<bool> = vec![false; (n + 1) as usize];
    building_road(&matrix, &mut visited, 1);

    let mut c = 0;
    let mut result: Vec<Vec<u32>> = Vec::new();
    for i in 1..=n {
        if !visited[i as usize] {
            c += 1;
            result.push(vec![i - 1, i]);
            building_road(&matrix, &mut visited, i)
        }
    }

    println!("{}", c);
    for it in result {
        println!("{} {}", it[0], it[1]);
    }
}
