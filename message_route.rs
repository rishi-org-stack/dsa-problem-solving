use std::collections::{HashMap, VecDeque};
use std::io::stdin;
pub fn message_route(matrix: &HashMap<u32, Vec<u32>>, starting: u32, target: u32, size: u32) {
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

    let steps = dist[target as usize];

    if steps < u32::MAX {
        let mut number_of_steps = steps.clone();
        let mut it_target = target.clone();
        let mut res: VecDeque<u32> = VecDeque::new();
        res.push_front(target);
        loop {
            if number_of_steps == 1 {
                break;
            }

            if matrix.contains_key(&it_target) {
                for child in &matrix[&it_target] {
                    if dist[*child as usize] == number_of_steps - 1 {
                        number_of_steps -= 1;
                        it_target = child.clone();
                        res.push_front(*child);
                    }
                }
            }
        }
        println!("{}", steps);
        for x in res {
            println!("{}", x)
        }

        return;
    }

    println!("IMPOSSIBLE");
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
    message_route(&matrix, 1, n, n);
}
