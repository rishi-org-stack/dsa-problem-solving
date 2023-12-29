use io::stdin;
use std::collections::{vec_deque, BTreeSet, HashMap, VecDeque};
use std::io;
use std::{cmp, vec};

use graph::{building_road, position};

use crate::graph::EdgeWeight;
pub mod euler;
pub mod graph;
fn climb_stairs(n: i32) -> i32 {
    if n <= 1 {
        return 1;
    }

    let mut prev = 1;
    let mut current = 1;
    for _ in 2..=n {
        current = prev + current;
        prev = current - prev;
    }

    current
}

fn dice_combinations(n: i64) -> i64 {
    if n == 0 {
        return 1;
    }

    let mut ans: i64 = 0;
    for i in 1..=n {
        if i <= n {
            ans += dice_combinations(n - i);
        }
    }
    ans
}

const MNUM: u64 = 1e7 as u64;
fn minimizing_coin(nums: &Vec<u64>, i: usize, target: u64) -> u64 {
    if i == 0 {
        if target % nums[i] == 0 {
            return target / nums[i];
        }

        return MNUM;
    }
    let mut take = MNUM;
    let not_take = minimizing_coin(nums, i - 1, target);

    if nums[i] <= target {
        take = 1 + minimizing_coin(nums, i, target - nums[i])
    }

    return cmp::min(take, not_take);
}

fn minimizing_coinT(nums: &Vec<u64>, size: usize, target: u64) -> u64 {
    let target_range: Vec<u64> = (0..=target).collect();
    let mut prev: Vec<u64> = target_range
        .iter()
        .map(|t| {
            if t % nums[0] == 0 {
                return t / nums[0];
            }
            MNUM
        })
        .collect();
    prev[0] = 0;
    for idx in 1..=size {
        let mut curr = vec![MNUM; (1 + target) as usize];
        curr[0] = 0;
        for t in 1..=target {
            let not_take = prev[t as usize];
            let mut take = MNUM;
            if nums[idx] <= t {
                take = 1 + curr[(t - nums[idx]) as usize];
            }

            curr[t as usize] = cmp::min(take, not_take);
        }

        prev = curr;
    }

    prev[target as usize]
}
fn coin_combination2(nums: &Vec<u64>, i: usize, target: u64) -> u64 {
    if target == 0 {
        return 1;
    }
    if i == 0 {
        if target % nums[i] == 0 {
            return 1;
        }

        return 0;
    }
    let mut take = 0;
    let not_take = minimizing_coin(nums, i - 1, target);

    if nums[i] <= target {
        take = minimizing_coin(nums, i, target - nums[i])
    }

    return take + not_take;
}

fn coin_combination2_T(nums: &Vec<u64>, size: usize, target: u64) -> u64 {
    let target_range: Vec<u64> = (0..=target).collect();
    let mut prev: Vec<u64> = target_range
        .iter()
        .map(|t| {
            if t % nums[0] == 0 {
                return 1;
            }
            0
        })
        .collect();
    prev[0] = 1;
    for idx in 1..=size {
        let mut curr = vec![0; (1 + target) as usize];
        curr[0] = 1;
        for t in 1..=target {
            let not_take = prev[t as usize];
            let mut take = 0;
            if nums[idx] <= t {
                take = curr[(t - nums[idx]) as usize] as u64;
            }

            curr[t as usize] = (take + not_take) as u64;
        }

        prev = curr;
    }
    prev[target as usize]
}

fn solve(grid: &Vec<Vec<i32>>, row: usize, col: usize) -> i32 {
    if row == 0 && col == 0 {
        if grid[0][0] == 1 {
            return 0;
        }
        return 1;
    }

    if grid[row][col] == 1 {
        return 0;
    }

    let up = if row >= 1 {
        solve(grid, row - 1, col)
    } else {
        0
    };

    let left = if col >= 1 {
        solve(grid, row, col - 1)
    } else {
        0
    };

    return up + left;
}

pub fn unique_paths_with_obstacles(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let rows = obstacle_grid.len();
    let cols = obstacle_grid[0].len();

    solve(&obstacle_grid, rows - 1, cols - 1)
}

pub fn unique_paths_with_obstaclesT(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let rows = obstacle_grid.len();
    let cols = obstacle_grid[0].len();

    if obstacle_grid[rows - 1][cols - 1] == 1 || obstacle_grid[0][0] == 1 {
        return 0;
    }

    let mut prev = vec![0; cols as usize];
    prev[0] = 1;

    for idx in 1..cols {
        if obstacle_grid[0][idx] != 1 {
            prev[idx] = 1;
        }
    }

    for idx in 0..rows {
        let mut curr = vec![0; cols];

        for col in 0..cols {
            if obstacle_grid[idx][col] == 0 {
                if idx == 0 && col == 0 {
                    curr[col] = 1
                } else {
                    let up = if idx >= 1 { prev[col] } else { 0 };
                    let left = if col >= 1 { curr[col - 1] } else { 0 };
                    curr[col] = up + left;
                }
            }
        }

        prev = curr;
    }

    prev[cols - 1]
}

fn sol_min_path_sum(grid: &Vec<Vec<i32>>, row: usize, col: usize) -> i32 {
    if row == 0 && col == 0 {
        return grid[0][0];
    };

    let up = if row >= 1 {
        grid[row][col] + sol_min_path_sum(grid, row - 1, col)
    } else {
        i32::MAX
    };

    let left = if col >= 1 {
        grid[row][col] + sol_min_path_sum(grid, row, col - 1)
    } else {
        i32::MAX
    };

    cmp::min(up, left)
}

fn min_path_sum(grid: Vec<Vec<i32>>) -> i32 {
    let rows = grid.len();
    let cols = grid[0].len();
    sol_min_path_sum(&grid, rows - 1, cols - 1)
}

fn min_path_sumT(grid: Vec<Vec<i32>>) -> i32 {
    let rows = grid.len();
    let cols = grid[0].len();

    let mut prev: Vec<i32> = vec![i32::MAX; cols];

    for idx in 0..rows {
        let mut curr = vec![i32::MAX; cols];

        for col in 0..cols {
            if idx == 0 && col == 0 {
                curr[0] = grid[0][0];
            } else {
                let up = if idx >= 1 {
                    grid[idx][col] + prev[col]
                } else {
                    i32::MAX
                };

                let left = if col >= 1 {
                    grid[idx][col] + curr[col - 1]
                } else {
                    i32::MAX
                };
                curr[col] = cmp::min(up, left)
            }
        }

        prev = curr;
    }

    prev[cols - 1]
}

fn solve_min_distance(word1: &[u8], word2: &[u8], idx1: usize, idx2: usize) -> i32 {
    if idx1 == 0 || idx2 == 0 {
        if word1[idx1] == word2[idx2] {
            return 1;
        }
        return 0;
    };

    if word1[idx1] == word2[idx2] {
        return 1 + solve_min_distance(word1, word2, idx1 - 1, idx2 - 1);
    }

    let left_word_1 = if idx1 >= 1 {
        solve_min_distance(word1, word2, idx1 - 1, idx2)
    } else {
        0
    };

    let left_word_2 = if idx2 >= 1 {
        solve_min_distance(word1, word2, idx1, idx2 - 1)
    } else {
        0
    };
    return cmp::max(left_word_1, left_word_2);
}

fn min_distance(word1: String, word2: String) -> i32 {
    let len1 = word1.len();
    let len2 = word2.len();
    let ans = solve_min_distance(word1.as_bytes(), word2.as_bytes(), len1 - 1, len2 - 1);
    println!("ans: {}", ans);

    (len1 + len2) as i32 - (2 * ans)
}

fn solve_num_decodings(letters: &[u8], index: usize, size: usize) -> i32 {
    if index == size {
        return 1;
    }

    if letters[index] - 48 == 0 {
        return 0;
    }

    let two_letter =
        if index < size - 1 && (letters[index + 1] - 48) < 7 && (letters[index] - 48) > 0 {
            solve_num_decodings(letters, index + 2, size)
        } else {
            0
        };

    let one_letter = solve_num_decodings(letters, index + 1, size);

    return two_letter + one_letter;
}
fn num_decodings(s: String) -> i32 {
    let len = s.len();

    solve_num_decodings(s.as_bytes(), 0, len)
}

// fn dijskatra_setup() {
//     let stdin = stdin();
//     let n = 7;
//     let mut lines = stdin.lines();
//     let mut matrix: HashMap<u32, Vec<graph::EdgeWeight>> = HashMap::new();
//     for _ in 0..n {
//         let n_x: Vec<u32> = lines
//             .next()
//             .unwrap()
//             .unwrap()
//             .split_whitespace()
//             .map(|s| s.parse::<u32>().unwrap())
//             .collect();
//         matrix
//             .entry(n_x[0])
//             .and_modify(|v| v.push(EdgeWeight::new(n_x[1], n_x[2])))
//             .or_insert(vec![EdgeWeight::new(n_x[1], n_x[2])]);

//         matrix
//             .entry(n_x[1])
//             .and_modify(|v| v.push(EdgeWeight::new(n_x[0], n_x[2])))
//             .or_insert(vec![EdgeWeight::new(n_x[0], n_x[2])]);
//     }
//     print!("matrix: {:?}", matrix);
//     let result: u32 = graph::dijskatra(&matrix, 4, 4, EdgeWeight::new(0, 0));
//     println!("{}", result);
// }
// fn main func hai resolve it a+later
fn main_k() {
    // let stdin = io::stdin();
    // let n = 4;
    // let mut lines = stdin.lines();
    // let mut matrix: HashMap<u32, Vec<u32>> = HashMap::new();
    // for _ in 0..n {
    //     let n_x: Vec<u32> = lines
    //         .next()
    //         .unwrap()
    //         .unwrap()
    //         .split_whitespace()
    //         .map(|s| s.parse::<u32>().unwrap())
    //         .collect();
    //     matrix
    //         .entry(n_x[0])
    //         .and_modify(|v| v.push(n_x[1]))
    //         .or_insert(vec![n_x[1]]);
    // }
    // print!("matrix: {:?}", matrix);
    // let result: Vec<u32> = graph::can_complete_p(&matrix, 3, 1);
    // println!("{:?}", result);
    let mut matrix: HashMap<u32, Vec<u32>> = HashMap::new();
    // matrix.insert(1, vec![2]);
    // matrix.insert(2, vec![3]);
    // matrix.insert(3, vec![4, 7]);
    // matrix.insert(7, vec![5]);
    // matrix.insert(5, vec![6]);
    // matrix.insert(8, vec![2, 9]);
    // matrix.insert(9, vec![10]);
    // matrix.insert(10, vec![8]);
    matrix.insert(1, vec![2, 3]);
    matrix.insert(4, vec![3]);
    matrix.insert(3, vec![1]);
    let mut visited: Vec<bool> = vec![false; 5];
    for i in 1..=4 {
        if !visited[i] {
            if graph::detect_cycle_directed_dfs(&matrix, &mut visited, i as u32, 4) {
                println!("{}", true);
                return;
            }
        }

        println!("{:?}", visited)
    }
    println!("{}", false)
}

fn maciwin() {
    let lengthx: i32 = 5;
    let lengthy: i32 = 8;
    let mut matrix: Vec<Vec<&str>> = vec![vec!["#"; 8]; 5];

    let mut lines = stdin().lines();
    matrix[1][1] = ".";
    matrix[1][2] = "A";
    matrix[1][4] = ".";
    matrix[1][5] = ".";
    matrix[1][6] = ".";
    matrix[2][1] = ".";
    matrix[2][4] = ".";
    matrix[2][6] = "B";
    matrix[3][1] = ".";
    matrix[3][2] = ".";
    matrix[3][3] = ".";
    matrix[3][4] = ".";
    matrix[3][5] = ".";
    matrix[3][6] = ".";
    //     ########
    // #.A#...#
    // #.##.#B#
    // #......#
    let mut start: graph::position = graph::position { row: 1, col: 2 };
    let mut end: graph::position = graph::position { row: 2, col: 6 }; // ########
    for i in 0..lengthx {
        for j in 0..lengthy {}
    }

    // for i in 0..lengthx {
    //     let n_x: Vec<&str> = lines.next().unwrap().unwrap().split_whitespace().collect();
    //     for j in 0..lengthy {
    //         let curr_char: &str = n_x[j as usize];

    //         if curr_char == "A" {
    //             start.row = i;
    //             start.col = j;
    //         }

    //         if curr_char == "B" {
    //             end.row = i;
    //             end.col = j;
    //         }

    //         matrix[i as usize][j as usize] = n_x[j as usize];
    //     }
    // }
    let mut visited: HashMap<i32, bool> = HashMap::new();
    visited.insert(start.normalize(lengthy), true);
    let minDist: i32 = graph::labyrinth(&matrix, start, end, lengthx, lengthy, &mut visited);
    println!("return {}", minDist);
}

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
}

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
fn round_trip_2(
    matrix: &HashMap<u32, Vec<u32>>,
    start: u32,
    stack: &mut VecDeque<Vec<u32>>,
    path: &mut Vec<u32>,
    size: u32,
) -> (u32, bool) {
    let mut c: u32 = 1;
    let mut possible: bool = false;
    let mut local_visited: Vec<bool> = vec![false; (size + 1) as usize];

    while let Some(parent_node) = stack.pop_back() {
        if local_visited[parent_node[1] as usize] {
            if parent_node[1] == start {
                c += 1;
                possible = true;
                path.push(start);
                return (c, possible);
            }
        } else {
            local_visited[parent_node[1] as usize] = true;

            if let Some(children) = matrix.get(&parent_node[1]) {
                for child in children {
                    if *child != parent_node[0] {
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
fn round_trip_3(
    matrix: &HashMap<u32, Vec<u32>>,
    start: u32,
    stack: &mut VecDeque<Vec<u32>>,
    path: &mut Vec<u32>,
    size: u32,
) -> (u32, bool) {
    let mut c: u32 = 1;
    let mut possible: bool = false;
    let mut local_visited: Vec<bool> = vec![false; (size + 1) as usize];

    while let Some(parent_node) = stack.pop_back() {
        if local_visited[parent_node[1] as usize] {
            if parent_node[1] == start {
                c += 1;
                possible = true;
                path.push(start);
                return (c, possible);
            }
        } else {
            local_visited[parent_node[1] as usize] = true;

            if let Some(children) = matrix.get(&parent_node[1]) {
                for child in children {
                    if *child != parent_node[0] {
                        let result = round_trip_3(matrix, start, stack, path, size);
                        if result.1 {
                            return result;
                        }
                    }
                }
            }
        }
    }

    (c, possible)
}

fn main_setup_round_trip() {
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
    println!("matrix: {:?}", matrix);

    for i in 1..=n {
        let mut stack: VecDeque<Vec<u32>> = VecDeque::new();
        stack.push_back(vec![0, i]);
        let mut path: Vec<u32> = vec![i];
        let possible = round_trip_3(&matrix, i, &mut stack, &mut path, n);
        if possible.1 {
            println!("{}", possible.0);
            for idx in 0..possible.0 {
                println!("{}", path[idx as usize]);
            }
            return;
        }
    }

    println!("IMPOSSIBlE");
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

    graph::dijskatra_shortest_path(&matrix, n, n, EdgeWeight::new(1, 0));
}
