pub fn yummy() {
    let num: u32 = 1000;

    let res: u32 = (3..num).collect::<Vec<u32>>().iter().fold(0, |mut acc, n| {
        if n % 3 == 0 || n % 5 == 0 {
            acc += n
        }
        acc
    });
    println!("{}", res)
}
