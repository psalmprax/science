use std::time::Instant;
use rayon::prelude::*;

fn sieve(n: usize) -> Vec<bool> {
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    if n >= 1 { is_prime[1] = false; }
    let limit = (n as f64).sqrt() as usize;
    for i in 2..=limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= n {
                is_prime[j] = false;
                j += i;
            }
        }
    }
    is_prime
}

fn main() {
    let limit: usize = 100_000_000;
    println!("=== RUST GOLDBACH PARALLEL VERIFICATION ===");
    println!("Testing all even integers up to {}...", limit);
    let start = Instant::now();
    
    let is_prime = sieve(limit);
    println!("Sieve completed in {:.2?}", start.elapsed());
    
    let t_verify = Instant::now();
    let num_evens = limit / 2 - 1; // 4, 6, ..., limit
    let violations: Vec<usize> = (2..=(limit / 2))
        .into_par_iter()
        .map(|k| k * 2)
        .filter(|&ev| {
            if ev == 4 { return false; } // 2 + 2
            let half = ev / 2;
            let mut found = false;
            for p in (3..=half).step_by(2) {
                if is_prime[p] && is_prime[ev - p] {
                    found = true;
                    break;
                }
            }
            !found
        })
        .collect();
    
    let total_time = start.elapsed();
    println!("Verification of {} even numbers completed in {:.2?}", num_evens, t_verify.elapsed());
    println!("Total execution time: {:.2?}", total_time);
    println!("Violations found: {}", violations.len());
}
