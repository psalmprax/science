//! High-Performance Parallel Goldbach Conjecture Verifier in Rust.
//!
//! Verifies that every even integer 2n in [4, N] can be expressed as the sum of two primes.
//! Uses a fast Eratosthenes sieve followed by multi-threaded Rayon work-stealing verification.

use std::env;
use std::time::Instant;
use rayon::prelude::*;

fn sieve(n: usize) -> Vec<bool> {
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    if n >= 1 {
        is_prime[1] = false;
    }
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
    let args: Vec<String> = env::args().collect();
    let is_json = args.iter().any(|a| a == "--json");
    let mut limit: usize = 10_000_000;

    for arg in args.iter().skip(1) {
        if arg != "--json" {
            if let Ok(val) = arg.replace('_', "").parse::<usize>() {
                limit = val;
            }
        }
    }

    if !is_json {
        println!("===============================================================");
        println!("  PARALLEL RUST GOLDBACH CONJECTURE EMPIRICAL VERIFIER (RAYON) ");
        println!("===============================================================");
        println!("Testing all even integers 2n in [4, {}]...", limit);
    }
    let start_total = Instant::now();

    let t_sieve = Instant::now();
    let is_prime = sieve(limit);
    let sieve_duration = t_sieve.elapsed();
    if !is_json {
        println!("Step 1: Sieve of Eratosthenes up to {}...", limit);
        println!("        Completed in {:.2?}", sieve_duration);
    }

    let num_evens = limit / 2 - 1; // 4, 6, 8, ..., limit
    if !is_json {
        println!("Step 2: Multi-threaded parallel verification of {} even integers...", num_evens);
    }
    let t_verify = Instant::now();

    let violations: Vec<usize> = (2..=(limit / 2))
        .into_par_iter()
        .map(|k| k * 2)
        .filter(|&ev| {
            if ev == 4 {
                return false; // 2 + 2 = 4
            }
            let half = ev / 2;
            let mut found = false;
            // Iterate over odd primes p up to ev/2
            for p in (3..=half).step_by(2) {
                if is_prime[p] && is_prime[ev - p] {
                    found = true;
                    break;
                }
            }
            !found
        })
        .collect();

    let verify_duration = t_verify.elapsed();
    let total_duration = start_total.elapsed();
    let rate_m_sec = (num_evens as f64 / 1_000_000.0) / verify_duration.as_secs_f64().max(1e-9);

    if is_json {
        println!(
            "{{\"limit\":{},\"evens_tested\":{},\"violations\":{},\"sieve_time_ms\":{:.2},\"verify_time_ms\":{:.2},\"total_time_ms\":{:.2},\"rate_m_evens_per_sec\":{:.2}}}",
            limit,
            num_evens,
            violations.len(),
            sieve_duration.as_secs_f64() * 1000.0,
            verify_duration.as_secs_f64() * 1000.0,
            total_duration.as_secs_f64() * 1000.0,
            rate_m_sec
        );
    } else {
        println!("        Completed in {:.2?}", verify_duration);
        println!("---------------------------------------------------------------");
        println!("Summary:");
        println!("  Even integers tested : {}", num_evens);
        println!("  Counterexamples found: {}", violations.len());
        println!("  Verification rate    : {:.2} million evens/sec", rate_m_sec);
        println!("  Total runtime        : {:.2?}", total_duration);
        println!("  Epistemic status     : C2_PREDICTIVE (Empirically verified on finite domain [4, {}])", limit);
        println!("===============================================================");

        if !violations.is_empty() {
            println!("COUNTEREXAMPLE DISCOVERED! Violations: {:?}", &violations[..violations.len().min(10)]);
        } else {
            println!("Zero violations. Conjecture holds for all {} even integers in tested range.", num_evens);
        }
    }
}
