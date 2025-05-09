use anyhow::{anyhow, bail, Result};
use clap::Parser;
use log::trace;
use phylotree::tree::{Node, NodeId, Tree, TreeError};
use rand::prelude::*;
use rand::rngs::{StdRng, ThreadRng};
use rand::RngCore;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};
use statrs::distribution::{Discrete, Hypergeometric};
use statrs::statistics::{Max, Min};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use rand_chacha::ChaCha20Rng;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short = 'n', long)]
    sample_size: u64,

    #[arg(short = 'N', long)]
    population_size: u64,

    #[arg(short = 'b', long)]
    bottleneck_size: u64,

    #[arg(short = 'g', long)]
    stasis_generations: u64,

    #[arg(short = 'u', long)]
    mutation_rate: f64,

    #[arg(short = 'o', long)]
    output_file: Option<PathBuf>,

    #[arg(short = 'a', long, default_value = "afs.csv")]
    afs_output: PathBuf,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(
        short = 'r',
        long,
        help = "Output relative frequencies instead of absolute counts"
    )]
    relative_freq: bool,

    #[arg(short = 's', long)]
    subsample: Option<u64>,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    let mut rng: Box<dyn RngCore> = if let Some(seed) = cli.seed {
        Box::new(StdRng::seed_from_u64(seed))
    } else {
        Box::new(ChaCha20Rng::from_os_rng())
    };

    let simulation_start = Instant::now();

    // Create initial sample
    let (mut tree, mut active_lineages) = create_initial_sample(cli.sample_size);

    // Initialize mutation tracking map (NodeId -> mutation count)
    let mut mutation_counts: HashMap<NodeId, usize> = HashMap::new();

    // Initialize timing variables
    let mut mutation_time = Duration::default();
    let mut stasis_time = Duration::default();
    let mut binary_time = Duration::default();
    let mut afs_time = Duration::default();

    // Initialize generation counter
    let mut current_gen = 0;

    // Count number of mutation calls
    let mut mutation_calls = 0;

    while active_lineages.len() > 1 {
        // Run stasis phase
        let stasis_start = Instant::now();
        let (new_gen, mut_time, calls) = stasis(
            &mut tree,
            &mut active_lineages,
            cli.stasis_generations,
            cli.population_size,
            current_gen,
            &mut rng,
            cli.mutation_rate,
            &mut mutation_counts,
        );
        current_gen = new_gen;
        stasis_time += stasis_start.elapsed();
        mutation_time += mut_time;
        mutation_calls += calls;

        // Check if we've reached a single lineage after stasis
        if active_lineages.len() <= 1 {
            break;
        }

        // Run binary phase
        let binary_start = Instant::now();
        let (new_gen, mut_time, calls) = binary(
            &mut tree,
            &mut active_lineages,
            cli.population_size,
            current_gen,
            &mut rng,
            cli.bottleneck_size,
            cli.mutation_rate,
            &mut mutation_counts,
        );
        current_gen = new_gen;
        binary_time += binary_start.elapsed();
        mutation_time += mut_time;
        mutation_calls += calls;
    }

    if let Some(path) = &cli.output_file {
        tree.to_file(path).unwrap();
    }

    // Calculate and export the AFS
    let afs_start = Instant::now();
    let afs = if let Some(subsample_size) = cli.subsample {
        let orig_afs = calculate_afs(&tree, &mutation_counts).unwrap();
        export_afs(
            &orig_afs,
            &cli.afs_output.with_extension("og.csv"),
            cli.relative_freq,
        )
        .unwrap();
        subsample_afs(
            &orig_afs,
            cli.sample_size.try_into().unwrap(),
            subsample_size,
        )
        .unwrap()
    } else {
        calculate_afs(&tree, &mutation_counts).unwrap()
    };
    // let afs = calculate_afs(&tree, &mutation_counts).unwrap();

    afs_time = afs_start.elapsed();
    export_afs(&afs, &cli.afs_output, cli.relative_freq).unwrap();

    let total_time = simulation_start.elapsed();
    let total_mutations = mutation_counts.values().sum::<usize>();

    // Print timing results
    println!("Simulation completed:");
    println!("  - AFS saved to: {}", cli.afs_output.display());
    println!("  - Total mutations: {}", total_mutations);
    println!("\nTiming Statistics:");
    println!("  - Total runtime: {:?}", total_time);
    println!(
        "  - Stasis phase: {:?} ({:.2}%)",
        stasis_time,
        (stasis_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - Binary phase: {:?} ({:.2}%)",
        binary_time,
        (binary_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - Mutation operations: {:?} ({:.2}%)",
        mutation_time,
        (mutation_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - AFS calculation: {:?} ({:.2}%)",
        afs_time,
        (afs_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!("  - Number of mutation calls: {}", mutation_calls);
    if mutation_calls > 0 {
        println!(
            "  - Average time per mutation call: {:?}",
            mutation_time / mutation_calls as u32
        );
    }
}

fn stasis(
    tree: &mut Tree,
    active_lineages: &mut Vec<NodeId>,
    stasis_generations: u64,
    population_size: u64,
    current_gen: u64,
    rng: &mut impl Rng,
    mutation_rate: f64,
    mutation_counts: &mut HashMap<NodeId, usize>,
) -> (u64, Duration, usize) {
    let mut gen = current_gen;
    let mut total_mutation_time = Duration::default();
    let mut total_mutation_calls = 0;

    for _ in 0..stasis_generations {
        if active_lineages.len() <= 1 {
            break;
        }

        gen += 1;

        // Calculate expected number of coalescences per generation
        let num_lineages = active_lineages.len() as u64;
        let coal_rate = (num_lineages * (num_lineages - 1)) as f64 / (2.0 * population_size as f64);
        let poisson = Poisson::new(coal_rate).unwrap();
        let expected_coals = poisson.sample(rng) as usize;

        if expected_coals > 0 {
            let (new_parents, mutation_time, mutation_calls) = perform_coalescence_with_mutations(
                tree,
                active_lineages,
                gen,
                expected_coals,
                mutation_rate,
                rng,
                mutation_counts,
            );
            active_lineages.extend(new_parents);
            total_mutation_time += mutation_time;
            total_mutation_calls += mutation_calls;
        }
    }

    (gen, total_mutation_time, total_mutation_calls)
}

fn binary(
    tree: &mut Tree,
    active_lineages: &mut Vec<NodeId>,
    population_size: u64,
    current_gen: u64,
    rng: &mut impl Rng,
    bottleneck_size: u64,
    mutation_rate: f64,
    mutation_counts: &mut HashMap<NodeId, usize>,
) -> (u64, Duration, usize) {
    let mut gen = current_gen;
    let mut total_mutation_time = Duration::default();
    let mut total_mutation_calls = 0;

    while active_lineages.len() > 1 {
        gen += 1;

        // Binary phase uses the bottleneck size for population
        let num_lineages = active_lineages.len() as u64;
        let coal_rate = (num_lineages * (num_lineages - 1)) as f64 / (2.0 * bottleneck_size as f64);
        let poisson = Poisson::new(coal_rate).unwrap();
        let expected_coals = poisson.sample(rng) as usize;

        if expected_coals > 0 {
            let (new_parents, mutation_time, mutation_calls) = perform_coalescence_with_mutations(
                tree,
                active_lineages,
                gen,
                expected_coals,
                mutation_rate,
                rng,
                mutation_counts,
            );
            active_lineages.extend(new_parents);
            total_mutation_time += mutation_time;
            total_mutation_calls += mutation_calls;
        }
    }

    (gen, total_mutation_time, total_mutation_calls)
}

// Helper function to create initial samples
fn create_initial_sample(sample_size: u64) -> (Tree, Vec<NodeId>) {
    let mut tree = Tree::new();
    let mut active_lineages = Vec::with_capacity(sample_size as usize);

    for i in 0..sample_size {
        let leaf = Node::new_named(&format!("Sample_{}", i));
        let leaf_id = tree.add(leaf);
        tree.get_mut(&leaf_id).unwrap().set_depth(0);
        active_lineages.push(leaf_id);
    }

    (tree, active_lineages)
}

fn perform_coalescence_with_mutations<R: Rng>(
    tree: &mut Tree,
    active_lineages: &mut Vec<NodeId>,
    current_generation: u64,
    expected_coals: usize,
    mutation_rate: f64,
    rng: &mut R,
    mutation_counts: &mut HashMap<NodeId, usize>,
) -> (Vec<NodeId>, Duration, usize) {
    let mut new_parents = Vec::new();
    let mut mut_times = Vec::new();
    let mut mutation_calls = 0;

    for c in 0..expected_coals {
        if active_lineages.len() <= 1 {
            break;
        }

        let child1 = remove_random(active_lineages, rng).unwrap();
        let child2 = remove_random(active_lineages, rng).unwrap();

        let edge1 = current_generation - tree.get(&child1).unwrap().get_depth() as u64;
        let edge2 = current_generation - tree.get(&child2).unwrap().get_depth() as u64;

        // Add mutations to the branches before merging
        let start = Instant::now();
        add_branch_mutations(
            tree,
            &child1,
            edge1 as f64,
            mutation_rate,
            rng,
            mutation_counts,
        );
        add_branch_mutations(
            tree,
            &child2,
            edge2 as f64,
            mutation_rate,
            rng,
            mutation_counts,
        );
        mutation_calls += 2; // Count each call to add_branch_mutations
        mut_times.push(start.elapsed());

        let parent_id = tree
            .merge_children(
                &child1,
                &child2,
                Some(edge1 as f64),
                Some(edge2 as f64),
                None,
                Some(format!("Gen{}_C{}", current_generation, c)),
            )
            .unwrap();

        tree.get_mut(&parent_id)
            .unwrap()
            .set_depth(current_generation as usize);

        new_parents.push(parent_id);
    }

    let mut_time_sum: Duration = if mut_times.is_empty() {
        Duration::default()
    } else {
        mut_times.iter().sum()
    };

    (new_parents, mut_time_sum, mutation_calls)
}

fn add_branch_mutations<R: Rng>(
    tree: &mut Tree,
    node_id: &NodeId,
    branch_length: f64,
    mutation_rate: f64,
    rng: &mut R,
    mutation_counts: &mut HashMap<NodeId, usize>,
) {
    // Calculate expected number of mutations on this branch
    let expected_mutations = branch_length * mutation_rate;

    // Use Poisson distribution to get the actual number of mutations
    let poisson = Poisson::new(expected_mutations).unwrap();
    let num_mutations = poisson.sample(rng) as usize;

    if num_mutations > 0 {
        *mutation_counts.entry(*node_id).or_insert(0) += num_mutations;
    }
}

pub fn calculate_afs(
    tree: &Tree,
    mutation_counts: &HashMap<NodeId, usize>,
) -> Result<HashMap<usize, usize>, TreeError> {
    let sample_size = tree.n_leaves();
    let mut afs = HashMap::new();

    for (&node, &num_mut) in mutation_counts {
        // get all the leaves descending from `node` in one shot:
        let leaves = tree.get_subtree_leaves(&node)?;
        let k = leaves.len(); // how many leaves carry this mutation

        // only count polymorphic sites (1 ≤ k < sample_size)
        if (1..sample_size).contains(&k) {
            *afs.entry(k).or_insert(0) += num_mut;
        }
    }

    Ok(afs)
}

fn subsample_afs(
    original_afs: &HashMap<usize, usize>,
    n_original: u64,
    n_subsample: u64,
) -> Result<HashMap<usize, usize>> {
    let mut subsampled_afs = HashMap::new();

    // Process each frequency bin in the original AFS
    for (&orig_freq, &variant_count) in original_afs.iter() {
        let orig_freq = orig_freq as u64; // Convert from usize to u64

        let hyper = Hypergeometric::new(n_original, orig_freq, n_subsample)
            .map_err(|e| anyhow!("Hypergeometric::new failed: {}", e))?;

        // iterate over possible subsample counts x
        let lower = hyper.min(); // = max(0, n + K - N)
        let upper = hyper.max(); // = min(K, n)
        for x in lower..=upper {
            // skip x=0 if you only care about polymorphic sites
            if x == 0 {
                continue;
            }

            // log-PMF is stable
            let logp = hyper.ln_pmf(x);
            let p = logp.exp(); // back to probability space
            if !p.is_finite() {
                continue;
            }

            let expected = (variant_count as f64) * p;
            let count = expected.round() as usize;
            if count > 0 {
                *subsampled_afs.entry(x as usize).or_insert(0) += count;
            }
        }
    }

    Ok(subsampled_afs)
}

fn export_afs(
    afs: &HashMap<usize, usize>,
    output_path: &std::path::Path,
    relative: bool,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(output_path)?;

    // Calculate total count for relative frequencies if needed
    let total_count: usize = if relative {
        afs.values().sum()
    } else {
        0 // Not used if relative is false
    };

    // Write header
    if relative {
        writeln!(file, "frequency,relative_frequency")?;
    } else {
        writeln!(file, "frequency,count")?;
    }

    let mut freqs: Vec<_> = afs.keys().collect();
    freqs.sort();

    for &freq in freqs {
        if let Some(&count) = afs.get(&freq) {
            if relative && total_count > 0 {
                // Output relative frequency (as a decimal)
                let rel_freq = count as f64 / total_count as f64;
                writeln!(file, "{},{}", freq, rel_freq)?;
            } else {
                // Output absolute count
                writeln!(file, "{},{}", freq, count)?;
            }
        }
    }

    Ok(())
}

fn remove_random<T, R: Rng>(vec: &mut Vec<T>, rng: &mut R) -> Option<T> {
    if vec.is_empty() {
        return None;
    }
    let idx = rng.gen_range(0..vec.len());
    Some(vec.swap_remove(idx))
}
