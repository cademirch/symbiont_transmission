use anyhow::{anyhow, bail, Result};
use clap::Parser;
use log::trace;
use phylotree::tree::{Node, NodeId, Tree, TreeError};
use rand::prelude::*;
use rand::rngs::{StdRng, ThreadRng};
use rand::RngCore;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, Uniform};
use statrs::distribution::{Discrete, Hypergeometric};
use statrs::statistics::{Max, Min};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::{BufRead, BufReader};
use serde::Serialize;

#[derive(Parser, Serialize)]
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

    #[arg(long, help = "Paths to files containing depth distributions", value_delimiter = ',')]
    depth_files: Vec<PathBuf>,

    #[arg(long, default_value = "2", help = "Minimum number of alternative reads required for variant detection")]
    min_alt_reads: u32,

    #[arg(long, default_value = "10", help = "Minimum depth required for variant calling")]
    min_depth: u32,

    #[arg(long, default_value = "output", help = "Output directory for results")]
    output_dir: PathBuf,

    #[arg(long, default_value = "0.05", help = "Bin size for folded AFS relative frequencies")]
    bin_size: f64,

    #[arg(short = 'R', long, default_value = "1", help = "Number of simulation runs")]
    runs: u32,
}

impl Cli {
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize CLI to JSON: {}", e))
    }
}

#[derive(Clone, Copy)]
enum AFSType {
    Absolute,
    Relative,
    Binned,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    
    // Create output directory
    std::fs::create_dir_all(&cli.output_dir).unwrap();

    let simulation_start = Instant::now();
    
    // Initialize cumulative timing variables
    let mut total_mutation_time = Duration::default();
    let mut total_stasis_time = Duration::default();
    let mut total_binary_time = Duration::default();
    let mut total_afs_time = Duration::default();
    let mut total_mutation_calls = 0;
    let mut total_mutations = 0;

    // Run the simulation R times
    for run in 0..cli.runs {
        println!("Starting run {} of {}", run + 1, cli.runs);
        
        let mut rng: Box<dyn RngCore> = if let Some(seed) = cli.seed {
            Box::new(StdRng::seed_from_u64(seed + run as u64))
        } else {
            Box::new(rand::rng())
        };

        // Create initial sample
        let (mut tree, mut active_lineages) = create_initial_sample(cli.sample_size);

        // Initialize mutation tracking map (NodeId -> mutation count)
        let mut mutation_counts: HashMap<NodeId, usize> = HashMap::new();

        // Initialize timing variables for this run
        let mut mutation_time = Duration::default();
        let mut stasis_time = Duration::default();
        let mut binary_time = Duration::default();

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
            let run_path = if cli.runs > 1 {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("tree");
                let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                path.with_file_name(format!("{}_{}.{}", stem, run, extension))
            } else {
                path.clone()
            };
            tree.to_file(&run_path).unwrap();
        }

        // Calculate and export the AFS
        let afs_start = Instant::now();
        
        // Always calculate the original AFS first
        let orig_afs = calculate_afs(&tree, &mutation_counts).unwrap();
        
        if !cli.depth_files.is_empty() {
            // Export original AFS (append mode for runs > 0)
            let orig_output = cli.output_dir.join("original_afs.csv");
            append_afs(&orig_afs, &orig_output, AFSType::Absolute, run == 0, run).unwrap();
            
            // Process each depth file
            for depth_file_path in &cli.depth_files {
                let depth_distribution = read_depth_distribution(depth_file_path).unwrap();
                
                // Get depth file stem for naming
                let depth_file_stem = depth_file_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("depth");
                
                // 1. Generate raw subsampled AFS
                let depth_afs = subsample_afs_with_depth(
                    &orig_afs,
                    cli.sample_size,
                    &depth_distribution,
                    cli.min_alt_reads,
                    cli.min_depth,
                    rng.as_mut(),
                ).unwrap();
                
                // Export raw AFS
                let raw_output = cli.output_dir.join(format!("afs_{}.csv", depth_file_stem));
                append_afs(&depth_afs, &raw_output, AFSType::Absolute, run == 0, run).unwrap();
                
                // 2. Generate and export folded AFS
                let folded_afs = fold_afs(&depth_afs);
                let folded_output = cli.output_dir.join(format!("folded_afs_{}.csv", depth_file_stem));
                append_afs(&folded_afs, &folded_output, AFSType::Absolute, run == 0, run).unwrap();
                
                // 3. Generate binned folded AFS from the folded AFS
                let binned_afs = bin_folded_afs(&folded_afs, cli.bin_size);
                
                // Export binned AFS
                let binned_output = cli.output_dir.join(format!("binned_folded_afs_{}.csv", depth_file_stem));
                append_binned_afs(&binned_afs, &binned_output, run == 0, run).unwrap();
            }
        } else if let Some(subsample_size) = cli.subsample {
            // Original subsampling without depth
            let orig_output = cli.output_dir.join("original_afs.csv");
            let afs_type = if cli.relative_freq { AFSType::Relative } else { AFSType::Absolute };
            append_afs(&orig_afs, &orig_output, afs_type, run == 0, run).unwrap();
            
            let subsampled_afs = subsample_afs(
                &orig_afs,
                cli.sample_size.try_into().unwrap(),
                subsample_size,
            ).unwrap();
            
            let subsample_output = cli.output_dir.join("subsampled_afs.csv");
            append_afs(&subsampled_afs, &subsample_output, afs_type, run == 0, run).unwrap();
        } else {
            // Just output the original AFS
            let output_path = cli.output_dir.join("afs.csv");
            let afs_type = if cli.relative_freq { AFSType::Relative } else { AFSType::Absolute };
            append_afs(&orig_afs, &output_path, afs_type, run == 0, run).unwrap();
        }

        let afs_time = afs_start.elapsed();
        
        // Accumulate timing statistics
        total_mutation_time += mutation_time;
        total_stasis_time += stasis_time;
        total_binary_time += binary_time;
        total_afs_time += afs_time;
        total_mutation_calls += mutation_calls;
        total_mutations += mutation_counts.values().sum::<usize>();
        
        println!("  - Run {} completed with {} mutations", run + 1, mutation_counts.values().sum::<usize>());
    }

    let total_time = simulation_start.elapsed();

    // Print timing results
    println!("\nAll simulations completed:");
    println!("  - Number of runs: {}", cli.runs);
    println!("  - Results saved to: {}", cli.output_dir.display());
    println!("  - Total mutations across all runs: {}", total_mutations);
    if !cli.depth_files.is_empty() {
        println!("  - Processed {} depth distributions", cli.depth_files.len());
    }
    println!("\nTiming Statistics (across all runs):");
    println!("  - Total runtime: {:?}", total_time);
    println!(
        "  - Stasis phase: {:?} ({:.2}%)",
        total_stasis_time,
        (total_stasis_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - Binary phase: {:?} ({:.2}%)",
        total_binary_time,
        (total_binary_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - Mutation operations: {:?} ({:.2}%)",
        total_mutation_time,
        (total_mutation_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - AFS calculation: {:?} ({:.2}%)",
        total_afs_time,
        (total_afs_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!("  - Number of mutation calls: {}", total_mutation_calls);
    if total_mutation_calls > 0 {
        println!(
            "  - Average time per mutation call: {:?}",
            total_mutation_time / total_mutation_calls as u32
        );
    }

    // Save CLI configuration to JSON
    let config_path = cli.output_dir.join("config.json");
    if let Ok(config_json) = cli.to_json() {
        if let Err(e) = std::fs::write(&config_path, config_json) {
            eprintln!("Warning: Failed to save config to {}: {}", config_path.display(), e);
        } else {
            println!("  - Configuration saved to: {}", config_path.display());
        }
    } else {
        eprintln!("Warning: Failed to serialize CLI configuration to JSON");
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

    
    for (&orig_freq, &variant_count) in original_afs.iter() {
        let orig_freq = orig_freq as u64; 

        let hyper = Hypergeometric::new(n_original, orig_freq, n_subsample) // (N, K, n)
            .map_err(|e| anyhow!("Hypergeometric::new failed: {}", e))?;

        // iterate over possible subsample counts x
        let lower = hyper.min(); // = max(0, n + K - N)
        let upper = hyper.max(); // = min(K, n)
        for x in lower..=upper {
            
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



fn read_depth_distribution(file_path: &std::path::Path) -> Result<Vec<u32>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut depths = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        // Assuming depth is in the 4th column (0-indexed column 3) of a BED file
        // or just a single depth value per line
        let depth: u32 = if line.contains('\t') {
            line.split('\t').nth(3)
                .ok_or_else(|| anyhow!("Missing depth column"))?
                .parse()?
        } else {
            line.trim().parse()?
        };
        depths.push(depth);
    }
    
    Ok(depths)
}

fn subsample_afs_with_depth(
    original_afs: &HashMap<usize, usize>,
    sample_size: u64,
    depth_distribution: &[u32],
    min_alt_reads: u32,
    min_depth: u32,
    rng: &mut dyn RngCore,
) -> Result<HashMap<usize, usize>> {
    let mut subsampled_afs = HashMap::new();
    let uniform = Uniform::new(0.0, 1.0)?;
    
    for (&freq, &variant_count) in original_afs.iter() {
        let true_frequency = freq as f64 / sample_size as f64;
        
        // For each individual mutation at this frequency
        for _ in 0..variant_count {
            // Sample a random depth from the distribution
            let depth_idx = rng.gen_range(0..depth_distribution.len());
            let depth = depth_distribution[depth_idx];
            
            // Skip if depth is too low
            if depth < min_depth {
                continue;
            }
            
            // Simulate individual reads using uniform random draws
            let mut alt_reads = 0;
            for _ in 0..depth {
                if uniform.sample(rng) < true_frequency {
                    alt_reads += 1;
                }
            }
            
            // Apply detection filters
            if alt_reads >= min_alt_reads {
                let observed_freq = alt_reads as usize;
                // Only count polymorphic sites (don't include fixed sites)
                if observed_freq > 0 && observed_freq < depth as usize {
                    *subsampled_afs.entry(observed_freq).or_insert(0) += 1;
                }
            }
        }
    }
    
    Ok(subsampled_afs)
}

fn bin_folded_afs(afs: &HashMap<usize, usize>, bin_size: f64) -> HashMap<String, usize> {
    let mut binned_mutations = HashMap::new();
    
    // Initialize bins from 0 to 0.5 (folded)
    let mut freq = 0.0;
    while freq < 0.5 {
        binned_mutations.insert(format!("{:.3}", freq), 0);
        freq += bin_size;
    }
    
    // Find the maximum frequency to determine sample size for the folded AFS
    let max_freq = afs.keys().max().copied().unwrap_or(0);
    let sample_size = max_freq * 2; // Since this is folded, multiply by 2 to get original sample size
    
    for (&freq, &count) in afs {
        // Convert frequency count back to relative frequency
        let relative_freq = freq as f64 / sample_size as f64;
        
        // Find appropriate bin
        if relative_freq > 0.0 {
            let mut bin_freq = 0.0;
            while bin_freq < 0.5 {
                if relative_freq >= bin_freq && relative_freq < (bin_freq + bin_size) {
                    let bin_key = format!("{:.3}", bin_freq);
                    *binned_mutations.entry(bin_key).or_insert(0) += count;
                    break;
                }
                bin_freq += bin_size;
            }
        }
    }
    
    binned_mutations
}

fn export_afs(
    data: &dyn std::any::Any,
    output_path: &std::path::Path,
    afs_type: AFSType,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(output_path)?;

    match afs_type {
        AFSType::Binned => {
            let binned_afs = data.downcast_ref::<HashMap<String, usize>>()
                .expect("Expected HashMap<String, usize> for binned AFS");
            
            writeln!(file, "frequency_bin,count")?;
            
            // Sort by frequency bin
            let mut bins: Vec<_> = binned_afs.keys().collect();
            bins.sort_by(|a, b| a.parse::<f64>().unwrap().partial_cmp(&b.parse::<f64>().unwrap()).unwrap());

            for bin in bins {
                if let Some(&count) = binned_afs.get(bin) {
                    writeln!(file, "{},{}", bin, count)?;
                }
            }
        }
        AFSType::Relative | AFSType::Absolute => {
            let afs = data.downcast_ref::<HashMap<usize, usize>>()
                .expect("Expected HashMap<usize, usize> for regular AFS");

            let relative = matches!(afs_type, AFSType::Relative);
            
            // Calculate total count for relative frequencies if needed
            let total_count: usize = if relative {
                afs.values().sum()
            } else {
                0
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
                        let rel_freq = count as f64 / total_count as f64;
                        writeln!(file, "{},{:.6}", freq, rel_freq)?;
                    } else {
                        writeln!(file, "{},{}", freq, count)?;
                    }
                }
            }
        }
    }

    Ok(())
}

// Remove the export_binned_afs function entirely

fn fold_afs(afs: &HashMap<usize, usize>) -> HashMap<usize, usize> {
    let mut folded = HashMap::new();
    
    // Find the maximum frequency to determine sample size
    let max_freq = afs.keys().max().copied().unwrap_or(0);
    let sample_size = max_freq + 1;
    
    for (&freq, &count) in afs {
        // For folded AFS, we take the minimum of freq and (sample_size - freq)
        // This matches the C++ logic: if freq > sample_size/2, fold it
        let folded_freq = if freq > sample_size / 2 {
            sample_size - freq
        } else {
            freq
        };
        
        // Only include if folded_freq > 0 (exclude monomorphic sites)
        if folded_freq > 0 {
            *folded.entry(folded_freq).or_insert(0) += count;
        }
    }
    
    folded
}

fn remove_random<T, R: Rng>(vec: &mut Vec<T>, rng: &mut R) -> Option<T> {
    if vec.is_empty() {
        return None;
    }
    let idx = rng.gen_range(0..vec.len());
    Some(vec.swap_remove(idx))
}

fn append_afs(
    afs: &HashMap<usize, usize>,
    output_path: &std::path::Path,
    afs_type: AFSType,
    write_header: bool,
    run_id: u32,
) -> std::io::Result<()> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let mut file = if write_header {
        std::fs::File::create(output_path)?
    } else {
        OpenOptions::new().create(true).append(true).open(output_path)?
    };

    let relative = matches!(afs_type, AFSType::Relative);
    
    // Calculate total count for relative frequencies if needed
    let total_count: usize = if relative {
        afs.values().sum()
    } else {
        0
    };

    // Write header only for first run
    if write_header {
        if relative {
            writeln!(file, "run,frequency,relative_frequency")?;
        } else {
            writeln!(file, "run,frequency,count")?;
        }
    }

    let mut freqs: Vec<_> = afs.keys().collect();
    freqs.sort();

    for &freq in freqs {
        if let Some(&count) = afs.get(&freq) {
            if relative && total_count > 0 {
                let rel_freq = count as f64 / total_count as f64;
                writeln!(file, "{},{},{:.6}", run_id, freq, rel_freq)?;
            } else {
                writeln!(file, "{},{},{}", run_id, freq, count)?;
            }
        }
    }

    Ok(())
}

fn append_binned_afs(
    binned_afs: &HashMap<String, usize>,
    output_path: &std::path::Path,
    write_header: bool,
    run_id: u32,
) -> std::io::Result<()> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let mut file = if write_header {
        std::fs::File::create(output_path)?
    } else {
        OpenOptions::new().create(true).append(true).open(output_path)?
    };

    // Write header only for first run
    if write_header {
        writeln!(file, "run,frequency_bin,count")?;
    }
    
    // Sort by frequency bin
    let mut bins: Vec<_> = binned_afs.keys().collect();
    bins.sort_by(|a, b| a.parse::<f64>().unwrap().partial_cmp(&b.parse::<f64>().unwrap()).unwrap());

    for bin in bins {
        if let Some(&count) = binned_afs.get(bin) {
            writeln!(file, "{},{},{}", run_id, bin, count)?;
        }
    }

    Ok(())
}
