use anyhow::{anyhow, bail, Result};
use clap::Parser;
use log::trace;
use phylotree::tree::{Node, NodeId, Tree, TreeError};
use rand::prelude::*;
use rand::rngs::{StdRng, ThreadRng};

use rand::RngCore;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, Uniform};
use serde::{Serialize, Deserialize};
use statrs::distribution::{Discrete, Hypergeometric};
use statrs::statistics::{Max, Min};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser, Serialize, Deserialize)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long, help = "Load configuration from JSON file")]
    config: Option<PathBuf>,

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

    #[arg(
        long,
        default_value = "2",
        help = "Minimum number of alternative reads required for variant detection"
    )]
    min_alt_reads: u32,

    #[arg(
        long,
        default_value = "10",
        help = "Minimum depth required for variant calling"
    )]
    min_depth: u32,

    #[arg(long, default_value = "output", help = "Output directory for results")]
    output_dir: PathBuf,

    #[arg(
        long,
        default_value = "0.05",
        help = "Bin size for folded AFS relative frequencies"
    )]
    bin_size: f64,

    #[arg(
        short = 'R',
        long,
        default_value = "1",
        help = "Number of simulation runs"
    )]
    runs: usize,
}

impl Cli {
    pub fn to_json(&self, file_path: &PathBuf) -> Result<()> {
        let json_string = serde_json::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize CLI to JSON: {}", e))?;
        
        let mut file = File::create(file_path)?;
        write!(file, "{}", json_string)?;
        
        Ok(())
    }
    fn from_config(config_path: &PathBuf) -> Result<Self> {
        let config_str = std::fs::read_to_string(config_path)?;
        let cli: Cli = serde_json::from_str(&config_str)?;
        Ok(cli)
    }
}

fn write_all_mutations_csv(
    all_mutations: &[HashMap<usize, usize>],
    file_path: &PathBuf,
) -> Result<()> {
    let mut file = File::create(file_path)?;

    writeln!(file, "run,allele_freq,num_observed")?;

    for (run_idx, mutations) in all_mutations.iter().enumerate() {
        for (&allele_freq, &num_observed) in mutations.iter() {
            writeln!(file, "{},{},{}", run_idx, allele_freq, num_observed)?;
        }
    }

    Ok(())
}

fn write_subsampled_mutations_csv(
    subsampled_mutations: &[Vec<(PathBuf, Vec<usize>)>],
    bin_size: f64,
    output_dir: &PathBuf,
) -> Result<()> {
    
    let mut depth_file_data: HashMap<PathBuf, Vec<(usize, Vec<usize>)>> = HashMap::new();
    
    for (run_idx, run_data) in subsampled_mutations.iter().enumerate() {
        for (depth_file, binned_counts) in run_data {
            depth_file_data
                .entry(depth_file.clone())
                .or_insert_with(Vec::new)
                .push((run_idx, binned_counts.clone()));
        }
    }
    
    for (depth_file, run_data) in depth_file_data {
        let depth_file_stem = depth_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        
        let output_path = output_dir.join(format!("{}_subsampled_mutations.csv", depth_file_stem));
        let mut file = File::create(&output_path)?;
        
        writeln!(file, "run,allele_freq_bin,num_observed")?;
        
        for (run_idx, binned_counts) in run_data {
            for (bin_idx, &num_observed) in binned_counts.iter().enumerate() {
                if num_observed > 0 {
                    let bin_start = bin_idx as f64 * bin_size;
                    writeln!(file, "{},{:.5},{}", run_idx, bin_start, num_observed)?;
                }
            }
        }
    }
    
    Ok(())
}

fn main() -> Result<()> {
    let cli = if let Some(config_path) = std::env::args().find(|arg| arg == "--config") {
        // Get the config file path (next argument)
        let config_file = std::env::args().nth(
            std::env::args().position(|x| x == "--config").unwrap() + 1
        ).ok_or_else(|| anyhow!("Config file path required after --config"))?;
        
        Cli::from_config(&PathBuf::from(config_file))?
    } else {
        Cli::parse()
    };

    std::fs::create_dir_all(&cli.output_dir).unwrap();

    let mut all_mutations: Vec<HashMap<usize, usize>> = Vec::with_capacity(cli.runs);
    let mut subsampled_mutations: Vec<Vec<(PathBuf, Vec<usize>)>> = Vec::with_capacity(cli.runs);

    for run in 0..cli.runs {
        let mut rng: Box<dyn RngCore> = if let Some(seed) = cli.seed {
            Box::new(StdRng::seed_from_u64(seed + run as u64))
        } else {
            Box::new(rand::rng())
        };
        let (_tree, branch_lengths) = run_single_simulation(
            cli.sample_size,
            cli.population_size,
            cli.bottleneck_size,
            cli.stasis_generations,
            &mut rng,
        );
        let mutations = draw_mutations(&branch_lengths, cli.mutation_rate, &mut rng);
        
        
        let mut run_subsampled = Vec::new();
        for depth_file in &cli.depth_files {
            let depth_dist = read_depth_distribution(depth_file)?;
            let subsampled = subsample_afs_with_depth(
                &mutations,
                cli.sample_size,
                &depth_dist,
                cli.min_alt_reads,
                cli.min_depth,
                cli.bin_size,
                &mut rng,
            )?;
            run_subsampled.push((depth_file.clone(), subsampled));
        }
        subsampled_mutations.push(run_subsampled);
        
        all_mutations.push(mutations);
    }
    
    let all_mutations_path = cli.output_dir.join("all_mutations.csv");
    write_all_mutations_csv(&all_mutations, &all_mutations_path)?;
    
    
    if !subsampled_mutations.is_empty() && !subsampled_mutations[0].is_empty() {
        write_subsampled_mutations_csv(&subsampled_mutations, cli.bin_size, &cli.output_dir)?;
    }

    let config_path = cli.output_dir.join("config.json");
    cli.to_json(&config_path)?;
    Ok(())
}

fn run_single_simulation(
    sample_size: u64,
    population_size: u64,
    bottleneck_size: u64,
    stasis_generations: u64,
    rng: &mut impl RngCore,
) -> (Tree, HashMap<usize, f64>) {
    let (mut tree, mut active_lineages) = create_initial_sample(sample_size);

    let mut branch_lengths: HashMap<usize, f64> = HashMap::new();

    let mut current_gen = 0;

    while active_lineages.len() > 1 {
        current_gen = stasis(
            &mut tree,
            &mut active_lineages,
            stasis_generations,
            population_size,
            current_gen,
            rng,
            &mut branch_lengths,
        );

        if active_lineages.len() <= 1 {
            break;
        }

        current_gen = binary(
            &mut tree,
            &mut active_lineages,
            current_gen,
            rng,
            bottleneck_size,
            &mut branch_lengths,
        );
    }

    (tree, branch_lengths)
}

fn stasis(
    tree: &mut Tree,
    active_lineages: &mut Vec<NodeId>,
    stasis_generations: u64,
    population_size: u64,
    current_gen: u64,
    rng: &mut impl Rng,
    branch_lengths: &mut HashMap<usize, f64>,
) -> u64 {
    let mut gen = current_gen;

    for _ in 0..stasis_generations {
        if active_lineages.len() <= 1 {
            break;
        }

        gen += 1;

        let num_lineages = active_lineages.len() as u64;
        let coal_rate = (num_lineages * (num_lineages - 1)) as f64 / (2.0 * population_size as f64);
        let poisson = Poisson::new(coal_rate).unwrap();
        let expected_coals = poisson.sample(rng) as usize;

        if expected_coals > 0 {
            let new_parents = coalesce(
                tree,
                active_lineages,
                gen,
                expected_coals,
                rng,
                branch_lengths,
            );
            active_lineages.extend(new_parents);
        }
    }

    gen
}

fn binary(
    tree: &mut Tree,
    active_lineages: &mut Vec<NodeId>,
    current_gen: u64,
    rng: &mut impl Rng,
    bottleneck_size: u64,
    branch_lengths: &mut HashMap<usize, f64>,
) -> u64 {
    let mut gen = current_gen;

    while active_lineages.len() > 1 {
        gen += 1;

        let num_lineages = active_lineages.len() as u64;
        let coal_rate = (num_lineages * (num_lineages - 1)) as f64 / (2.0 * bottleneck_size as f64);
        let poisson = Poisson::new(coal_rate).unwrap();
        let expected_coals = poisson.sample(rng) as usize;

        if expected_coals > 0 {
            let new_parents = coalesce(
                tree,
                active_lineages,
                gen,
                expected_coals,
                rng,
                branch_lengths,
            );
            active_lineages.extend(new_parents);
        }
    }

    gen
}

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

fn coalesce<R: Rng>(
    tree: &mut Tree,
    active_lineages: &mut Vec<NodeId>,
    current_generation: u64,
    expected_coals: usize,
    rng: &mut R,
    branch_lengths: &mut HashMap<usize, f64>,
) -> Vec<NodeId> {
    let mut new_parents = Vec::new();

    for c in 0..expected_coals {
        if active_lineages.len() <= 1 {
            break;
        }

        let child1 = remove_random(active_lineages, rng).unwrap();
        let child2 = remove_random(active_lineages, rng).unwrap();

        let edge1 = current_generation - tree.get(&child1).unwrap().get_depth() as u64;
        let edge2 = current_generation - tree.get(&child2).unwrap().get_depth() as u64;

        // number of leaves for each child node we are coalescing
        let descendants1 = tree.get_subtree_leaves(&child1).unwrap().len();
        let descendants2 = tree.get_subtree_leaves(&child2).unwrap().len();

        *branch_lengths.entry(descendants1).or_insert(0.0) += edge1 as f64;
        *branch_lengths.entry(descendants2).or_insert(0.0) += edge2 as f64;

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

    new_parents
}

fn draw_mutations<R: Rng>(
    branch_lengths: &HashMap<usize, f64>,
    mutation_rate: f64,
    rng: &mut R,
) -> HashMap<usize, usize> {
    let mut mutations: HashMap<usize, usize> = HashMap::new();

    // Draw mutations in relation to branch lengths
    // The frequency that a mutation appears at within the population will be
    // proportional to the number of branch segments that end in number of forks = frequency
    for (&num_descendants, &total_branch_length) in branch_lengths {
        let expected_mutations = total_branch_length * mutation_rate;
        let poisson = Poisson::new(expected_mutations).unwrap();
        let num_mutations = poisson.sample(rng) as usize;

        if num_mutations > 0 {
            mutations.insert(num_descendants, num_mutations);
        }
    }

    mutations
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
            line.split('\t')
                .nth(3)
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
    bin_size: f64,
    rng: &mut impl RngCore,
) -> Result<Vec<usize>> {
    let uniform = Uniform::new(0.0, 1.0)?;
    let num_bins = (0.5 / bin_size).ceil() as usize;
    let mut binned_counts = vec![0; num_bins];
    for (&freq, &variant_count) in original_afs.iter() {
        // freq = number of individuals carrying this variant
        // variant_count = number of independent mutations at this frequency

        // For each individual mutation at this frequency
        for _ in 0..variant_count {
            let depth = *depth_distribution.choose(rng).unwrap();

            if depth < min_depth {
                continue;
            }

            let alt_probability = freq as f64 / sample_size as f64;

            let mut alt_reads = 0;
            for _ in 0..depth {
                if uniform.sample(rng) < alt_probability {
                    alt_reads += 1;
                }
            }

            if alt_reads >= min_alt_reads {
                let allele_freq = alt_reads as f64 / depth as f64;
                let folded_freq = allele_freq.min(1.0 - allele_freq);

                // folded freq should never be ==0.5 but check incase. would go out of bounds of bin vec if is.
                if folded_freq > 0.0 && folded_freq < 0.5 {
                    //  bin index using left-closed, right-open intervals [a, b)
                    let bin_index = (folded_freq / bin_size).floor() as usize;

                    binned_counts[bin_index] += 1;
                }
            }
        }
    }

    Ok(binned_counts)
}

fn remove_random<T, R: Rng>(vec: &mut Vec<T>, rng: &mut R) -> Option<T> {
    if vec.is_empty() {
        return None;
    }
    let idx = rng.random_range(0..vec.len());
    Some(vec.swap_remove(idx))
}
