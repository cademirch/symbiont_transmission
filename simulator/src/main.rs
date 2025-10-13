use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand::RngCore;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, Hypergeometric, Poisson};
use rand_xoshiro::Xoshiro128StarStar;
use serde::{Deserialize, Serialize};
// use statrs::distribution::{Discrete, Hypergeometric};

use statrs::statistics::{Max, Min};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

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

    #[arg(short = 's', long)]
    subsample: Option<u64>,

    #[arg(
        long,
        help = "Paths to files containing depth distributions",
        value_delimiter = ','
    )]
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

#[derive(Debug, Clone)]
struct SimpleNode {
    id: usize,
    depth: u64,
    parent: Option<usize>,
    children: Vec<usize>,
    descendant_count: usize,
}

struct SimpleTree {
    nodes: HashMap<usize, SimpleNode>,
    next_id: usize,
}

impl SimpleTree {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    fn add_leaf(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let node = SimpleNode {
            id,
            depth: 0,
            parent: None,
            children: Vec::new(),
            descendant_count: 1, // Leaves always have 1 descendant (themselves)
        };

        self.nodes.insert(id, node);
        id
    }

    fn merge_children(&mut self, child1: usize, child2: usize, depth: u64) -> usize {
        let parent_id = self.next_id;
        self.next_id += 1;

        // Get descendant counts from children
        let descendants1 = self.nodes[&child1].descendant_count;
        let descendants2 = self.nodes[&child2].descendant_count;

        let parent = SimpleNode {
            id: parent_id,
            depth,
            parent: None,
            children: vec![child1, child2],
            descendant_count: descendants1 + descendants2, // Sum of children's counts
        };

        // Update children to point to parent
        if let Some(node) = self.nodes.get_mut(&child1) {
            node.parent = Some(parent_id);
        }
        if let Some(node) = self.nodes.get_mut(&child2) {
            node.parent = Some(parent_id);
        }

        self.nodes.insert(parent_id, parent);
        parent_id
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
    subsampled_mutations: &[Vec<(PathBuf, Vec<(u32, u32)>)>],
    output_dir: &PathBuf,
) -> Result<()> {
    let mut depth_file_data: HashMap<PathBuf, Vec<(usize, Vec<(u32, u32)>)>> = HashMap::new();

    for (run_idx, run_data) in subsampled_mutations.iter().enumerate() {
        for (depth_file, mutations) in run_data {
            depth_file_data
                .entry(depth_file.clone())
                .or_insert_with(Vec::new)
                .push((run_idx, mutations.clone()));
        }
    }

    for (depth_file, run_data) in depth_file_data {
        let depth_file_stem = depth_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let output_path = output_dir.join(format!("{}_subsampled_mutations.csv", depth_file_stem));
        let mut file = File::create(&output_path)?;

        writeln!(file, "run,depth,alt_count")?;

        for (run_idx, mutations) in run_data {
            for &(depth, alt_count) in mutations.iter() {
                writeln!(file, "{},{},{}", run_idx, depth, alt_count)?;
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = if let Some(config_path) = std::env::args().find(|arg| arg == "--config") {
        // Get the config file path (next argument)
        let config_file = std::env::args()
            .nth(std::env::args().position(|x| x == "--config").unwrap() + 1)
            .ok_or_else(|| anyhow!("Config file path required after --config"))?;

        Cli::from_config(&PathBuf::from(config_file))?
    } else {
        Cli::parse()
    };

    std::fs::create_dir_all(&cli.output_dir).unwrap();

    let mut all_mutations: Vec<HashMap<usize, usize>> = Vec::with_capacity(cli.runs);
    let mut subsampled_mutations: Vec<Vec<(PathBuf, Vec<(u32, u32)>)>> =
        Vec::with_capacity(cli.runs);

    for run in 0..cli.runs {
        println!("Starting simulation run {}/{}", run + 1, cli.runs);

        let mut rng: Box<dyn RngCore> = if let Some(seed) = cli.seed {
            Box::new(Xoshiro128StarStar::seed_from_u64(seed + run as u64))
        } else {
            let mut thread_rng = rand::rng();
            Box::new(Xoshiro128StarStar::from_rng(&mut thread_rng))
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
            let mut depth_dist = read_depth_distribution(depth_file)?;
            let subsampled = subsample_afs_with_depth(
                &mutations,
                cli.sample_size,
                &mut depth_dist,
                cli.min_alt_reads,
                cli.min_depth,
                &mut rng,
            )?;
            run_subsampled.push((depth_file.clone(), subsampled));
        }
        subsampled_mutations.push(run_subsampled);

        all_mutations.push(mutations);
        println!("Completed simulation run {}/{}", run + 1, cli.runs);
    }

    let all_mutations_path = cli.output_dir.join("all_mutations.csv");
    write_all_mutations_csv(&all_mutations, &all_mutations_path)?;

    if !subsampled_mutations.is_empty() && !subsampled_mutations[0].is_empty() {
        write_subsampled_mutations_csv(&subsampled_mutations, &cli.output_dir)?;
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
) -> (SimpleTree, HashMap<usize, f64>) {
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
    tree: &mut SimpleTree,
    active_lineages: &mut Vec<usize>,
    stasis_generations: u64,
    population_size: u64,
    current_gen: u64,
    rng: &mut impl Rng,
    branch_lengths: &mut HashMap<usize, f64>,
) -> u64 {
    let mut gen = current_gen;

    for i in 0..stasis_generations {
        if active_lineages.len() <= 1 {
            break;
        }

        gen += 1;

        if gen % 100 == 0 {
            println!(
                "Stasis phase - Generation: {}/{}, Active lineages: {}",
                gen,
                stasis_generations,
                active_lineages.len()
            );
        }

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
    tree: &mut SimpleTree,
    active_lineages: &mut Vec<usize>,
    current_gen: u64,
    rng: &mut impl Rng,
    bottleneck_size: u64,
    branch_lengths: &mut HashMap<usize, f64>,
) -> u64 {
    let mut gen = current_gen;

    while active_lineages.len() > 1 {
        gen += 1;

        if gen % 10 == 0 {
            println!(
                "Bottleneck phase - Generation: {}, Active lineages: {}",
                gen,
                active_lineages.len()
            );
        }

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
fn create_initial_sample(sample_size: u64) -> (SimpleTree, Vec<usize>) {
    let mut tree = SimpleTree::new();
    let mut active_lineages = Vec::with_capacity(sample_size as usize);

    for _ in 0..sample_size {
        let leaf_id = tree.add_leaf();
        active_lineages.push(leaf_id);
    }

    (tree, active_lineages)
}

fn coalesce<R: Rng>(
    tree: &mut SimpleTree,
    active_lineages: &mut Vec<usize>,
    current_generation: u64,
    expected_coals: usize,
    rng: &mut R,
    branch_lengths: &mut HashMap<usize, f64>,
) -> Vec<usize> {
    let mut new_parents = Vec::new();

    if active_lineages.len() < 2 {
        return new_parents;
    }

    active_lineages.shuffle(rng);

    // Limit coalescences to prevent popping from empty vector
    // Each coalescence needs 2 lineages, so max possible is len/2
    let max_possible_coals = active_lineages.len() / 2;
    let actual_coals = expected_coals.min(max_possible_coals);

    for _ in 0..actual_coals {
        let child1 = active_lineages.pop().unwrap();
        let child2 = active_lineages.pop().unwrap();

        let edge1 = current_generation - tree.nodes[&child1].depth;
        let edge2 = current_generation - tree.nodes[&child2].depth;

        let descendants1 = tree.nodes[&child1].descendant_count;
        let descendants2 = tree.nodes[&child2].descendant_count;

        *branch_lengths.entry(descendants1).or_insert(0.0) += edge1 as f64;
        *branch_lengths.entry(descendants2).or_insert(0.0) += edge2 as f64;

        let parent_id = tree.merge_children(child1, child2, current_generation);

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
        // let lower = hyper.min(); // = max(0, n + K - N)
        // let upper = hyper.max(); // = min(K, n)
        // for x in lower..=upper {
        //     if x == 0 {
        //         continue;
        //     }

        //     // log-PMF is stable
        //     let logp = hyper.ln_pmf(x);
        //     let p = logp.exp(); // back to probability space
        //     if !p.is_finite() {
        //         continue;
        //     }

        // let expected = (variant_count as f64) * p;
        // let count = expected.round() as usize;
        // if count > 0 {
        //     *subsampled_afs.entry(x as usize).or_insert(0) += count;
        // }
        // }
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
    depth_distribution: &mut [u32],
    min_alt_reads: u32,
    min_depth: u32,
    rng: &mut impl RngCore,
) -> Result<Vec<(u32, u32)>> {
    let mut detectable_mutations = Vec::new();

    for (&freq, &variant_count) in original_afs.iter() {
        let alt_probability = freq as f64 / sample_size as f64;

        for _ in 0..variant_count {
            let depth = *depth_distribution
                .choose(rng)
                .ok_or_else(|| anyhow!("Empty depth distribution"))?;

            if depth < min_depth {
                continue;
            }

            // Binomial distribution: each read has probability alt_probability 
            // of being an alt allele, independently across depth reads.
            // 
            // Note: Biologically, sequencing is sampling without replacement 
            // (there are a finite number of alleles in the sample), which would 
            // be modeled by a hypergeometric distribution. However, when 
            // sample_size >>> depth (typically sample_size is in the thousands 
            // while depth is 10-1000x), the draws become essentially independent
            // and the binomial approximation is accurate. This is analogous to
            // how sampling a few balls from a very large urn approximates 
            // sampling with replacement.
            let binomial = Binomial::new(depth as u64, alt_probability)?;
            let alt_reads = binomial.sample(rng) as u32;

            if alt_reads >= min_alt_reads {
                detectable_mutations.push((depth, alt_reads));
            }
        }
    }

    Ok(detectable_mutations)
}

fn sample_hypergeometric_stable(
    population_size: u64,
    success_states: u64,
    draws: u64,
    rng: &mut impl RngCore,
) -> Result<u32> {
    if draws == 0 {
        return Ok(0);
    }
    
    if success_states == 0 {
        return Ok(0);
    }
    
    if draws >= population_size {
        return Ok(success_states as u32);
    }
    
    // Sequential sampling: for each draw, calculate probability of drawing success
    let mut successes = 0;
    let mut remaining_success = success_states;
    let mut remaining_population = population_size;
    
    for _ in 0..draws {
        let prob = remaining_success as f64 / remaining_population as f64;
        
        if rng.gen::<f64>() < prob {
            successes += 1;
            remaining_success -= 1;
        }
        
        remaining_population -= 1;
        
        if remaining_success == 0 {
            break;
        }
    }
    
    Ok(successes)
}