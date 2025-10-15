use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand::RngCore;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, Hypergeometric, Poisson};
use rand_xoshiro::Xoshiro128StarStar;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const CIRCLES_JSON: &str = include_str!("../data/circles_depth_cdf.json");
const DUPLEX_JSON: &str = include_str!("../data/duplex_depth_cdf.json");

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DepthStats {
    pub mean: f64,
    pub median: u32,
    pub min: u32,
    pub max: u32,
    pub total_sites: usize,
    pub unique_depths: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DepthDistribution {
    cdf: Vec<(u32, f64)>,
    stats: DepthStats,
}

impl DepthDistribution {
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        #[derive(Deserialize)]
        struct JsonFormat {
            cdf: Vec<[f64; 2]>,
            stats: DepthStats,
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let json: JsonFormat = serde_json::from_reader(reader)?;

        // Convert from [[depth, prob], ...] to Vec<(u32, f64)>
        let cdf = json
            .cdf
            .into_iter()
            .map(|[depth, prob]| (depth as u32, prob))
            .collect();

        Ok(Self {
            cdf,
            stats: json.stats,
        })
    }

    pub fn sample(&self, rng: &mut impl RngCore) -> u32 {
        let u: f64 = rng.random();

        match self
            .cdf
            .binary_search_by(|(_, prob)| prob.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => self.cdf[idx].0,
            Err(idx) => self
                .cdf
                .get(idx)
                .map(|(d, _)| *d)
                .unwrap_or(self.cdf.last().unwrap().0),
        }
    }

    pub fn stats(&self) -> &DepthStats {
        &self.stats
    }
}

#[derive(Debug, Clone)]
struct Mutation {
    depth: u32,
    alt_count: u32,
}

#[derive(Debug, Clone)]
struct SharedMutation {
    circles_depth: u32,
    circles_alt: u32,
    duplex_depth: u32,
    duplex_alt: u32,
}

struct SubsampledMutations {
    circles: Vec<Mutation>,
    duplex: Vec<Mutation>,
    shared: Vec<SharedMutation>,
}

#[derive(Parser, Serialize, Deserialize)]
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

    #[arg(long)]
    seed: Option<u64>,

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

    #[arg(
        long,
        default_value = "10",
        help = "Minimum depth required for circles variant calling"
    )]
    min_depth_circles: u32,

    #[arg(
        long,
        default_value = "10",
        help = "Minimum depth required for duplex variant calling"
    )]
    min_depth_duplex: u32,

    #[arg(long, default_value = "output", help = "Output prefix for results")]
    output_prefix: PathBuf,

    #[arg(
        short = 'R',
        long,
        default_value = "1",
        help = "Number of simulation runs"
    )]
    runs: usize,
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

fn write_mutations_csv(all_runs: &[SubsampledMutations], output_prefix: &PathBuf) -> Result<()> {
    let circles_path = PathBuf::from(format!("{}_circles.csv", output_prefix.display()));
    let mut circles_file = File::create(&circles_path)?;
    writeln!(circles_file, "run,depth,alt_count")?;

    for (run_idx, run_mutations) in all_runs.iter().enumerate() {
        for mutation in &run_mutations.circles {
            writeln!(
                circles_file,
                "{},{},{}",
                run_idx, mutation.depth, mutation.alt_count
            )?;
        }
    }

    let duplex_path = PathBuf::from(format!("{}_duplex.csv", output_prefix.display()));
    let mut duplex_file = File::create(&duplex_path)?;
    writeln!(duplex_file, "run,depth,alt_count")?;

    for (run_idx, run_mutations) in all_runs.iter().enumerate() {
        for mutation in &run_mutations.duplex {
            writeln!(
                duplex_file,
                "{},{},{}",
                run_idx, mutation.depth, mutation.alt_count
            )?;
        }
    }

    let shared_path = PathBuf::from(format!("{}_shared.csv", output_prefix.display()));
    let mut shared_file = File::create(&shared_path)?;
    writeln!(
        shared_file,
        "run,circles_depth,circles_alt,duplex_depth,duplex_alt"
    )?;

    for (run_idx, run_mutations) in all_runs.iter().enumerate() {
        for mutation in &run_mutations.shared {
            writeln!(
                shared_file,
                "{},{},{},{},{}",
                run_idx,
                mutation.circles_depth,
                mutation.circles_alt,
                mutation.duplex_depth,
                mutation.duplex_alt
            )?;
        }
    }

    println!("Wrote results to:");
    println!("  - {}", circles_path.display());
    println!("  - {}", duplex_path.display());
    println!("  - {}", shared_path.display());

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let circles_dist = serde_json::from_str::<DepthDistribution>(CIRCLES_JSON)?;
    let duplex_dist = serde_json::from_str::<DepthDistribution>(DUPLEX_JSON)?;

    println!("Circles distribution:");
    println!(
        "  Mean: {:.1}, Median: {}, Range: [{}, {}]",
        circles_dist.stats().mean,
        circles_dist.stats().median,
        circles_dist.stats().min,
        circles_dist.stats().max
    );
    println!("Duplex distribution:");
    println!(
        "  Mean: {:.1}, Median: {}, Range: [{}, {}]",
        duplex_dist.stats().mean,
        duplex_dist.stats().median,
        duplex_dist.stats().min,
        duplex_dist.stats().max
    );

    let mut all_subsampled: Vec<SubsampledMutations> = Vec::with_capacity(cli.runs);

    for run in 0..cli.runs {
        println!("\nStarting simulation run {}/{}", run + 1, cli.runs);

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

        let subsampled = subsample_afs_with_depth(
            &mutations,
            cli.sample_size,
            &circles_dist,
            &duplex_dist,
            cli.min_alt_reads,
            cli.min_depth_circles,
            cli.min_depth_duplex,
            &mut rng,
        )?;

        println!(
            "  Circles: {} detected, Duplex: {} detected, Shared: {} detected",
            subsampled.circles.len(),
            subsampled.duplex.len(),
            subsampled.shared.len()
        );

        all_subsampled.push(subsampled);
    }

    println!("\nWriting output files...");
    write_mutations_csv(&all_subsampled, &cli.output_prefix)?;

    println!("\n✓ Simulation complete!");
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

fn subsample_afs_with_depth(
    original_afs: &HashMap<usize, usize>,
    sample_size: u64,
    circles_depth_dist: &DepthDistribution,
    duplex_depth_dist: &DepthDistribution,
    min_alt_reads: u32,
    min_depth_circles: u32,
    min_depth_duplex: u32,
    rng: &mut impl RngCore,
) -> Result<SubsampledMutations> {
    let mut circles = Vec::new();
    let mut duplex = Vec::new();
    let mut shared = Vec::new();

    for (&freq, &variant_count) in original_afs.iter() {
        let alt_probability = freq as f64 / sample_size as f64;

        for _ in 0..variant_count {
            let circles_depth = circles_depth_dist.sample(rng);
            let duplex_depth = duplex_depth_dist.sample(rng);

            // Check circles detection
            let circles_alt = if circles_depth >= min_depth_circles {
                let binomial = Binomial::new(circles_depth as u64, alt_probability)?;
                binomial.sample(rng) as u32
            } else {
                0
            };

            // Check duplex detection
            let duplex_alt = if duplex_depth >= min_depth_duplex {
                let binomial = Binomial::new(duplex_depth as u64, alt_probability)?;
                binomial.sample(rng) as u32
            } else {
                0
            };

            let circles_detected = circles_alt >= min_alt_reads;
            let duplex_detected = duplex_alt >= min_alt_reads;

            if circles_detected {
                circles.push(Mutation {
                    depth: circles_depth,
                    alt_count: circles_alt,
                });
            }

            if duplex_detected {
                duplex.push(Mutation {
                    depth: duplex_depth,
                    alt_count: duplex_alt,
                });
            }

            if circles_detected && duplex_detected {
                shared.push(SharedMutation {
                    circles_depth,
                    circles_alt,
                    duplex_depth,
                    duplex_alt,
                });
            }
        }
    }

    Ok(SubsampledMutations {
        circles,
        duplex,
        shared,
    })
}
