use assert_cmd::Command as AssertCommand;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use tempfile::TempDir;
use std::sync::Once;

static INIT: Once = Once::new();

/// Setup function that is only run once, even if called multiple times.
fn setup() {
    INIT.call_once(|| {
        env_logger::init();
    });
}

#[test]
fn test_simulator_with_embedded_depths() {
    setup();
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output_prefix = temp_dir.path().join("test_output");

    let mut cmd = AssertCommand::cargo_bin("simulator").unwrap();
    cmd.args([
        "-n",
        "10000",
        "-N",
        "1000000",
        "-b",
        "50",
        "-g",
        "500",
        "-u",
        "0.006",
        "--output-prefix",
        output_prefix.to_str().unwrap(),
        "-R",
        "2",
        "--seed",
        "123",
        "--min-alt-reads",
        "2",
        "--min-depth-circles",
        "10",
        "--min-depth-duplex",
        "10",
    ])
    .assert()
    .success();

    // Check that all three output files exist
    let circles_file = PathBuf::from(format!("{}_circles.csv", output_prefix.display()));
    let duplex_file = PathBuf::from(format!("{}_duplex.csv", output_prefix.display()));
    let shared_file = PathBuf::from(format!("{}_shared.csv", output_prefix.display()));

    assert!(circles_file.exists(), "circles output file should exist");
    assert!(duplex_file.exists(), "duplex output file should exist");
    assert!(shared_file.exists(), "shared output file should exist");

    println!("\n=== circles.csv ===");
    let circles_content = fs::read_to_string(&circles_file).unwrap();
    for (i, line) in circles_content.lines().enumerate() {
        if i < 10 {
            println!("{}", line);
        }
    }

    println!("\n=== duplex.csv ===");
    let duplex_content = fs::read_to_string(&duplex_file).unwrap();
    for (i, line) in duplex_content.lines().enumerate() {
        if i < 10 {
            println!("{}", line);
        }
    }

    println!("\n=== shared.csv ===");
    let shared_content = fs::read_to_string(&shared_file).unwrap();
    for (i, line) in shared_content.lines().enumerate() {
        if i < 10 {
            println!("{}", line);
        }
    }

    // Verify file formats
    let circles_lines: Vec<&str> = circles_content.lines().collect();
    assert_eq!(circles_lines[0], "run,depth,alt_count", "circles header should be correct");
    
    let duplex_lines: Vec<&str> = duplex_content.lines().collect();
    assert_eq!(duplex_lines[0], "run,depth,alt_count", "duplex header should be correct");
    
    let shared_lines: Vec<&str> = shared_content.lines().collect();
    assert_eq!(shared_lines[0], "run,circles_depth,circles_alt,duplex_depth,duplex_alt", "shared header should be correct");
}

#[test]
fn test_multiple_runs() {
    setup();
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output_prefix = temp_dir.path().join("multi_run");

    let mut cmd = AssertCommand::cargo_bin("simulator").unwrap();
    cmd.args([
        "-n", "5000",
        "-N", "500000",
        "-b", "25",
        "-g", "100",
        "-u", "0.003",
        "--output-prefix", output_prefix.to_str().unwrap(),
        "-R", "5",  // 5 runs
        "--seed", "999",
    ])
    .assert()
    .success();

    let shared_file = PathBuf::from(format!("{}_shared.csv", output_prefix.display()));
    let shared_content = fs::read_to_string(&shared_file).unwrap();
    
    // Check that we have data from multiple runs (run indices 0-4)
    let has_run_0 = shared_content.contains("\n0,");
    let has_run_4 = shared_content.contains("\n4,");
    
    assert!(has_run_0, "Should have mutations from run 0");
    assert!(has_run_4, "Should have mutations from run 4");
    
    println!("\n=== Multi-run test summary ===");
    println!("Total lines: {}", shared_content.lines().count());
}