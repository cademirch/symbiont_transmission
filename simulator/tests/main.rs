use assert_cmd::Command as AssertCommand;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use tempfile::TempDir;

#[test]
fn without_depth() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output_dir = temp_dir.path().join("test_output");

    let mut cmd = AssertCommand::cargo_bin("simulator").unwrap();
    cmd.args([
        "-n",
        "10000",
        "-N",
        "100000000",
        "-b",
        "100",
        "-g",
        "1000",
        "-u",
        "0.012",
        "--output-dir",
        output_dir.to_str().unwrap(),
        "-R",
        "2",
        "--seed",
        "42",
    ])
    .assert()
    .success();

    assert!(output_dir.join("all_mutations.csv").exists());
    assert!(output_dir.join("config.json").exists());

    assert!(!output_dir.join("subsampled_mutations.csv").exists());

    println!("\n=== all_mutations.csv ===");
    let mutations_content = fs::read_to_string(output_dir.join("all_mutations.csv")).unwrap();
    for (i, line) in mutations_content.lines().enumerate() {
        if i < 10 {
            // Print first 10 lines
            println!("{}", line);
        }
    }

    println!("\n=== config.json ===");
    let config_content = fs::read_to_string(output_dir.join("config.json")).unwrap();
    println!("{}", config_content);
}

#[test]
fn with_depth() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output_dir = temp_dir.path().join("test_output");

    let depth_file = PathBuf::from_str("tests/depths.bed").unwrap();

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
        "--output-dir",
        output_dir.to_str().unwrap(),
        "-R",
        "2",
        "--seed",
        "123",
        "--depth-files",
        depth_file.to_str().unwrap(),
        "--min-alt-reads",
        "2",
        "--min-depth",
        "10",
        "--bin-size",
        "0.05",
    ])
    .assert()
    .success();

    assert!(output_dir.join("all_mutations.csv").exists());
    assert!(output_dir.join("subsampled_mutations.csv").exists());
    assert!(output_dir.join("config.json").exists());

    println!("\n=== all_mutations.csv ===");
    let mutations_content = fs::read_to_string(output_dir.join("all_mutations.csv")).unwrap();
    for (i, line) in mutations_content.lines().enumerate() {
        if i < 10 {
            println!("{}", line);
        }
    }

    println!("\n=== subsampled_mutations.csv ===");
    let subsampled_content =
        fs::read_to_string(output_dir.join("subsampled_mutations.csv")).unwrap();
    for (i, line) in subsampled_content.lines().enumerate() {
        if i < 10 {
            println!("{}", line);
        }
    }

    println!("\n=== config.json ===");
    let config_content = fs::read_to_string(output_dir.join("config.json")).unwrap();
    println!("{}", config_content);
}
