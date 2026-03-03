//! Verify all shipped nuclei firmware programs parse correctly
//! and declare the :nuclei namespace.

use runes_core::ast::{Expr, Program};
use runes_parser::{Lexer, Parser};

const FIRMWARE_DIR: &str = "firmware/nuclei";

const EXPECTED_PROGRAMS: &[&str] = &[
    "pacemaker",
    "selective_gate",
    "memory_access",
    "novelty_detect",
    "sensory_transduce",
    "motor_integrate",
    "multimodal_bind",
    "similarity_detect",
];

fn parse_rune_file(path: &std::path::Path) -> Program {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    let tokens = Lexer::new(&source)
        .tokenize()
        .unwrap_or_else(|e| panic!("Lexer error in {}: {}", path.display(), e.message));
    Parser::new(tokens)
        .parse_program()
        .unwrap_or_else(|e| panic!("Parse error in {}: {}", path.display(), e.message))
}

fn extract_namespace(program: &Program) -> Option<String> {
    for entry in &program.metadata.entries {
        if entry.key == "namespace" {
            if let Some(val) = entry.values.first() {
                if let Expr::Symbol(s) = &val.node {
                    return Some(s.clone());
                }
            }
        }
    }
    None
}

#[test]
fn all_nuclei_programs_parse() {
    let dir = std::path::Path::new(FIRMWARE_DIR);
    assert!(dir.is_dir(), "firmware/nuclei directory must exist");

    let mut found = Vec::new();

    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rune") {
            continue;
        }

        let program = parse_rune_file(&path);
        let name = &program.metadata.name;

        // Verify namespace is :nuclei
        let ns = extract_namespace(&program);
        assert_eq!(
            ns.as_deref(),
            Some("nuclei"),
            "Program '{}' in {} must declare namespace :nuclei, got {:?}",
            name, path.display(), ns,
        );

        found.push(name.clone());
    }

    // Verify all expected programs exist
    found.sort();
    let mut expected: Vec<String> = EXPECTED_PROGRAMS.iter().map(|s| s.to_string()).collect();
    expected.sort();

    assert_eq!(
        found, expected,
        "Firmware directory should contain exactly the expected programs.\nFound: {:?}\nExpected: {:?}",
        found, expected,
    );
}

#[test]
fn each_program_has_use_statements() {
    let dir = std::path::Path::new(FIRMWARE_DIR);

    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rune") {
            continue;
        }

        let program = parse_rune_file(&path);

        // Every nuclei program should use at least :signal or :cascade
        assert!(
            !program.uses.is_empty(),
            "Program '{}' in {} has no use statements — nuclei programs must import modules",
            program.metadata.name, path.display(),
        );
    }
}
