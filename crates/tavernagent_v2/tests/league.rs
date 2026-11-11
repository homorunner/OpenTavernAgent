use rayon::prelude::*;
use tavernagent_v2::{
    arena,
    league::{self, LeagueModel},
    net::PolicyValueNet,
    rules::PolicyRules,
    train::TrainOptions,
};

#[test]
fn mixed_checkpoint_league_matches_arena_rules_and_is_worker_independent() {
    let opts = TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 5,
        ..Default::default()
    };
    let cfg = opts.game_config().unwrap();
    let models: Vec<_> = (0..8)
        .map(|i| LeagueModel {
            net: PolicyValueNet::new(if i < 4 { 71 } else { 93 }, opts.net_config()).unwrap(),
            rules: PolicyRules {
                block_consecutive_freeze: i < 4,
            },
        })
        .collect();
    let run = |workers| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| {
                (0..8)
                    .into_par_iter()
                    .map(|offset| {
                        let table: Vec<_> = (0..8).map(|s| (s + offset) % 8).collect();
                        league::play(&models, &table, &cfg, 17 + offset as u64).unwrap()
                    })
                    .collect::<Vec<_>>()
            })
    };
    let actual = run(1);
    assert_eq!(actual, run(4));
    for (offset, result) in actual.into_iter().enumerate() {
        let expected = arena::play(
            &models[0].net,
            &models[4].net,
            &cfg,
            models[0].rules,
            models[4].rules,
            17 + offset as u64,
            offset,
        )
        .unwrap();
        assert_eq!(result.placements, expected.placements);
        assert_eq!(result.turn_limit_end, expected.turn_limit_end);
        assert_eq!(result.truncated_combats, expected.truncated_combats);
    }
}

#[test]
fn league_checkpoint_cli_publishes_complete_rankings_without_overwriting() {
    use tavernagent_v2::train::{self, TrainArgs};
    let root = std::env::temp_dir().join(format!("ppo-league-cli-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    for (name, hidden, turns) in [("a", 16, 3), ("b", 24, 3), ("incompatible", 16, 4)] {
        train::train(TrainArgs {
            device: 0,
            iterations: 0,
            workers: 1,
            out: root.join(format!("{name}.npz")),
            save_every: 0,
            options: TrainOptions {
                hidden,
                blocks: 1,
                embedding_dim: 2,
                max_turns: turns,
                ..Default::default()
            },
        })
        .unwrap();
    }
    let args = |out: &str, bad: bool| league::LeagueArgs {
        models: (0..8)
            .map(|i| {
                format!(
                    "m{i}={}",
                    root.join(format!(
                        "{}.npz",
                        if bad && i == 7 {
                            "incompatible"
                        } else if i % 2 == 0 {
                            "a"
                        } else {
                            "b"
                        }
                    ))
                    .display()
                )
            })
            .collect(),
        rounds: 2,
        tables: None,
        seed: 42,
        workers: 2,
        out: root.join(out),
    };
    league::run(args("games.jsonl", false)).unwrap();
    let original = std::fs::read(root.join("games.jsonl")).unwrap();
    let rows: Vec<serde_json::Value> = String::from_utf8(original.clone())
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row["models"].as_array().unwrap().len(), 8);
        assert_eq!(row["placements"].as_array().unwrap().len(), 8);
        assert_eq!(row["backend"], "cpu");
        assert_eq!(row["sample"], false);
    }
    assert!(
        league::run(args("games.jsonl", false))
            .unwrap_err()
            .contains("already exists")
    );
    assert_eq!(std::fs::read(root.join("games.jsonl")).unwrap(), original);
    assert!(
        league::run(args("bad.jsonl", true))
            .unwrap_err()
            .contains("identical engine rules")
    );
    assert!(!root.join("bad.jsonl").exists());
    std::fs::remove_dir_all(root).unwrap();
}
