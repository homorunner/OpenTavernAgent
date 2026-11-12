use serde_json::json;
use tavernagent_v2::{
    actions,
    backend::Execution,
    batched_eval,
    eval::{self, Opponent},
    heroes,
    net::PolicyValueNet,
    observation, policy,
    replay::{JsonlWriter, Recorder},
    rollout,
    rules::PolicyRules,
    train::TrainOptions,
};
use taverngym::Game;

#[test]
fn recording_preserves_full_engine_state_rng_and_action_semantics() {
    let opts = TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 4,
        ..Default::default()
    };
    let net = PolicyValueNet::new(91, opts.net_config()).unwrap();
    for rules in [
        PolicyRules::default(),
        PolicyRules {
            freeze_at_end: true,
            ..Default::default()
        },
    ] {
        let mut original = Game::new(44, opts.game_config().unwrap());
        heroes::select_heroes(&mut original, 44).unwrap();
        let mut recorded = original.clone();
        let mut recorder = Recorder::new(&recorded);
        let mut rng = rollout::game_rng(44);
        let mut histories = [Default::default(); 8];
        while !original.is_over() {
            for (seat, history) in histories.iter_mut().enumerate() {
                if !original.needs_action(seat) {
                    continue;
                }
                let turn = original.turn;
                let obs = observation::observe_seat_with_rules(&original, seat, rules, history);
                let decision = policy::decide(&net, &obs, &mut rng, false).unwrap();
                actions::apply_with_rules(&mut original, seat, decision.action, rules).unwrap();
                recorder
                    .apply_model(&mut recorded, seat, decision, rules)
                    .unwrap();
                history.record(turn, decision.action);
                assert_eq!(
                    serde_json::to_value(&recorded).unwrap(),
                    serde_json::to_value(&original).unwrap()
                );
            }
        }
        let replay = recorder.finish(&recorded);
        assert!(
            replay["events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["type"] == "combat_start")
        );
        for event in replay["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["type"] == "combat_start")
        {
            assert_eq!(event["players"].as_array().unwrap().len(), 8);
            for p in event["players"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|p| p["alive"] == true)
            {
                assert_eq!(p["ended_turn"], true);
                assert!(p["board"].is_array());
            }
        }
    }
}

#[test]
fn full_replays_preserve_eval_and_are_independent_of_batch_and_workers() {
    let opts = TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 6,
        ..Default::default()
    };
    let net = PolicyValueNet::new(53, opts.net_config()).unwrap();
    let cfg = opts.game_config().unwrap();
    let rules = PolicyRules {
        freeze_at_end: true,
        ..Default::default()
    };
    for opponent in [Opponent::Scripted, Opponent::Teacher, Opponent::SelfPlay] {
        for sample in [false, true] {
            let mut first = None;
            for (workers, batch) in [(1, 1), (3, 2)] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                let results = batched_eval::evaluate_recorded(
                    &net,
                    &Execution::Cpu,
                    &[81, 82],
                    &cfg,
                    rules,
                    opponent,
                    sample,
                    batch,
                    &pool,
                    true,
                )
                .unwrap();
                let mut replays = Vec::new();
                for (seed, result) in [81, 82].into_iter().zip(results) {
                    let expected =
                        eval::play_detailed(&net, seed, &cfg, sample, opponent, rules).unwrap();
                    assert_eq!(result.placement, expected.placement);
                    assert_eq!(result.turn_limit_end, expected.turn_limit_end);
                    assert_eq!(result.truncated_combats, expected.truncated_combats);
                    assert_eq!(
                        serde_json::to_value(result.diagnostics).unwrap(),
                        serde_json::to_value(expected.diagnostics).unwrap()
                    );
                    let replay = result.replay.unwrap();
                    assert_eq!(replay["placements"].as_array().unwrap().len(), 8);
                    assert!(
                        replay["placements"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|v| v.as_u64().unwrap() > 0)
                    );
                    assert_eq!(
                        replay["events"].as_array().unwrap().last().unwrap()["type"],
                        "game_end"
                    );
                    replays.push(replay);
                }
                if let Some(expected) = &first {
                    assert_eq!(&replays, expected);
                } else {
                    first = Some(replays);
                }
            }
        }
    }
}

#[test]
fn replay_finishes_lobby_after_evaluated_seat_is_eliminated() {
    let opts = TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 40,
        ..Default::default()
    };
    let net = PolicyValueNet::new(53, opts.net_config()).unwrap();
    let cfg = opts.game_config().unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let game = batched_eval::evaluate_recorded(
        &net,
        &Execution::Cpu,
        &[81],
        &cfg,
        PolicyRules::default(),
        Opponent::Teacher,
        false,
        1,
        &pool,
        true,
    )
    .unwrap()
    .pop()
    .unwrap();
    let expected = eval::play_detailed(
        &net,
        81,
        &cfg,
        false,
        Opponent::Teacher,
        PolicyRules::default(),
    )
    .unwrap();
    assert_eq!(game.placement, expected.placement);
    assert!(game.placement > 2);
    let replay = game.replay.unwrap();
    let events = replay["events"].as_array().unwrap();
    let dead = events
        .iter()
        .position(|e| e["type"] == "combat_end" && e["players"][0]["alive"] == false)
        .unwrap();
    assert!(events[dead + 1..].iter().any(|e| e["type"] == "action"));
    assert_eq!(events.last().unwrap()["type"], "game_end");
    assert!(!replay["placements"].as_array().unwrap().contains(&json!(0)));
}

#[test]
fn jsonl_publish_is_complete_and_never_overwrites() {
    let root = std::env::temp_dir().join(format!("tavern-replay-writer-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let p = root.join("games.jsonl");
    let model = root.join("model.npz");
    std::fs::write(&model, b"model").unwrap();
    assert!(
        tavernagent_v2::replay::validate_paths(&model, Some(&root.join("./games.jsonl")), &p)
            .is_err()
    );
    assert!(
        tavernagent_v2::replay::validate_paths(&model, Some(&root.join("games.jsonl.partial")), &p)
            .is_err()
    );
    assert!(tavernagent_v2::replay::validate_paths(&model, Some(&model), &p).is_err());
    assert!(
        tavernagent_v2::replay::validate_paths(&model, Some(&root.join("summary.json")), &p)
            .is_ok()
    );
    {
        let mut writer = JsonlWriter::new(&p).unwrap();
        writer.write(&json!({"game":1})).unwrap();
    }
    assert!(!p.exists());
    assert!(!root.join("games.jsonl.partial").exists());
    let mut writer = JsonlWriter::new(&p).unwrap();
    writer.write(&json!({"game":1})).unwrap();
    writer.write(&json!({"game":2})).unwrap();
    writer.finish().unwrap();
    assert_eq!(std::fs::read_to_string(&p).unwrap().lines().count(), 2);
    assert!(JsonlWriter::new(&p).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
