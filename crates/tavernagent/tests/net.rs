//! Network: masked policy, card identity handling, hand-written gradients, checkpoints.

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use tavernagent::linalg::Mat;
use tavernagent::mcts::observe_seat;
use tavernagent::net::{
    self, Activation, N_ACTIONS, N_CARD_SLOTS, N_FEATURES, NetConfig, PolicyValueNet,
    allowed_cards, masked_softmax,
};
use tavernagent::npz::Npz;
use taverngym::Game;
use taverngym::cards::{CardKind, db};

fn position(seed: u64) -> Game {
    let mut g = Game::new(seed, Default::default());
    tavernagent::heroes::select_heroes(&mut g, seed).unwrap();
    g
}

fn cfg(activation: Activation, head_norm: bool) -> NetConfig {
    NetConfig {
        embedding_dim: 4,
        activation,
        residual_scale: 0.7,
        head_norm,
        ..NetConfig::default()
    }
}

#[test]
fn parameter_count_and_input_layout() {
    let net = PolicyValueNet::new(0, NetConfig::default()).unwrap();
    assert_eq!(net.n_input_features(), 1713);
    assert_eq!(net.n_card_slots(), 69);
    assert_eq!(net.n_params(), 2_378_357);
    assert_eq!(net.p.tensors.len(), 9 + 6 * net.cfg.blocks);
    assert_eq!(net.p.names().last().unwrap(), "bv");
    let wide = PolicyValueNet::new(
        0,
        NetConfig {
            hidden: 256,
            blocks: 2,
            ..NetConfig::default()
        },
    )
    .unwrap();
    assert_eq!(wide.p.tensors.len(), 9 + 6 * 2);
    assert_eq!(wide.p.tensors[wide.p.wp()].rows, 256);
    assert!(wide.p.is_weight(wide.p.wv()));
    assert!(wide.p.is_weight(net::w_up(0)) && wide.p.is_weight(net::w_down(0)));
    assert!(!wide.p.is_weight(wide.p.bp()) && !wide.p.is_weight(wide.p.bv()));
    assert!(net::input_feature_indices().contains(&net::ENDED_TURN_FEATURE));
    assert!(!net::input_feature_indices().contains(&1));
    assert!(!net::input_feature_indices().contains(&15));
    let slots = net::card_slot_indices();
    assert!(
        !slots.contains(&3) && !slots.contains(&14) && !slots.contains(&18) && slots.contains(&19)
    );
}

#[test]
fn masked_softmax_respects_the_mask_and_empty_masks_are_zero() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let p = masked_softmax(&logits, &[true, false, true, false]);
    assert_eq!(p[1], 0.0);
    assert_eq!(p[3], 0.0);
    assert!((p[0] + p[2] - 1.0).abs() < 1e-6);
    assert!(p[2] > p[0]);
    assert!(
        masked_softmax(&logits, &[false; 4])
            .iter()
            .all(|v| *v == 0.0)
    );
}

#[test]
fn card_identity_matters_and_ignored_slots_do_not() {
    let g = position(17);
    let obs = observe_seat(&g, 0);
    let net = PolicyValueNet::new(2, NetConfig::default()).unwrap();
    for kind in [
        CardKind::Minion,
        CardKind::TavernSpell,
        CardKind::Hero,
        CardKind::HeroPower,
        CardKind::Quest,
    ] {
        let ids: Vec<i32> = db()
            .iter()
            .filter(|c| c.kind == kind)
            .map(|c| c.cid.0 as i32 + 1)
            .take(2)
            .collect();
        let mut a = vec![0i32; N_CARD_SLOTS];
        let mut b = a.clone();
        a[30] = ids[0];
        b[30] = ids[1];
        let (pa, va) = net.predict(&obs.x, &a, &obs.mask).unwrap();
        let (pb, vb) = net.predict(&obs.x, &b, &obs.mask).unwrap();
        assert!(pa != pb || va != vb, "{kind:?} identity ignored");
    }
    let empty = vec![0i32; N_CARD_SLOTS];
    let mut ignored = empty.clone();
    ignored[3] = db()
        .iter()
        .find(|c| c.kind == CardKind::Hero)
        .unwrap()
        .cid
        .0 as i32
        + 1;
    ignored[14] = db()
        .iter()
        .find(|c| c.kind == CardKind::QuestReward)
        .unwrap()
        .cid
        .0 as i32
        + 1;
    assert_eq!(
        net.predict(&obs.x, &empty, &obs.mask).unwrap(),
        net.predict(&obs.x, &ignored, &obs.mask).unwrap()
    );
    let mut dark_gift = empty.clone();
    dark_gift[30] = db()
        .iter()
        .find(|c| c.kind == CardKind::DarkGift)
        .unwrap()
        .cid
        .0 as i32
        + 1;
    assert_eq!(
        net.predict(&obs.x, &empty, &obs.mask).unwrap(),
        net.predict(&obs.x, &dark_gift, &obs.mask).unwrap()
    );
    let mut bad = empty.clone();
    bad[30] = db().len() as i32 + 1;
    assert!(net.predict(&obs.x, &bad, &obs.mask).is_err());
}

/// Central-difference check of every tensor kind, with a repeated card and an ended row.
fn gradient_check(activation: Activation, head_norm: bool) {
    let mut net = PolicyValueNet::new(3, cfg(activation, head_norm)).unwrap();
    let mut rng = Pcg64Mcg::seed_from_u64(4);
    let g = position(23);
    let real = observe_seat(&g, 0);
    let mut ended_game = g.clone();
    ended_game.end_turn_effects(0).unwrap();
    let ended = observe_seat(&ended_game, 0);
    let noisy: Vec<f32> = (0..N_FEATURES)
        .map(|_| rng.random_range(-0.3f32..0.3))
        .collect();
    let cid = allowed_cards().iter().position(|a| *a).unwrap() as i32;
    let mut repeated = vec![0i32; N_CARD_SLOTS];
    repeated[19..22].fill(cid);
    let xs: Vec<&[f32]> = vec![&real.x, &ended.x, &noisy];
    let cards: Vec<&[i32]> = vec![&real.cards, &ended.cards, &repeated];
    let masks: Vec<&[bool]> = vec![&real.mask, &ended.mask, &real.mask];
    let dlogits = Mat::from_vec(
        3,
        N_ACTIONS,
        (0..3 * N_ACTIONS)
            .map(|_| rng.random_range(-0.01f32..0.01))
            .collect(),
    );
    let dvalue_raw = [0.03f32, -0.02, 0.05];
    let (_, value, cache) = net.forward(&xs, &cards, &masks).unwrap();
    let dvalue: Vec<f32> = (0..3)
        .map(|i| dvalue_raw[i] * (1.0 - value[i] * value[i]))
        .collect();
    let grads = net.backward(&cache, &dlogits, &dvalue);
    assert!(grads.tensors[net::E].row(0).iter().all(|v| *v == 0.0));
    let loss = |net: &PolicyValueNet| -> f64 {
        let (logits, value, _) = net.forward(&xs, &cards, &masks).unwrap();
        let mut total = 0.0f64;
        for (l, d) in logits.data.iter().zip(&dlogits.data) {
            total += *l as f64 * *d as f64;
        }
        for (v, d) in value.iter().zip(&dvalue_raw) {
            total += *v as f64 * *d as f64;
        }
        total
    };
    let slot_of_19 = net::card_slot_indices()
        .iter()
        .position(|s| *s == 19)
        .unwrap();
    let names = net.p.names();
    let mut probes: Vec<(usize, usize)> = Vec::new();
    for (i, t) in net.p.tensors.iter().enumerate() {
        let n = t.len();
        probes.push((i, rng.random_range(0..n)));
        probes.push((i, rng.random_range(0..n)));
    }
    probes.push((net::E, cid as usize * net.cfg.embedding_dim + 1));
    probes.push((
        net::W0,
        (net.n_input_features() + slot_of_19 * net.cfg.embedding_dim) * net.cfg.hidden + 5,
    ));
    probes.push((
        net::W0,
        (net.n_input_features()
            + net.n_card_slots() * net.cfg.embedding_dim
            + real.mask.iter().position(|m| *m).unwrap())
            * net.cfg.hidden
            + 5,
    ));
    probes.push((net::E, 0)); // the empty row stays zero
    let eps = 2e-3f32;
    for (i, j) in probes {
        if i == net::E && j / net.cfg.embedding_dim == 0 {
            assert_eq!(grads.tensors[i].data[j], 0.0);
            continue;
        }
        let old = net.p.tensors[i].data[j];
        net.p.tensors[i].data[j] = old + eps;
        let hi = loss(&net);
        net.p.tensors[i].data[j] = old - eps;
        let lo = loss(&net);
        net.p.tensors[i].data[j] = old;
        let numeric = (hi - lo) / (2.0 * eps as f64);
        let analytic = grads.tensors[i].data[j] as f64;
        let tol = 3e-3 + 0.03 * numeric.abs();
        assert!(
            (numeric - analytic).abs() < tol,
            "{} ({activation:?}, head_norm={head_norm}) [{j}]: numeric {numeric:.6} analytic {analytic:.6}",
            names[i]
        );
    }
}

#[test]
fn search_inference_matches_the_training_forward() {
    for (activation, head_norm) in [
        (Activation::Gelu, false),
        (Activation::Silu, true),
        (Activation::Relu, true),
    ] {
        let net = PolicyValueNet::new(11, cfg(activation, head_norm)).unwrap();
        for seed in [3u64, 4, 5] {
            let game = position(seed);
            for seat in 0..2 {
                let obs = observe_seat(&game, seat);
                let (logits, value, _) =
                    net.forward(&[&obs.x], &[&obs.cards], &[&obs.mask]).unwrap();
                // The f32 path repeats the training arithmetic exactly.
                let (exact_logits, exact_value) =
                    net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap();
                assert_eq!(logits.row(0), &exact_logits[..]);
                assert_eq!(value[0], exact_value);
                // The search path (bfloat16 weights, polynomial tanh) stays close to it.
                let (fast_logits, fast_value) = net.infer(&obs.x, &obs.cards, &obs.mask).unwrap();
                let span = logits
                    .row(0)
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()))
                    .max(1.0);
                for (a, b) in exact_logits.iter().zip(&fast_logits) {
                    assert!((a - b).abs() < 0.01 * span, "{a} vs {b}");
                }
                assert!(
                    (exact_value - fast_value).abs() < 0.01,
                    "{exact_value} vs {fast_value}"
                );
                let (policy, v) = net.predict(&obs.x, &obs.cards, &obs.mask).unwrap();
                assert_eq!(policy, masked_softmax(&fast_logits, &obs.mask));
                assert_eq!(v, fast_value);
            }
        }
    }
}

#[test]
fn search_inference_follows_parameter_updates() {
    let mut net = PolicyValueNet::new(2, cfg(Activation::Gelu, false)).unwrap();
    let game = position(8);
    let obs = observe_seat(&game, 0);
    let before = net.infer(&obs.x, &obs.cards, &obs.mask).unwrap();
    let wp = net.p.wp();
    for v in &mut net.p.tensors[wp].data {
        *v += 0.5;
    }
    // Without the notification the bfloat16 copy is stale; with it the change is visible.
    assert_eq!(
        net.infer(&obs.x, &obs.cards, &obs.mask).unwrap().0,
        before.0
    );
    net.params_changed();
    let after = net.infer(&obs.x, &obs.cards, &obs.mask).unwrap();
    assert_ne!(after.0, before.0);
    let (exact, _) = net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap();
    for (a, b) in exact.iter().zip(&after.0) {
        assert!((a - b).abs() < 0.005 * a.abs().max(1.0), "{a} vs {b}");
    }
}

#[test]
fn gradients_match_central_differences() {
    gradient_check(Activation::Gelu, false);
    gradient_check(Activation::Silu, true);
    gradient_check(Activation::Relu, false);
}

#[test]
fn checkpoints_round_trip_and_reject_other_schemas() {
    let dir = std::env::temp_dir().join(format!("tavernagent-net-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.npz");
    let net = PolicyValueNet::new(5, cfg(Activation::Silu, true)).unwrap();
    net.save(&path).unwrap();
    let restored = PolicyValueNet::load(&path).unwrap();
    assert_eq!(restored.p, net.p);
    assert_eq!(restored.cfg, net.cfg);
    let g = position(9);
    let obs = observe_seat(&g, 0);
    assert_eq!(
        net.predict(&obs.x, &obs.cards, &obs.mask).unwrap(),
        restored.predict(&obs.x, &obs.cards, &obs.mask).unwrap()
    );
    // Numpy-compatible layout: matrices are row-major and bias/norm vectors are 1-D.
    let npz = Npz::load(&path).unwrap();
    assert_eq!(
        npz.get("W0").unwrap().shape,
        vec![1713 + 69 * 4 + N_ACTIONS, net.cfg.hidden]
    );
    assert_eq!(npz.get("bp").unwrap().shape, vec![N_ACTIONS]);
    for version in [2, 9] {
        let mut wrong = npz.clone();
        let meta = wrong.get("meta").unwrap().as_text().unwrap().replace(
            &format!("\"version\":{}", net::MODEL_VERSION),
            &format!("\"version\":{version}"),
        );
        wrong.insert("meta", tavernagent::npz::NpyArray::text(&meta));
        assert!(
            PolicyValueNet::from_npz(&wrong)
                .unwrap_err()
                .contains(&format!("v{version}"))
        );
    }
    let mut other_cards = npz.clone();
    let meta = other_cards.get("meta").unwrap().as_text().unwrap();
    let meta = meta.replace(
        &format!("\"card_fingerprint\":\"{}\"", net.card_fingerprint),
        "\"card_fingerprint\":\"1\"",
    );
    other_cards.insert("meta", tavernagent::npz::NpyArray::text(&meta));
    assert!(
        PolicyValueNet::from_npz(&other_cards)
            .unwrap_err()
            .contains("schema")
    );
    let mut missing = npz.clone();
    missing.arrays.remove("W2_down");
    assert!(PolicyValueNet::from_npz(&missing).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn mask_is_a_shared_input_and_invalid_mask_shapes_are_rejected() {
    let net = PolicyValueNet::new(23, cfg(Activation::Gelu, false)).unwrap();
    let obs = observe_seat(&position(23), 0);
    let zeros = vec![false; N_ACTIONS];
    assert_ne!(obs.mask, zeros);
    // Same board and ended flag: changing only legality changes raw logits AND value,
    // before the output softmax ever sees a mask.
    let (logits, value) = net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap();
    let (empty_logits, empty_value) = net.infer_f32(&obs.x, &obs.cards, &zeros).unwrap();
    assert_ne!(logits, empty_logits);
    assert!((value - empty_value).abs() > 1e-6);
    let (policy, value) = net.predict(&obs.x, &obs.cards, &zeros).unwrap();
    assert!(policy.iter().all(|p| *p == 0.0));
    assert!(value.is_finite());
    assert!(
        net.predict(&obs.x, &obs.cards, &zeros[..N_ACTIONS - 1])
            .is_err()
    );
    assert!(net.forward(&[&obs.x], &[&obs.cards], &[]).is_err());
    assert!(
        net.forward(&[&obs.x], &[&obs.cards], &[&zeros[..N_ACTIONS - 1]])
            .is_err()
    );
}

#[test]
fn reusable_workspaces_match_reference_across_depths_models_and_threads() {
    use rayon::prelude::*;
    use tavernagent::net::{InferenceBackend, InferenceWorkspace};
    let mut positions = Vec::new();
    for seed in 0..3 {
        let mut game = position(seed);
        positions.push(observe_seat(&game, 0));
        if game.end_turn_effects(0).is_ok() {
            positions.push(observe_seat(&game, 0));
        }
    }
    let mut ws = InferenceWorkspace::default();
    for blocks in [6, 8, 10, 12] {
        let net = PolicyValueNet::new(
            90 + blocks as u64,
            NetConfig {
                blocks,
                ..Default::default()
            },
        )
        .unwrap();
        for obs in &positions {
            let (exact, ev) = net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap();
            for backend in [InferenceBackend::Fma, InferenceBackend::Dot] {
                if backend == InferenceBackend::Dot && !tavernagent::linalg::native_bf16_available()
                {
                    continue;
                }
                let (logits, value) = net
                    .infer_with_workspace(&obs.x, &obs.cards, &obs.mask, backend, &mut ws)
                    .unwrap();
                assert!(
                    (value - ev).abs() < 0.01,
                    "depth={blocks} {backend:?}: {value} vs {ev}"
                );
                for (a, b) in exact.iter().zip(logits) {
                    assert!(
                        (a - b).abs() < 0.01,
                        "depth={blocks} {backend:?}: {a} vs {b}"
                    );
                }
            }
        }
        let sequential: Vec<_> = positions
            .iter()
            .map(|o| net.predict(&o.x, &o.cards, &o.mask).unwrap())
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel: Vec<_> = pool.install(|| {
            positions
                .par_iter()
                .map(|o| net.predict(&o.x, &o.cards, &o.mask).unwrap())
                .collect()
        });
        assert_eq!(sequential, parallel);
        // A failed call must not contaminate the next call on the same workspace.
        let obs = &positions[0];
        assert!(
            net.infer_with_workspace(
                &obs.x[..1],
                &obs.cards,
                &obs.mask,
                InferenceBackend::Fma,
                &mut ws
            )
            .is_err()
        );
        let mut clean = InferenceWorkspace::default();
        let (a, va) = net
            .infer_with_workspace(
                &obs.x,
                &obs.cards,
                &obs.mask,
                InferenceBackend::Fma,
                &mut ws,
            )
            .unwrap();
        let (b, vb) = net
            .infer_with_workspace(
                &obs.x,
                &obs.cards,
                &obs.mask,
                InferenceBackend::Fma,
                &mut clean,
            )
            .unwrap();
        assert_eq!((a, va), (b, vb));
    }
    // Shrink all scratch dimensions and exercise SIMD tails after the large models.
    let small = PolicyValueNet::new(
        1,
        NetConfig {
            hidden: 17,
            embedding_dim: 3,
            blocks: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let o = &positions[0];
    let mut fresh = InferenceWorkspace::default();
    let (a, va) = small
        .infer_with_workspace(&o.x, &o.cards, &o.mask, InferenceBackend::Auto, &mut ws)
        .unwrap();
    let (b, vb) = small
        .infer_with_workspace(&o.x, &o.cards, &o.mask, InferenceBackend::Auto, &mut fresh)
        .unwrap();
    assert_eq!((a, va), (b, vb));
}
