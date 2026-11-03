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
    assert_eq!(net.n_input_features(), 1716);
    assert_eq!(net.n_card_slots(), 69);
    assert_eq!(net.n_params(), 803_191);
    assert_eq!(net.p.tensors.len(), 11 + 4 * 4);
    assert_eq!(net.p.names().last().unwrap(), "bc");
    let wide = PolicyValueNet::new(
        0,
        NetConfig {
            hidden: 256,
            blocks: 2,
            ..NetConfig::default()
        },
    )
    .unwrap();
    assert_eq!(wide.p.tensors.len(), 11 + 4 * 2);
    assert_eq!(wide.p.tensors[wide.p.wp()].rows, 256);
    assert!(wide.p.is_weight(wide.p.wv()) && wide.p.is_weight(wide.p.wc()));
    assert!(!wide.p.is_weight(wide.p.bp()) && !wide.p.is_weight(wide.p.bc()));
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
    let dlogits = Mat::from_vec(
        3,
        N_ACTIONS,
        (0..3 * N_ACTIONS)
            .map(|_| rng.random_range(-0.01f32..0.01))
            .collect(),
    );
    let dvalue_raw = [0.03f32, -0.02, 0.05];
    let dcombat_raw = [-0.04f32, 0.01, 0.02];
    let (_, value, combat, cache) = net.forward(&xs, &cards).unwrap();
    let dvalue: Vec<f32> = (0..3)
        .map(|i| dvalue_raw[i] * (1.0 - value[i] * value[i]))
        .collect();
    let dcombat: Vec<f32> = (0..3)
        .map(|i| dcombat_raw[i] * (1.0 - combat[i] * combat[i]))
        .collect();
    let grads = net.backward(&cache, &dlogits, &dvalue, &dcombat);
    assert!(grads.tensors[net::E].row(0).iter().all(|v| *v == 0.0));
    let loss = |net: &PolicyValueNet| -> f64 {
        let (logits, value, combat, _) = net.forward(&xs, &cards).unwrap();
        let mut total = 0.0f64;
        for (l, d) in logits.data.iter().zip(&dlogits.data) {
            total += *l as f64 * *d as f64;
        }
        for (v, d) in value.iter().zip(&dvalue_raw) {
            total += *v as f64 * *d as f64;
        }
        for (c, d) in combat.iter().zip(&dcombat_raw) {
            total += *c as f64 * *d as f64;
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
    // Numpy-compatible layout: the arrays carry the Python-era names and 1-D vectors.
    let npz = Npz::load(&path).unwrap();
    assert_eq!(npz.get("W0").unwrap().shape, vec![1716 + 69 * 4, 192]);
    assert_eq!(npz.get("bp").unwrap().shape, vec![N_ACTIONS]);
    let mut wrong = npz.clone();
    let meta = wrong
        .get("meta")
        .unwrap()
        .as_text()
        .unwrap()
        .replace("\"version\":4", "\"version\":2");
    wrong.insert("meta", tavernagent::npz::NpyArray::text(&meta));
    assert!(PolicyValueNet::from_npz(&wrong).unwrap_err().contains("v2"));
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
    missing.arrays.remove("W2");
    assert!(PolicyValueNet::from_npz(&missing).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
