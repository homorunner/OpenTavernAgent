//! Structural, inference and numerical-gradient checks; no optimizer updates.
use tavernagent_v2::{
    heroes,
    linalg::Mat,
    net::{self, Architecture, NetConfig, PolicyValueNet},
    observation::{Observation, observe_seat},
};
use taverngym::{Game, cards::cid};

fn config(architecture: Architecture) -> NetConfig {
    NetConfig {
        architecture,
        hidden: 16,
        blocks: 2,
        embedding_dim: 4,
        ..Default::default()
    }
}

fn observations() -> Vec<Observation> {
    let mut game = Game::new(91, Default::default());
    heroes::select_heroes(&mut game, 91).unwrap();
    let mut rows = vec![observe_seat(&game, 0), observe_seat(&game, 1)];
    // Populate every entity type, including meta slots and empty board/hand inputs,
    // to exercise every trainable path independently of rollout exploration.
    for (r, obs) in rows.iter_mut().enumerate() {
        for (j, value) in obs.x.iter_mut().enumerate() {
            *value = ((j * 7 + r * 3) % 13 + 1) as f32 / 20.0;
        }
        for (j, card) in obs.cards.iter_mut().enumerate() {
            *card = if (j + r) % 3 == 0 {
                0
            } else {
                cid("BGS_029").idx() as i32 + 1
            };
        }
        obs.mask.fill(true);
    }
    rows
}

#[test]
fn entity_parameter_budgets_and_checkpoint_roundtrip() {
    {
        let architecture = Architecture::EntityActions;
        let net = PolicyValueNet::new(
            37,
            NetConfig {
                architecture,
                hidden: 320,
                ..Default::default()
            },
        )
        .unwrap();
        println!("{architecture:?}: {} parameters", net.n_params());
        assert_eq!(net.n_params(), 3_473_038);
        assert_eq!(net.p.names().len(), net.p.tensors.len());
        let restored = PolicyValueNet::from_npz(&net.to_npz()).unwrap();
        assert_eq!(restored.cfg, net.cfg);
        assert_eq!(restored.p, net.p);
    }
    // Unsupported historical layouts must fail explicitly, including missing metadata.
    for architecture in [Some("flat"), Some("entity"), None] {
        let net = PolicyValueNet::new(7, config(Architecture::EntityActions)).unwrap();
        let mut npz = net.to_npz();
        let mut meta: serde_json::Value =
            serde_json::from_str(&npz.get("meta").unwrap().as_text().unwrap()).unwrap();
        if let Some(value) = architecture {
            meta["architecture"] = value.into();
        } else {
            meta.as_object_mut().unwrap().remove("architecture");
        }
        npz.insert(
            "meta",
            tavernagent_v2::npz::NpyArray::text(&meta.to_string()),
        );
        assert!(PolicyValueNet::from_npz(&npz).is_err());
    }
}

#[test]
fn entity_single_and_batch_inference_match_and_all_gradients_are_correct() {
    let rows = observations();
    let xs: Vec<_> = rows.iter().map(|o| o.x.as_slice()).collect();
    let cards: Vec<_> = rows.iter().map(|o| o.cards.as_slice()).collect();
    let masks: Vec<_> = rows.iter().map(|o| o.mask.as_slice()).collect();
    {
        let architecture = Architecture::EntityActions;
        let mut net = PolicyValueNet::new(39, config(architecture)).unwrap();
        let (logits, values, cache) = net.forward(&xs, &cards, &masks).unwrap();
        for (r, obs) in rows.iter().enumerate() {
            let (l, v) = net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap();
            for (a, b) in l.iter().zip(logits.row(r)) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
            assert!((v - values[r]).abs() < 1e-5);
        }
        let dl = Mat::from_vec(
            2,
            net::N_ACTIONS,
            (0..2 * net::N_ACTIONS)
                .map(|i| ((i * 7 % 19) as f32 - 9.0) / 100.0)
                .collect(),
        );
        let dv: Vec<_> = values.iter().map(|v| 0.3 * (1.0 - v * v)).collect();
        let grads = net.backward(&cache, &dl, &dv);
        assert!(grads.all_finite());
        let objective = |net: &PolicyValueNet| {
            let (l, v, _) = net.forward(&xs, &cards, &masks).unwrap();
            l.data
                .iter()
                .zip(&dl.data)
                .map(|(x, d)| f64::from(*x) * f64::from(*d))
                .sum::<f64>()
                + v.iter().map(|x| f64::from(*x) * 0.3).sum::<f64>()
        };
        for (i, grad) in grads.tensors.iter().enumerate() {
            let j = (0..grad.len())
                .max_by(|&a, &b| grad.data[a].abs().total_cmp(&grad.data[b].abs()))
                .unwrap();
            let original = net.p.tensors[i].data[j];
            let eps = 0.002;
            net.p.tensors[i].data[j] = original + eps;
            let hi = objective(&net);
            net.p.tensors[i].data[j] = original - eps;
            let lo = objective(&net);
            net.p.tensors[i].data[j] = original;
            let numeric = (hi - lo) / (2.0 * f64::from(eps));
            let analytic = f64::from(grad.data[j]);
            assert!(
                (numeric - analytic).abs() < 0.002 + 0.025 * numeric.abs(),
                "{architecture:?} {}[{j}]: numeric {numeric} analytic {analytic}",
                net.p.names()[i]
            );
        }
        assert!(grads.tensors[net::E].row(0).iter().all(|&x| x == 0.0));
    }
}

#[test]
fn object_heads_share_shop_scoring_and_ignore_absent_entity_contents() {
    use taverngym::{actions as a, obs as o};
    let mut net = PolicyValueNet::new(22, config(Architecture::EntityActions)).unwrap();
    // Hold global context fixed, so the test isolates the direct object-head path.
    net.p.tensors[net::W0].data.fill(0.0);
    let mut obs = observations().remove(0);
    let shop_float = o::GLOBAL_F
        + o::N_QUESTS * o::QUEST_F
        + o::PENDING_KINDS
        + a::N_BOARD * o::MINION_F
        + a::N_HAND * o::ITEM_F;
    let shop_card = 1 + a::N_POWERS + 2 + 3 * o::N_QUESTS + o::N_SECRETS + a::N_BOARD + a::N_HAND;
    let before = net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap().0;
    obs.cards.swap(shop_card, shop_card + 1);
    for i in 0..o::ITEM_F {
        obs.x.swap(shop_float + i, shop_float + o::ITEM_F + i);
    }
    let swapped = net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap().0;
    assert_eq!(before[a::A_BUY], swapped[a::A_BUY + 1]);
    assert_eq!(before[a::A_BUY + 1], swapped[a::A_BUY]);
    assert_ne!(before[a::A_BUY], before[a::A_BUY + 1]);
    for i in 0..net::N_ACTIONS {
        if i != a::A_BUY && i != a::A_BUY + 1 {
            assert_eq!(before[i], swapped[i]);
        }
    }
    obs.x[shop_float] = 0.0;
    let absent = net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap().0;
    obs.cards[shop_card] = cid("BG30_MagicItem_841").idx() as i32 + 1;
    obs.x[shop_float + 1..shop_float + o::ITEM_F].fill(99.0);
    assert_eq!(
        absent,
        net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap().0
    );
}

#[test]
fn c_defaults_and_cuda_rejection_are_explicit() {
    let cfg = NetConfig::default();
    assert_eq!(cfg.hidden, 320);
    assert_eq!(cfg.blocks, 6);
    assert_eq!(cfg.architecture, Architecture::EntityActions);
    let options = tavernagent_v2::train::TrainOptions::default();
    assert_eq!(options.net_config(), cfg);
    let trainer = tavernagent_v2::ppo::Trainer::new(
        PolicyValueNet::new(1, config(Architecture::EntityActions)).unwrap(),
    );
    assert!(
        tavernagent_v2::backend::Execution::new(
            tavernagent_v2::backend::Backend::Cuda,
            &trainer,
            0
        )
        .is_err()
    );
}
