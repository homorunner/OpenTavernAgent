use tavernagent_v2::{
    actions,
    net::{self, NetConfig, PolicyValueNet},
    npz::NpyArray,
};

#[test]
fn evaluation_adapter_only_removes_zero_inputs_and_never_enables_training_load() {
    let mut net = PolicyValueNet::new(
        42,
        NetConfig {
            hidden: 16,
            blocks: 1,
            embedding_dim: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let names = net.p.names();
    let shop = names.iter().position(|s| s == "W_shop_in").unwrap();
    let tensor = &mut net.p.tensors[shop];
    let end = tensor.len() - tensor.cols;
    tensor.data[end..].fill(0.0);
    let mut old = net.to_npz();
    let removed_slot =
        net::card_slot_indices().partition_point(|&i| i < net::SHOP_CARD_START + actions::N_SHOP);
    let missing_entity = removed_slot * 64..(removed_slot + 1) * 64;
    let missing_mask = (net.n_card_slots() + 1) * 64 + 128 + actions::A_BUY + 7;
    let w = &net.p.tensors[names.iter().position(|s| s == "W0").unwrap()];
    let mut data = Vec::new();
    let mut source = w.data.chunks_exact(w.cols);
    for row in 0..w.rows + 65 {
        if missing_entity.contains(&row) || row == missing_mask {
            data.extend(vec![123.0; w.cols]);
        } else {
            data.extend_from_slice(source.next().unwrap());
        }
    }
    assert!(source.next().is_none());
    old.insert("W0", NpyArray::f32(vec![w.rows + 65, w.cols], data));
    let w = &net.p.tensors[shop];
    old.insert(
        "W_shop_in",
        NpyArray::f32(
            vec![w.rows - 1, w.cols],
            w.data[..w.len() - w.cols].to_vec(),
        ),
    );
    let mut meta: serde_json::Value =
        serde_json::from_str(&old.get("meta").unwrap().as_text().unwrap()).unwrap();
    meta["version"] = 1.into();
    meta["features"] = 3236.into();
    meta["input_features"] = 1732.into();
    old.insert("meta", NpyArray::text(&meta.to_string()));
    assert!(PolicyValueNet::from_npz(&old).is_err());
    let adapted = PolicyValueNet::from_npz_for_evaluation(&old).unwrap();
    assert_eq!(adapted.p, net.p);
    meta["features"] = 3244.into();
    old.insert("meta", NpyArray::text(&meta.to_string()));
    assert!(PolicyValueNet::from_npz_for_evaluation(&old).is_err());
}
