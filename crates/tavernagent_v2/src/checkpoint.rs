//! Atomic iteration-boundary snapshots: FP32 weights, AdamW moments, RNG and options.

use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::net::PolicyValueNet;
use crate::npz::{NpyArray, Npz, ensure_parent};
use crate::ppo::Trainer;
use crate::train::TrainOptions;

pub const FORMAT: u32 = 1;
pub const ALGORITHM: &str = "tavernagent_v2/ppo-clip-gae-fp32-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunState {
    pub format: u32,
    pub algorithm: String,
    pub iteration: usize,
    pub games: u64,
    pub transitions: u64,
    pub options: TrainOptions,
    pub shuffle_rng: Pcg64Mcg,
    pub history: Vec<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    state: RunState,
    optimizer_step: u64,
}

pub fn save(path: &Path, trainer: &Trainer, state: &RunState) -> Result<(), String> {
    state.options.validate()?;
    if state.algorithm != ALGORITHM
        || state.format != FORMAT
        || state.options.net_config() != trainer.net.cfg
    {
        return Err("checkpoint options do not match this trainer".into());
    }
    let mut npz = trainer.net.to_npz();
    for (prefix, params) in [("adam_m", &trainer.m), ("adam_v", &trainer.v)] {
        for (name, tensor) in params.names().into_iter().zip(&params.tensors) {
            npz.insert(
                format!("{prefix}/{name}"),
                NpyArray::f32(vec![tensor.rows, tensor.cols], tensor.data.clone()),
            );
        }
    }
    npz.insert(
        "trainer_meta",
        NpyArray::text(
            &serde_json::to_string(&Metadata {
                state: state.clone(),
                optimizer_step: trainer.optimizer_step,
            })
            .map_err(|e| e.to_string())?,
        ),
    );
    ensure_parent(path)?;
    npz.save(path)
}

pub fn load(path: &Path) -> Result<(Trainer, RunState), String> {
    let npz = Npz::load(path)?;
    let meta: Metadata = serde_json::from_str(&npz.get("trainer_meta")?.as_text()?)
        .map_err(|e| format!("invalid v2 trainer metadata: {e}"))?;
    if meta.state.format != FORMAT || meta.state.algorithm != ALGORITHM {
        return Err("unsupported v2 trainer format or algorithm".into());
    }
    meta.state.options.validate()?;
    let net = PolicyValueNet::from_npz(&npz)?;
    if net.cfg != meta.state.options.net_config() || !net.p.all_finite() {
        return Err("checkpoint network does not match trainer options or is nonfinite".into());
    }
    let mut trainer = Trainer::new(net);
    for (prefix, params) in [("adam_m", &mut trainer.m), ("adam_v", &mut trainer.v)] {
        for (name, tensor) in params.names().into_iter().zip(&mut params.tensors) {
            let array = npz.get(&format!("{prefix}/{name}"))?;
            let data = array.as_f32()?;
            if array.shape != [tensor.rows, tensor.cols]
                || data.len() != tensor.len()
                || data
                    .iter()
                    .any(|v| !v.is_finite() || (prefix == "adam_v" && *v < 0.0))
            {
                return Err(format!("invalid optimizer tensor: {prefix}/{name}"));
            }
            tensor.data.copy_from_slice(data);
        }
    }
    trainer.optimizer_step = meta.optimizer_step;
    Ok((trainer, meta.state))
}

/// Read policy and evaluation settings without restoring an optimizer for training.
pub fn load_for_evaluation(path: &Path) -> Result<(PolicyValueNet, RunState), String> {
    let npz = Npz::load(path)?;
    let meta: Metadata = serde_json::from_str(&npz.get("trainer_meta")?.as_text()?)
        .map_err(|e| format!("invalid evaluation checkpoint: {e}"))?;
    if meta.state.format != FORMAT || meta.state.algorithm != ALGORITHM {
        return Err("unsupported evaluation trainer format".into());
    }
    meta.state.options.validate()?;
    let net = PolicyValueNet::from_npz_for_evaluation(&npz)?;
    if net.cfg != meta.state.options.net_config() || !net.p.all_finite() {
        return Err("invalid evaluation network/options".into());
    }
    Ok((net, meta.state))
}
