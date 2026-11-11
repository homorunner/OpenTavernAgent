//! Independent PPO-Clip + GAE agent. All rollout and evaluation inference is FP32.
//! The observation, compact actions and residual network follow tavernagent.

pub mod actions;
pub mod arena;
pub mod checkpoint;
pub mod diagnostics;
pub mod eval;
pub mod gae;
pub mod heroes;
pub mod linalg;
pub mod net;
pub mod npz;
pub mod observation;
pub mod policy;
pub mod ppo;
pub mod rollout;
pub mod rules;
pub mod teacher;
pub mod train;
