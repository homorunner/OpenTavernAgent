//! Independent PPO-Clip + GAE agent. All rollout and evaluation inference is FP32.
//! The observation, compact actions and residual network follow tavernagent.

pub mod actions;
pub mod arena;
pub mod backend;
pub mod batched;
pub mod batched_arena;
pub mod batched_eval;
pub mod checkpoint;
pub mod critic;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod diagnostics;
pub mod eval;
pub mod gae;
pub mod heroes;
pub mod league;
pub mod linalg;
pub mod net;
pub mod npz;
pub mod observation;
pub mod policy;
pub mod ppo;
pub mod replay;
pub mod rollout;
pub mod rules;
pub mod teacher;
pub mod train;
