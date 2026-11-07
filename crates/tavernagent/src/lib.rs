//! TavernAgent: a policy/value network plus PUCT search over TavernGym, trained by
//! all-seat self-play. Everything runs natively; checkpoints are numpy-readable `.npz`.

pub mod actions;
pub mod arena;
pub mod bench;
pub mod eval;
pub mod heroes;
pub mod league;
pub mod linalg;
pub mod mcts;
pub mod net;
pub mod npz;
pub mod patterns;
pub mod play;
pub mod probe;
pub mod rules;
pub mod selfplay;
pub mod tournament;
pub mod train;
