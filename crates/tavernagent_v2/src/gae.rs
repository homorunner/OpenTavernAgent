//! GAE on one seat's decision sequence. EndTurn is not a terminal transition.

#[derive(Clone, Copy, Debug)]
pub struct ValueStep {
    pub value: f32,
    pub reward: f32,
    pub terminated: bool,
}

/// Returns (advantage, lambda-return). At a collection boundary bootstrap from the
/// next state; at a real terminal both the next value and advantage trace are zero.
pub fn estimate(
    steps: &[ValueStep],
    bootstrap: f32,
    gamma: f32,
    lambda: f32,
) -> Result<Vec<(f32, f32)>, String> {
    if !(0.0..=1.0).contains(&gamma)
        || !(0.0..=1.0).contains(&lambda)
        || !bootstrap.is_finite()
        || steps
            .iter()
            .any(|s| !s.value.is_finite() || !s.reward.is_finite())
    {
        return Err("invalid GAE discount, lambda, reward or value".into());
    }
    let mut out = vec![(0.0, 0.0); steps.len()];
    let (mut next_value, mut advantage) = (bootstrap as f64, 0.0f64);
    for (i, step) in steps.iter().enumerate().rev() {
        if step.terminated {
            next_value = 0.0;
            advantage = 0.0;
        }
        let delta = step.reward as f64 + gamma as f64 * next_value - step.value as f64;
        advantage = delta + gamma as f64 * lambda as f64 * advantage;
        let target = advantage + step.value as f64;
        if !(advantage as f32).is_finite() || !(target as f32).is_finite() {
            return Err("nonfinite GAE result".into());
        }
        out[i] = (advantage as f32, target as f32);
        next_value = step.value as f64;
    }
    Ok(out)
}
