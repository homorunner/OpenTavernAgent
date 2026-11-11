//! The same masked categorical distribution is used by actors and the learner.

use rand::Rng;

use crate::net::PolicyValueNet;
use crate::observation::Observation;

#[derive(Debug)]
pub struct Categorical {
    pub probabilities: Vec<f64>,
    pub log_probabilities: Vec<f64>,
    pub entropy: f64,
}

impl Categorical {
    pub fn new(logits: &[f32], mask: &[bool]) -> Result<Self, String> {
        if logits.len() != mask.len() || !mask.iter().any(|m| *m) {
            return Err("categorical needs matching logits and a nonempty legal mask".into());
        }
        if logits.iter().zip(mask).any(|(v, m)| *m && !v.is_finite()) {
            return Err("nonfinite legal logit".into());
        }
        let top = logits
            .iter()
            .zip(mask)
            .filter(|(_, m)| **m)
            .map(|(v, _)| *v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits
            .iter()
            .zip(mask)
            .filter(|(_, m)| **m)
            .map(|(v, _)| (*v as f64 - top).exp())
            .sum();
        let log_sum = sum.ln();
        let log_probabilities: Vec<f64> = logits
            .iter()
            .zip(mask)
            .map(|(v, m)| {
                if *m {
                    (*v as f64 - top) - log_sum
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        let probabilities: Vec<f64> = log_probabilities.iter().map(|v| v.exp()).collect();
        let entropy = probabilities
            .iter()
            .zip(&log_probabilities)
            .filter(|(p, _)| **p > 0.0)
            .map(|(p, lp)| -p * lp)
            .sum();
        Ok(Self {
            probabilities,
            log_probabilities,
            entropy,
        })
    }

    pub fn sample(&self, rng: &mut impl Rng) -> usize {
        let mut draw = rng.random::<f64>();
        let mut last = 0;
        for (a, &p) in self.probabilities.iter().enumerate() {
            if p > 0.0 {
                last = a;
                if draw < p {
                    return a;
                }
                draw -= p;
            }
        }
        last // Rounding at the final cumulative probability.
    }

    pub fn greedy(&self) -> usize {
        let mut best = 0;
        for a in 1..self.probabilities.len() {
            if self.probabilities[a] > self.probabilities[best] {
                best = a;
            }
        }
        best
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Decision {
    pub action: usize,
    pub log_prob: f64,
    pub value: f32,
}

pub fn decide(
    net: &PolicyValueNet,
    obs: &Observation,
    rng: &mut impl Rng,
    greedy: bool,
) -> Result<Decision, String> {
    let (logits, value) = net.infer_f32(&obs.x, &obs.cards, &obs.mask)?;
    from_logits(&logits, value, &obs.mask, rng, greedy)
}

pub fn from_logits(
    logits: &[f32],
    value: f32,
    mask: &[bool],
    rng: &mut impl Rng,
    greedy: bool,
) -> Result<Decision, String> {
    if !value.is_finite() {
        return Err("nonfinite actor value".into());
    }
    let dist = Categorical::new(logits, mask)?;
    let action = if greedy {
        dist.greedy()
    } else {
        dist.sample(rng)
    };
    Ok(Decision {
        action,
        log_prob: dist.log_probabilities[action],
        value,
    })
}
