//! Execution choice is independent of the model/checkpoint tensor format.
use crate::{
    linalg::Mat,
    net::PolicyValueNet,
    observation::Observation,
    ppo::{self, PpoOptions, Trainer, UpdateMetrics},
    rollout::Sample,
};
use clap::ValueEnum;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    Cpu,
    Cuda,
}

pub enum Execution {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(Box<crate::cuda::CudaNet>),
}
impl Execution {
    pub fn new(backend: Backend, trainer: &Trainer, device: usize) -> Result<Self, String> {
        match backend {
            Backend::Cpu => Ok(Self::Cpu),
            Backend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Self::Cuda(Box::new(crate::cuda::CudaNet::new(
                        trainer, device,
                    )?)))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (trainer, device);
                    Err("CUDA backend requires a binary built with --features cuda".into())
                }
            }
        }
    }
    pub fn infer(
        &self,
        net: &PolicyValueNet,
        obs: &[&Observation],
    ) -> Result<(Mat, Vec<f32>), String> {
        match self {
            Self::Cpu => {
                let mut logits = Mat::zeros(obs.len(), crate::net::N_ACTIONS);
                let mut values = Vec::with_capacity(obs.len());
                for (i, o) in obs.iter().enumerate() {
                    let (l, v) = net.infer_f32(&o.x, &o.cards, &o.mask)?;
                    logits.row_mut(i).copy_from_slice(&l);
                    values.push(v);
                }
                Ok((logits, values))
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(gpu) => gpu.infer(net, obs),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        trainer: &mut Trainer,
        samples: &[Sample],
        epochs: usize,
        batch: usize,
        lr: f64,
        options: &PpoOptions,
        rng: &mut impl Rng,
    ) -> Result<UpdateMetrics, String> {
        match self {
            Self::Cpu => ppo::update(trainer, samples, epochs, batch, lr, options, rng),
            #[cfg(feature = "cuda")]
            Self::Cuda(gpu) => {
                let result = gpu.update(&trainer.net, samples, epochs, batch, lr, options, rng)?;
                gpu.download(trainer)?;
                Ok(result)
            }
        }
    }
}
