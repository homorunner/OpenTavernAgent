//! Reserved CUDA entry point. The C architecture currently runs on CPU only.
#[derive(Debug, clap::Args)]
pub struct CheckArgs {
    #[arg(long, default_value_t = 0)]
    pub device: usize,
}
pub fn check(_args: CheckArgs) -> Result<(), String> {
    Err("CUDA is not implemented for the C architecture; use --backend cpu".into())
}
