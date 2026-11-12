//! Batched mixed-policy evaluation with the same paired seats and per-game ordering.
use crate::{
    arena::{self, ArenaGame},
    backend::Execution,
    diagnostics::GameDiagnostics,
    heroes::select_heroes,
    net::PolicyValueNet,
    observation::{Observation, observe_seat_with_rules},
    policy,
    rollout::game_rng,
    rules::{ActionHistory, PolicyRules},
};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use taverngym::{Game, RulesConfig};
struct Running {
    game: Game,
    offset: usize,
    cursor: usize,
    rng: Pcg64Mcg,
    histories: [ActionHistory; 8],
    diagnostics: Vec<GameDiagnostics>,
}
impl Running {
    fn new(seed: u64, offset: usize, cfg: &RulesConfig) -> Result<Self, String> {
        let mut game = Game::new(seed, cfg.clone());
        select_heroes(&mut game, seed)?;
        Ok(Self {
            game,
            offset,
            cursor: 0,
            rng: game_rng(seed),
            histories: [ActionHistory::default(); 8],
            diagnostics: vec![GameDiagnostics::default(); 8],
        })
    }
    fn ready(
        &mut self,
        a: PolicyRules,
        b: PolicyRules,
    ) -> Result<Option<(usize, bool, Observation)>, String> {
        if self.game.stopped_reason.is_some() {
            return Err("batched arena simulator stopped".into());
        }
        if self.game.is_over() {
            return Ok(None);
        }
        for _ in 0..8 {
            let seat = self.cursor;
            self.cursor = (seat + 1) % 8;
            if self.game.needs_action(seat) {
                let candidate = (seat + self.offset) % 8 < 4;
                return Ok(Some((
                    seat,
                    candidate,
                    observe_seat_with_rules(
                        &self.game,
                        seat,
                        if candidate { a } else { b },
                        &self.histories[seat],
                    ),
                )));
            }
        }
        Err("batched arena stalled".into())
    }
    fn apply(
        &mut self,
        seat: usize,
        candidate: bool,
        obs: &Observation,
        logits: &[f32],
        value: f32,
        rules: PolicyRules,
    ) -> Result<(), String> {
        let turn = self.game.turn;
        let decision = policy::from_logits(logits, value, &obs.mask, &mut self.rng, true)?;
        if candidate {
            self.diagnostics[seat].record(&self.game, seat, &self.histories[seat], decision.action);
        }
        crate::actions::apply_with_rules(&mut self.game, seat, decision.action, rules)
            .map_err(str::to_owned)?;
        self.histories[seat].record(turn, decision.action);
        Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    candidate: &PolicyValueNet,
    reference: &PolicyValueNet,
    a: &Execution,
    b: &Execution,
    cfg: &RulesConfig,
    a_rules: PolicyRules,
    b_rules: PolicyRules,
    seeds: &[u64],
    batch: usize,
    pool: &rayon::ThreadPool,
) -> Result<Vec<ArenaGame>, String> {
    cfg.validate()?;
    if cfg.num_players != 8 || batch == 0 {
        return Err("arena requires eight seats and a positive inference batch".into());
    }
    let mut output = Vec::with_capacity(seeds.len() * 2);
    let indices: Vec<_> = (0..seeds.len() * 2).collect();
    for chunk in indices.chunks(batch) {
        let mut games = pool.install(|| {
            chunk
                .par_iter()
                .map(|&i| Running::new(seeds[i / 2], (i / 2) % 4 + 4 * (i % 2), cfg))
                .collect::<Result<Vec<_>, _>>()
        })?;
        loop {
            let ready = pool.install(|| {
                games
                    .par_iter_mut()
                    .map(|g| g.ready(a_rules, b_rules))
                    .collect::<Result<Vec<_>, _>>()
            })?;
            if ready.iter().all(Option::is_none) {
                break;
            }
            let mut results = Vec::new();
            for (is_candidate, net, execution) in [(true, candidate, a), (false, reference, b)] {
                let obs: Vec<_> = ready
                    .iter()
                    .filter_map(|r| {
                        r.as_ref()
                            .and_then(|(_, c, o)| (*c == is_candidate).then_some(o))
                    })
                    .collect();
                results.push(if obs.is_empty() {
                    None
                } else {
                    Some(execution.infer(net, &obs)?)
                });
            }
            let mut counters = [0, 0];
            let work: Vec<_> = ready
                .into_iter()
                .map(|r| {
                    r.map(|(seat, c, obs)| {
                        let group = usize::from(!c);
                        let row = counters[group];
                        counters[group] += 1;
                        let (logits, values) = results[group].as_ref().unwrap();
                        (seat, c, obs, logits.row(row).to_vec(), values[row])
                    })
                })
                .collect();
            pool.install(|| {
                games
                    .par_iter_mut()
                    .zip(work.into_par_iter())
                    .try_for_each(|(g, w)| {
                        if let Some((s, c, o, l, v)) = w {
                            g.apply(s, c, &o, &l, v, if c { a_rules } else { b_rules })
                        } else {
                            Ok(())
                        }
                    })
            })?;
        }
        output.extend(
            games
                .into_iter()
                .map(|g| arena::finish(g.game, g.offset, g.diagnostics))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(output)
}
