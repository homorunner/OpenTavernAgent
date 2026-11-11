//! Fixed-opponent evaluation, preserving the CPU driver's seat order while batching games.
use crate::{
    backend::Execution,
    diagnostics::GameDiagnostics,
    eval::{GameEvaluation, Opponent},
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
    seed: u64,
    cursor: usize,
    acted: bool,
    rng: Pcg64Mcg,
    history: ActionHistory,
    diagnostics: GameDiagnostics,
}
impl Running {
    fn new(seed: u64, cfg: &RulesConfig) -> Result<Self, String> {
        let mut game = Game::new(seed, cfg.clone());
        select_heroes(&mut game, seed)?;
        Ok(Self {
            game,
            seed,
            cursor: 0,
            acted: true,
            rng: game_rng(seed),
            history: ActionHistory::default(),
            diagnostics: GameDiagnostics::default(),
        })
    }
    fn ready(
        &mut self,
        rules: PolicyRules,
        opponent: Opponent,
    ) -> Result<Option<Observation>, String> {
        loop {
            if self.cursor == 0 {
                if self.game.is_over() || !self.game.players[0].alive {
                    return Ok(None);
                }
                if self.game.stopped_reason.is_some() || !self.acted {
                    return Err(format!("batched evaluation stalled: seed={}", self.seed));
                }
                self.acted = false;
            }
            let seat = self.cursor;
            self.cursor = (seat + 1) % self.game.players.len();
            if !self.game.needs_action(seat) {
                continue;
            }
            self.acted = true;
            if seat == 0 {
                return Ok(Some(observe_seat_with_rules(
                    &self.game,
                    0,
                    rules,
                    &self.history,
                )));
            }
            let saved = self.game.rng.clone();
            let action = match opponent {
                Opponent::Scripted => taverngym::bot::choose_action(&mut self.game, seat),
                Opponent::Teacher => crate::teacher::choose_action(&mut self.game, seat),
            };
            self.game.rng = saved;
            self.game.apply(seat, action).map_err(str::to_owned)?;
        }
    }
    fn apply(
        &mut self,
        obs: &Observation,
        logits: &[f32],
        value: f32,
        sample: bool,
    ) -> Result<(), String> {
        let turn = self.game.turn;
        let decision = policy::from_logits(logits, value, &obs.mask, &mut self.rng, !sample)?;
        self.diagnostics
            .record(&self.game, 0, &self.history, decision.action);
        crate::actions::apply(&mut self.game, 0, decision.action).map_err(str::to_owned)?;
        self.history.record(turn, decision.action);
        Ok(())
    }
    fn finish(self) -> Result<GameEvaluation, String> {
        let placement = self.game.players[0].placement;
        if placement == 0 || self.game.stopped_reason.is_some() {
            return Err("batched evaluation missing placement".into());
        }
        Ok(GameEvaluation {
            placement,
            diagnostics: self.diagnostics,
            turn_limit_end: self.game.turn_limit_end,
            truncated_combats: self.game.truncated_combats,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    net: &PolicyValueNet,
    execution: &Execution,
    seeds: &[u64],
    cfg: &RulesConfig,
    rules: PolicyRules,
    opponent: Opponent,
    sample: bool,
    batch: usize,
    pool: &rayon::ThreadPool,
) -> Result<Vec<GameEvaluation>, String> {
    cfg.validate()?;
    if batch == 0 {
        return Err("inference-batch must be positive".into());
    }
    let mut results = Vec::with_capacity(seeds.len());
    for chunk in seeds.chunks(batch) {
        let mut games = pool.install(|| {
            chunk
                .par_iter()
                .map(|&seed| Running::new(seed, cfg))
                .collect::<Result<Vec<_>, _>>()
        })?;
        loop {
            let ready = pool.install(|| {
                games
                    .par_iter_mut()
                    .map(|game| game.ready(rules, opponent))
                    .collect::<Result<Vec<_>, _>>()
            })?;
            let obs: Vec<_> = ready.iter().filter_map(Option::as_ref).collect();
            if obs.is_empty() {
                break;
            }
            let (logits, values) = execution.infer(net, &obs)?;
            let mut row = 0;
            let work: Vec<_> = ready
                .into_iter()
                .map(|o| {
                    o.map(|o| {
                        let result = (o, logits.row(row).to_vec(), values[row]);
                        row += 1;
                        result
                    })
                })
                .collect();
            pool.install(|| {
                games
                    .par_iter_mut()
                    .zip(work.into_par_iter())
                    .try_for_each(|(game, work)| {
                        if let Some((obs, logits, value)) = work {
                            game.apply(&obs, &logits, value, sample)
                        } else {
                            Ok(())
                        }
                    })
            })?;
        }
        results.extend(
            games
                .into_iter()
                .map(Running::finish)
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(results)
}
