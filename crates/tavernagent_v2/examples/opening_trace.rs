//! Trace recruit turns 1 and 2 using the same policy, rules and RNG as evaluation.
use clap::Parser;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};
use tavernagent_v2::rules::{ActionHistory, PolicyRules};
use tavernagent_v2::{actions, checkpoint, heroes, observation, policy, rollout, teacher};
use taverngym::{
    Action, Game, RulesConfig,
    cards::{CardId, db},
    state::Item,
};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1024)]
    games: usize,
    #[arg(long, default_value_t = 16)]
    workers: usize,
    #[arg(long, default_value_t = 10_000_000)]
    seed: u64,
    #[arg(long)]
    sample: bool,
}

fn card(c: CardId) -> Value {
    let d = db().get(c);
    json!({"id":d.id,"name":d.name,"name_zh":d.name_zh})
}

fn kind(a: Action) -> &'static str {
    match a {
        Action::EndTurn => "EndTurn",
        Action::Refresh => "Refresh",
        Action::Freeze => "Freeze",
        Action::Upgrade => "Upgrade",
        Action::HeroPower(_) => "HeroPower",
        Action::DarkGift => "DarkGift",
        Action::Cancel => "Cancel",
        Action::Buy(_) => "Buy",
        Action::Sell(_) => "Sell",
        Action::Activate(_) => "Activate",
        Action::Choose(_) => "Choose",
        Action::Play { .. } | Action::Magnetize { .. } => "Play",
        Action::Move { .. } => "Move",
    }
}

fn snapshot(g: &Game) -> Value {
    let p = &g.players[0];
    json!({"gold":p.gold,"tier":p.tier,"upgrade_cost":p.upgrade_cost,
        "board":p.board.iter().map(|m| json!({"card":card(m.card),"atk":m.atk,"hp":m.hp})).collect::<Vec<_>>(),
        "hand":p.hand.iter().map(|h| json!({"card":card(h.item.card()),"locked":h.locked,"held":h.held})).collect::<Vec<_>>(),
        "shop":p.shop.iter().map(|s| json!({"card":card(s.item.card()),"frozen":s.frozen})).collect::<Vec<_>>()})
}

fn trace(
    net: &tavernagent_v2::net::PolicyValueNet,
    seed: u64,
    cfg: &RulesConfig,
    sample: bool,
    rules: PolicyRules,
) -> Result<Value, String> {
    let mut game = Game::new(seed, cfg.clone());
    heroes::select_heroes(&mut game, seed)?;
    let hero = game.players[0].hero.map(card);
    let mut rng = rollout::game_rng(seed);
    let mut rows = Vec::new();
    let mut history = ActionHistory::default();
    while game.turn <= 2 && !game.is_over() && game.players[0].alive {
        if game.stopped_reason.is_some() {
            return Err(format!("stopped at seed {seed}"));
        }
        let mut acted = false;
        for seat in 0..game.players.len() {
            if game.turn > 2 {
                break;
            }
            if !game.needs_action(seat) {
                continue;
            }
            if seat == 0 {
                let obs = observation::observe_seat_with_rules(&game, 0, rules, &history);
                let (logits, _) = net.infer_f32(&obs.x, &obs.cards, &obs.mask)?;
                let dist = policy::Categorical::new(&logits, &obs.mask)?;
                let action = if sample {
                    dist.sample(&mut rng)
                } else {
                    dist.greedy()
                };
                let engine = actions::engine_action(&game, 0, action).map_err(str::to_string)?;
                let mut masses = BTreeMap::<&str, f64>::new();
                for (i, &p) in dist.probabilities.iter().enumerate() {
                    if p > 0.0 {
                        let a = actions::engine_action(&game, 0, i).map_err(str::to_string)?;
                        let label = if rules.freeze_at_end && i == taverngym::actions::A_FREEZE {
                            "FreezeAndEnd"
                        } else {
                            kind(a)
                        };
                        *masses.entry(label).or_default() += p;
                    }
                }
                let item = match engine {
                    Action::Buy(i) => Some(&game.players[0].shop[i].item),
                    Action::Play { hand, .. } | Action::Magnetize { hand, .. } => {
                        Some(&game.players[0].hand[hand].item)
                    }
                    _ => None,
                };
                let item_type = item.map(|x| match x {
                    Item::Minion(_) => "minion",
                    Item::Spell { .. } => "spell",
                });
                let chosen_card = item.map(|x| card(x.card()));
                let turn = game.turn;
                let before = snapshot(&game);
                actions::apply_with_rules(&mut game, 0, action, rules).map_err(str::to_string)?;
                history.record(turn, action);
                rows.push(json!({"turn":turn,"action":if rules.freeze_at_end && action == taverngym::actions::A_FREEZE { "FreezeAndEnd" } else { kind(engine) },"engine_action":engine,
                    "action_index":action,"probability":dist.probabilities[action],"type_probabilities":masses,
                    "card":chosen_card,"item_type":item_type,"before":before,"after":snapshot(&game),
                    "ended":!game.needs_action(0)||game.turn!=turn}));
            } else {
                let saved = game.rng.clone();
                let action = teacher::choose_action(&mut game, seat);
                game.rng = saved;
                game.apply(seat, action).map_err(str::to_string)?;
            }
            acted = true;
        }
        if !acted && game.turn <= 2 {
            return Err(format!("stalled at seed {seed}"));
        }
    }
    Ok(json!({"seed":seed,"hero":hero,"actions":rows}))
}

fn main() -> Result<(), String> {
    let a = Args::parse();
    if a.games == 0 || a.workers == 0 {
        return Err("games/workers must be positive".into());
    }
    let (trainer, state) = checkpoint::load(&a.model)?;
    let cfg = state.options.game_config()?;
    let seeds = (0..a.games)
        .map(|i| a.seed.checked_add(i as u64).ok_or("seed overflow"))
        .collect::<Result<Vec<_>, _>>()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.workers)
        .build()
        .map_err(|e| e.to_string())?;
    let traces = pool.install(|| {
        seeds
            .par_iter()
            .map(|&s| trace(&trainer.net, s, &cfg, a.sample, state.options.policy_rules))
            .collect::<Result<Vec<_>, _>>()
    })?;
    let result = json!({"model":a.model,"model_seed":state.options.seed,"iteration":state.iteration,
        "sample":a.sample,"opponent":"teacher","evaluated_seat":0,"seed":a.seed,"games":a.games,
        "policy_rules":state.options.policy_rules,"traces":traces});
    tavernagent_v2::npz::ensure_parent(&a.out)?;
    std::fs::write(
        &a.out,
        serde_json::to_vec(&result).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"out":a.out,"games":a.games,"model_seed":state.options.seed,"sample":a.sample})
    );
    Ok(())
}
