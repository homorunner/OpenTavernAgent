//! Observational full-lobby replay. Combat boundaries use the engine pause/resume API;
//! recording never samples RNG or changes the policy's decision order.
use crate::{actions, policy::Decision, rules::PolicyRules};
use serde_json::{Value, json};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use taverngym::{Action, Game};

pub fn snapshot(game: &Game, seat: usize) -> Value {
    let mut view = taverngym::debug::public_view_without_combats(game, seat);
    let mut player = view["me"].take();
    // Debug views also contain accumulated effect breakdowns and internal counters.
    // Keep the stable fields the viewer needs, not a full engine/debug save state.
    player.as_object_mut().unwrap().retain(|key, _| {
        matches!(
            key.as_str(),
            "hero"
                | "hero_powers"
                | "health"
                | "armor"
                | "alive"
                | "placement"
                | "tier"
                | "gold"
                | "upgrade_price"
                | "free_refreshes"
                | "board"
                | "hand"
                | "shop"
                | "trinkets"
                | "pending"
                | "rewards"
                | "quests"
                | "next_opponent"
                | "next_is_ghost"
                | "ended_turn"
        )
    });
    compact_card(&mut player["hero"]);
    for field in [
        "board",
        "hand",
        "shop",
        "trinkets",
        "hero_powers",
        "rewards",
    ] {
        if let Some(cards) = player[field].as_array_mut() {
            for card in cards {
                compact_card(card);
            }
        }
    }
    let p = &game.players[seat];
    player["seat"] = json!(seat);
    player["alive"] = json!(p.alive);
    player["placement"] = json!(p.placement);
    player["ended_turn"] = json!(p.ended_turn);
    player["turn"] = json!(game.turn);
    player
}
fn compact_card(card: &mut Value) {
    if let Some(card) = card.as_object_mut() {
        card.retain(|key, _| {
            matches!(
                key.as_str(),
                "eid"
                    | "id"
                    | "name"
                    | "name_zh"
                    | "text"
                    | "text_zh"
                    | "atk"
                    | "hp"
                    | "max_hp"
                    | "golden"
                    | "keywords"
                    | "counter"
                    | "locked"
                    | "held"
                    | "frozen"
                    | "price"
                    | "costs_health"
                    | "slot"
                    | "usable"
                    | "passive"
                    | "uses_turn"
            )
        });
    }
}
fn players(game: &Game) -> Vec<Value> {
    (0..game.players.len())
        .map(|seat| snapshot(game, seat))
        .collect()
}

pub struct Recorder {
    events: Vec<Value>,
}
impl Recorder {
    pub fn new(game: &Game) -> Self {
        Self {
            events: vec![json!({"type":"turn_start", "turn":game.turn,"players":players(game)})],
        }
    }
    pub fn apply_model(
        &mut self,
        game: &mut Game,
        seat: usize,
        decision: Decision,
        rules: PolicyRules,
    ) -> Result<(), &'static str> {
        let action = actions::engine_action(game, seat, decision.action)?;
        let i = self.begin(game, seat, action, "model");
        self.events[i]["agent_action"] = json!(decision.action);
        self.events[i]["probability"] = json!(decision.log_prob.exp());
        self.events[i]["value"] = json!(decision.value);
        self.events[i]["freeze_and_end"] =
            json!(rules.freeze_at_end && decision.action == taverngym::actions::A_FREEZE);
        actions::apply_with(game, seat, decision.action, rules, |g, s, a| {
            self.apply_engine(g, s, a)
        })?;
        self.events[i]["resolved_combat"] = json!(
            self.events[i + 1..]
                .iter()
                .any(|e| e["type"] == "combat_start")
        );
        self.events[i]["after"] = snapshot(game, seat);
        Ok(())
    }
    pub fn apply_bot(
        &mut self,
        game: &mut Game,
        seat: usize,
        action: Action,
        opponent: &str,
    ) -> Result<(), &'static str> {
        let i = self.begin(game, seat, action, opponent);
        self.apply_engine(game, seat, action)?;
        self.events[i]["resolved_combat"] = json!(
            self.events[i + 1..]
                .iter()
                .any(|e| e["type"] == "combat_start")
        );
        self.events[i]["after"] = snapshot(game, seat);
        Ok(())
    }
    fn begin(&mut self, game: &Game, seat: usize, action: Action, controller: &str) -> usize {
        let i = self.events.len();
        self.events.push(json!({"type":"action","turn":game.turn,"seat":seat,"controller":controller,
            "engine_action":action,"engine_action_index":action.encode(),"before":snapshot(game, seat)}));
        i
    }
    fn apply_engine(
        &mut self,
        game: &mut Game,
        seat: usize,
        action: Action,
    ) -> Result<(), &'static str> {
        if game.apply_index_until_combat(seat, action.encode())? {
            let turn = game.turn;
            self.events
                .push(json!({"type":"combat_start","turn":turn,"players":players(game)}));
            game.resume_combat()?;
            // Frames are intentionally omitted: this format records decisions and warbands,
            // not individual combat attacks. Do not enable record_log on the engine.
            let combats: Vec<_> = game.last_combats.iter().map(|c| json!({"a":c.a,"b":c.b,"winner":c.winner,"damage":c.damage,"truncated":c.truncated})).collect();
            self.events.push(
                json!({"type":"combat_end","turn":turn,"combats":combats,"players":players(game)}),
            );
            if !game.is_over() {
                self.events
                    .push(json!({"type":"turn_start","turn":game.turn,"players":players(game)}));
            }
        }
        Ok(())
    }
    pub fn finish(mut self, game: &Game) -> Value {
        self.events
            .push(json!({"type":"game_end","turn":game.turn,"players":players(game)}));
        json!({"schema":"tavernagent_v2.replay","version":1,"seed":game.seed.to_string(),
            "placements":game.players.iter().map(|p|p.placement).collect::<Vec<_>>(),
            "turn_limit_end":game.turn_limit_end,"truncated_combats":game.truncated_combats,
            "events":self.events})
    }
}

/// Reject aliases between the input model and every final/temporary output path.
pub fn validate_paths(model: &Path, summary: Option<&Path>, replay: &Path) -> Result<(), String> {
    let mut paths = vec![model.to_path_buf(), replay.to_path_buf()];
    let mut partial = replay.as_os_str().to_os_string();
    partial.push(".partial");
    paths.push(PathBuf::from(partial));
    if let Some(summary) = summary {
        paths.push(summary.to_path_buf());
        paths.push(summary.with_extension("json.tmp"));
    }
    let mut normalized = std::collections::HashSet::new();
    for path in paths {
        crate::npz::ensure_parent(&path)?;
        let canonical = if path.exists() {
            fs::canonicalize(&path).map_err(|e| e.to_string())?
        } else {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            fs::canonicalize(parent)
                .map_err(|e| e.to_string())?
                .join(path.file_name().ok_or("output path needs a filename")?)
        };
        if !normalized.insert(canonical) {
            return Err(
                "model, summary and replay paths (including temporary outputs) must be distinct"
                    .into(),
            );
        }
    }
    Ok(())
}

/// One complete game per line, ordered by seed. Publish only after every game succeeds.
/// A failed export removes its partial file and never overwrites an existing replay.
pub struct JsonlWriter {
    path: PathBuf,
    partial: PathBuf,
    writer: Option<BufWriter<File>>,
}
impl JsonlWriter {
    pub fn new(path: &Path) -> Result<Self, String> {
        if path.exists() {
            return Err("replay output already exists; choose a new path".into());
        }
        crate::npz::ensure_parent(path)?;
        let mut name = path.as_os_str().to_os_string();
        name.push(".partial");
        let partial = PathBuf::from(name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            path: path.into(),
            partial,
            writer: Some(BufWriter::new(file)),
        })
    }
    pub fn write(&mut self, game: &Value) -> Result<(), String> {
        let w = self.writer.as_mut().ok_or("replay already closed")?;
        serde_json::to_writer(&mut *w, game).map_err(|e| e.to_string())?;
        w.write_all(b"\n").map_err(|e| e.to_string())
    }
    pub fn finish(mut self) -> Result<(), String> {
        let mut writer = self.writer.take().ok_or("replay already closed")?;
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        drop(writer);
        // hard_link fails atomically if a destination appeared during evaluation.
        fs::hard_link(&self.partial, &self.path).map_err(|e| e.to_string())?;
        fs::remove_file(&self.partial).map_err(|e| e.to_string())
    }
}
impl Drop for JsonlWriter {
    fn drop(&mut self) {
        self.writer.take();
        let _ = fs::remove_file(&self.partial);
    }
}
