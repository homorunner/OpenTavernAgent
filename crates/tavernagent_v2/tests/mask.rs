//! Training-mask rules, including sole-action escape hatches.
use serde_json::Value;
use tavernagent_v2::{
    heroes::select_heroes,
    net::N_ACTIONS,
    observation::{MUST_SPEND_GOLD, agent_mask},
    rules::Budgets,
};
use taverngym::{
    Game,
    actions::{A_CANCEL, A_CHOOSE, A_END as A_END_TURN, A_HERO_POWER, A_PLAY, A_REFRESH, N_CHOICE},
    cards::{CardId, CardKind, cid, db},
    state::{Item, Pending},
};
fn position(seed: u64) -> Game {
    let mut game = Game::new(seed, Budgets::default().config().unwrap());
    select_heroes(&mut game, seed).unwrap();
    // Give the controlled seat a plain recruit phase, without hero-specific menus.
    let mut state: Value = serde_json::to_value(&game).unwrap();
    let me = &mut state["players"][0];
    me["pending"] = Value::Array(vec![]);
    me["skipping"] = Value::Bool(false);
    me["gold"] = Value::from(30);
    serde_json::from_value(state).unwrap()
}

#[test]
fn the_agent_cannot_cancel_a_menu_it_opened() {
    // A targeted hand spell (Winner's Bread: give a friendly minion +2/+3) opens a target
    // menu; the engine offers Cancel, the agent is not allowed to take it.
    let mut game = position(19);
    let shop_minion = game.players[0]
        .shop
        .iter()
        .position(|s| s.item.as_minion().is_some())
        .expect("a minion in the tavern");
    game.apply_index(0, taverngym::actions::A_BUY + shop_minion)
        .unwrap();
    let hand = game.players[0].hand.len() - 1;
    game.apply_index(0, A_PLAY + hand * taverngym::actions::N_POS)
        .unwrap();
    assert_eq!(game.players[0].board.len(), 1);
    game.with_recruit(0, |ctx| {
        ctx.add_to_hand(
            0,
            Item::Spell {
                card: cid("BG36_883"),
                data: 0,
            },
        );
    });
    let spell = game.players[0]
        .hand
        .iter()
        .position(|h| matches!(h.item, Item::Spell { .. }))
        .unwrap();
    game.apply_index(0, A_PLAY + spell * taverngym::actions::N_POS)
        .unwrap();
    assert!(matches!(
        game.players[0].pending.last(),
        Some(Pending::Target { .. })
    ));
    assert!(game.legal_mask(0)[A_CANCEL]);
    let mask = agent_mask(&game, 0);
    assert!(!mask[A_CANCEL]);
    let choices: Vec<usize> = (A_CHOOSE..A_CHOOSE + N_CHOICE)
        .filter(|a| mask[*a])
        .collect();
    assert!(!choices.is_empty());
    assert_eq!(
        mask.iter().filter(|m| **m).count(),
        choices.len(),
        "only the menu's options are open"
    );
    let atk = game.players[0].board[0].atk;
    game.apply_index(0, choices[0]).unwrap();
    assert!(game.players[0].pending.is_empty());
    assert_eq!(game.players[0].board[0].atk, atk + 2);

    // Same for a Hero Power's target menu (Lich Baz'hial: steal a Tavern card).
    game.players[0].set_hero_power(0, cid("TB_BaconShop_HP_049"));
    game.apply_index(0, A_HERO_POWER).unwrap();
    assert!(game.legal_mask(0)[A_CANCEL]);
    let mask = agent_mask(&game, 0);
    assert!(!mask[A_CANCEL]);
    assert!((A_CHOOSE..A_CHOOSE + N_CHOICE).any(|a| mask[a]));
}

#[test]
fn cancel_stays_when_it_is_the_only_legal_action() {
    // A paid trinket offer with nothing affordable is the one state where the engine makes
    // Cancel the sole legal action. `offer_trinkets` skips such an offer, so build it by
    // hand: the rule's empty-mask guard must leave Cancel in place.
    let mut game = position(19);
    game.players[0].gold = 0;
    let dear: Vec<_> = (0..db().len() as u16)
        .map(CardId)
        .filter(|c| db().get(*c).kind == CardKind::TrinketGreater && db().get(*c).cost > 0)
        .take(3)
        .collect();
    assert_eq!(dear.len(), 3);
    game.players[0].pending.push(Pending::Trinket {
        options: dear,
        free: false,
        replace: None,
        source: None,
    });
    let legal = game.legal_mask(0);
    assert!(legal[A_CANCEL]);
    assert_eq!(legal.iter().filter(|m| **m).count(), 1);
    assert_eq!(agent_mask(&game, 0), legal[..N_ACTIONS]);
}

#[test]
fn the_must_spend_rule_hides_end_turn_while_gold_is_left() {
    let mut game = position(23);
    assert!(game.legal_mask(0)[A_END_TURN], "the game itself allows it");
    let mask = agent_mask(&game, 0);
    assert!(!mask[A_END_TURN], "30 Gold is not a place to stop");
    assert!(
        mask.iter().any(|m| *m),
        "masking End Turn never leaves the seat without an action"
    );
    // The rule is a plain Gold check, so it releases only once Gold itself is gone.
    let mut steps = 0;
    while !agent_mask(&game, 0)[A_END_TURN] {
        let pending = !game.players[0].pending.is_empty();
        let mask = agent_mask(&game, 0);
        let from = if pending { 0 } else { A_REFRESH };
        let action = (from..N_ACTIONS).find(|a| mask[*a]).unwrap();
        tavernagent_v2::actions::apply(&mut game, 0, action).unwrap();
        steps += 1;
        assert!(steps < 512, "the rule never released End Turn");
    }
    assert!(game.players[0].gold < MUST_SPEND_GOLD);
    assert!(game.legal_mask(0)[A_END_TURN]);
}

#[test]
fn a_free_refresh_does_not_unlock_end_turn() {
    let mut game = position(29);
    let mut state: Value = serde_json::to_value(&game).unwrap();
    let me = &mut state["players"][0];
    me["gold"] = Value::from(MUST_SPEND_GOLD);
    me["shop"] = Value::Array(vec![]);
    me["free_refreshes"] = Value::from(1);
    me["tier"] = Value::from(6);
    me["powers"] = Value::Array(vec![]);
    me["hand"] = Value::Array(vec![]);
    me["board"] = Value::Array(vec![]);
    game = serde_json::from_value(state).unwrap();
    // Nothing is affordable right now, but a free Refresh may turn something up: the rule
    // keys on Gold alone and keeps End Turn masked, so the seat has to take the Refresh.
    let mask = agent_mask(&game, 0);
    assert!(!mask[A_END_TURN], "3 Gold is still Gold");
    assert!(
        mask[A_REFRESH],
        "the free Refresh is what the rule pushes toward"
    );
}

#[test]
fn the_action_cap_overrides_the_must_spend_rule() {
    let mut game = position(31);
    game.players[0].actions_this_turn = game.action_cap_of(0);
    let mask = agent_mask(&game, 0);
    assert!(
        mask[A_END_TURN],
        "past the cap End Turn is the only legal action, so the rule steps aside"
    );
    assert!(mask.iter().any(|m| *m), "the mask is never emptied");
}
