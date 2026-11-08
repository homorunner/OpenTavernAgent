//! Acquisition provenance affects the agent's card mapping, never the game effects.
use tavernagent::mcts::observe_seat;
use tavernagent::net::{OWNED_TRINKET_SLOTS, TRINKET_SLOT_START, card_slot_indices};
use taverngym::actions::Action;
use taverngym::cards::{CardId, cid};
use taverngym::state::{Pending, TrinketSource};
use taverngym::{Game, RulesConfig, content};

fn game() -> Game {
    let mut g = Game::new(
        841,
        RulesConfig {
            hero_pool: vec!["TB_BaconShop_HERO_34".into()],
            start_gold: 100,
            gold_refill_cap: 100,
            ..RulesConfig::default()
        },
    );
    for p in 0..8 {
        g.apply(p, Action::Choose(0)).unwrap();
    }
    g.players[0].pending.clear();
    g
}

fn buy(g: &mut Game, card: CardId, source: Option<TrinketSource>) {
    g.players[0].pending.push(Pending::Trinket {
        options: vec![card],
        free: true,
        replace: None,
        source,
    });
    g.apply(0, Action::Choose(0)).unwrap();
}

fn index(card: CardId) -> i32 {
    card.idx() as i32 + 1
}

#[test]
fn two_ordinary_slots_with_power_and_reward_sources_are_observation_only() {
    let mut g = game();
    let reward = cid("BG33_Reward_014");
    g.players[0].rewards.push(reward);
    let ordinary = cid("BG30_MagicItem_841");
    let special = cid("BG30_MagicItem_998");
    // Interleaving acquisition order must not decide which two trinkets are ordinary.
    buy(&mut g, special, Some(TrinketSource::HeroPower(0)));
    buy(&mut g, ordinary, None);
    buy(&mut g, special, Some(TrinketSource::QuestReward(reward)));
    buy(&mut g, ordinary, None);
    let before = serde_json::to_value(&g).unwrap();
    let raw = taverngym::obs::observe(&g, 0);
    let obs = observe_seat(&g, 0);
    let mut expected = raw.cards;
    expected[1] = index(special);
    expected[7] = index(special);
    expected[TRINKET_SLOT_START..TRINKET_SLOT_START + 8].fill(0);
    expected[TRINKET_SLOT_START..TRINKET_SLOT_START + 2].fill(index(ordinary));
    assert_eq!(obs.cards, expected);
    assert_eq!(obs.x, raw.floats);
    assert_eq!(serde_json::to_value(&g).unwrap(), before);
    assert_eq!(g.players[0].trinkets.len(), 4);
    assert_eq!(OWNED_TRINKET_SLOTS, 2);
    assert_eq!(card_slot_indices().len(), 57);
    assert!(
        card_slot_indices()
            .iter()
            .all(|&i| !(TRINKET_SLOT_START + 2..TRINKET_SLOT_START + 8).contains(&i))
    );
    let restored: Game = serde_json::from_value(before).unwrap();
    assert_eq!(observe_seat(&restored.clone(), 0).cards, obs.cards);
}

#[test]
fn marin_and_buttons_offers_remember_the_actual_power_slot() {
    for (power, turn, slot) in [("BG30_HERO_304p", 5, 0), ("BG32_HERO_002p", 8, 1)] {
        let mut g = game();
        g.turn = turn;
        g.with_recruit(0, |c| c.set_hero_power(0, slot, cid(power)));
        g.with_recruit(0, |c| c.run_player_hook(0, |s| s.start_of_turn, "turn"));
        let Some(Pending::Trinket {
            options, source, ..
        }) = g.players[0].pending.last()
        else {
            panic!("missing {power} offer");
        };
        assert!(!options.is_empty());
        assert_eq!(*source, Some(TrinketSource::HeroPower(slot as u8)));
        // Resolve a stable non-Orb option through the real purchase path.
        let target = cid("BG30_MagicItem_841");
        let Some(Pending::Trinket { options, .. }) = g.players[0].pending.last_mut() else {
            unreachable!()
        };
        *options = vec![target];
        g.apply(0, Action::Choose(0)).unwrap();
        assert_eq!(observe_seat(&g, 0).cards[1 + slot], index(target));
        assert_eq!(g.players[0].powers[slot].card, cid(power));
        assert_eq!(observe_seat(&g, 0).cards[TRINKET_SLOT_START], 0);
    }
}

#[test]
fn boutique_trinkets_are_observed_only_for_the_first_owned_reward() {
    for reward in ["BG33_Reward_014", "BG33_Reward_015"] {
        let mut g = game();
        let reward = cid(reward);
        g.players[0].rewards = vec![cid("BG24_Reward_363"), reward];
        g.with_recruit(0, |c| c.set_card_data(0, reward, 1));
        g.with_recruit(0, |c| c.run_player_hook(0, |s| s.start_of_turn, "turn"));
        let Some(Pending::Trinket {
            source, options, ..
        }) = g.players[0].pending.last_mut()
        else {
            panic!("missing reward offer")
        };
        assert_eq!(*source, Some(TrinketSource::QuestReward(reward)));
        let target = cid("BG30_MagicItem_841");
        *options = vec![target];
        g.apply(0, Action::Choose(0)).unwrap();
        let obs = observe_seat(&g, 0);
        assert!(
            card_slot_indices()
                .iter()
                .all(|&i| obs.cards[i] != index(target))
        );
        assert_eq!(obs.cards[TRINKET_SLOT_START], 0);
        assert_eq!(g.players[0].rewards[1], reward);
        // Once this is the first owned reward, its trinket occupies that slot.
        g.players[0].rewards.remove(0);
        assert_eq!(observe_seat(&g, 0).cards[7], index(target));
    }
}

#[test]
fn orb_options_stay_visible_and_generated_trinkets_inherit_the_source() {
    for (orb, target) in [
        ("BG35_MagicItem_816", "BG30_MagicItem_998"),
        ("BG35_MagicItem_816t", "BG36_MagicItem_414"),
    ] {
        for source in [
            None,
            Some(TrinketSource::HeroPower(0)),
            Some(TrinketSource::QuestReward(cid("BG33_Reward_014"))),
        ] {
            let mut g = game();
            let reward = cid("BG33_Reward_014");
            g.players[0].rewards.push(reward);
            let (orb, target) = (cid(orb), cid(target));
            g.trinket_universe = vec![target];
            g.players[0].pending.push(Pending::Trinket {
                options: vec![orb],
                free: true,
                replace: None,
                source,
            });
            // The first pending option immediately precedes the trinket block by N_CHOICE.
            let option_slot = TRINKET_SLOT_START - taverngym::actions::N_CHOICE;
            assert_eq!(observe_seat(&g, 0).cards[option_slot], index(orb));
            g.apply(0, Action::Choose(0)).unwrap();
            assert_eq!(g.players[0].trinkets.len(), 2);
            let generated = g.players[0].trinkets[1];
            assert_eq!(generated.base(), target);
            assert_eq!(g.players[0].trinket_source(generated), source);
            let obs = observe_seat(&g, 0);
            let slot = match source {
                None => TRINKET_SLOT_START,
                Some(TrinketSource::HeroPower(_)) => 1,
                Some(TrinketSource::QuestReward(_)) => 7,
            };
            assert_eq!(obs.cards[slot], index(target));
            assert!(!obs.cards[TRINKET_SLOT_START..TRINKET_SLOT_START + 8].contains(&index(orb)));
            assert_eq!(obs.cards[TRINKET_SLOT_START + 1], 0);
        }
    }
}

#[test]
fn replacing_one_of_two_copies_keeps_their_distinct_sources() {
    let mut g = game();
    let old = cid("BG30_MagicItem_841");
    buy(&mut g, old, None);
    buy(&mut g, old, Some(TrinketSource::HeroPower(0)));
    let copy = g.players[0].trinkets[1];
    assert_ne!(copy, old);
    let new = cid("BG30_MagicItem_998");
    g.with_recruit(0, |c| content::replace_trinket(c, 0, copy, new));
    let obs = observe_seat(&g, 0);
    assert_eq!(obs.cards[1], index(new));
    assert_eq!(obs.cards[TRINKET_SLOT_START], index(old));
    assert_eq!(g.players[0].trinket_source(old), None);
    assert_eq!(
        g.players[0].trinket_source(new),
        Some(TrinketSource::HeroPower(0))
    );
}

#[test]
fn replacement_orbs_and_power_orbs_keep_generated_effects_in_the_source_slot() {
    for as_power in [false, true] {
        let mut g = game();
        let target = cid("BG30_MagicItem_998");
        let orb = cid("BG35_MagicItem_816");
        g.trinket_universe = vec![target];
        if as_power {
            g.set_hero_power(0, 1, orb);
        } else {
            let old = cid("BG30_MagicItem_841");
            buy(&mut g, old, Some(TrinketSource::HeroPower(1)));
            g.with_recruit(0, |c| content::replace_trinket(c, 0, old, orb));
        }
        let obs = observe_seat(&g, 0);
        assert_eq!(obs.cards[2], index(target));
        assert!(
            obs.cards[TRINKET_SLOT_START..TRINKET_SLOT_START + 8]
                .iter()
                .all(|&c| c == 0)
        );
        let mut simulation = g.clone();
        simulation.players[0].trinket_sources.clear();
        simulation.players[0].trinkets.clear();
        tavernagent::mcts::rewind_turn(&mut simulation, &g, 0, 123);
        assert_eq!(observe_seat(&simulation, 0).cards, obs.cards);
    }
}

#[test]
fn ordinary_slots_are_capped_even_for_debug_states_with_extra_trinkets() {
    let mut g = game();
    for _ in 0..3 {
        buy(&mut g, cid("BG30_MagicItem_841"), None);
    }
    let obs = observe_seat(&g, 0);
    assert_eq!(
        obs.cards[TRINKET_SLOT_START..TRINKET_SLOT_START + 8]
            .iter()
            .filter(|&&card| card != 0)
            .count(),
        2
    );
    assert_eq!(g.players[0].trinkets.len(), 3);
}
