//! Compact policy actions must agree with engine legality and execution.
use tavernagent_v2::actions as a;
use tavernagent_v2::actions::{N_ACTIONS, N_POS, apply, engine_action, legal_mask};
use taverngym::actions as e;
use taverngym::cards::cid;
use taverngym::types::Races;
use taverngym::{Action, Game};

fn position(board: usize, card: &str) -> Game {
    let mut g = Game::new(951, Default::default());
    tavernagent_v2::heroes::select_heroes(&mut g, 951).unwrap();
    g.cfg.triple_copies = 99;
    let p = &mut g.players[0];
    p.pending.clear();
    p.skipping = false;
    p.board.clear();
    p.hand.clear();
    p.powers.clear();
    p.effects.clear();
    p.quests.clear();
    p.hero = Some(cid("TB_BaconShop_HERO_34"));
    for _ in 0..board {
        let m = g.with_recruit(0, |c| c.new_minion(0, cid("BG26_146")));
        g.players[0].board.push(m);
    }
    g.with_recruit(0, |c| c.give_card(0, cid(card)).unwrap());
    g
}

#[test]
fn compact_mapping_covers_all_hand_slots_and_full_board_hosts() {
    assert_eq!((N_ACTIONS, N_POS), (115, 7));
    assert_eq!((e::N_ACTIONS, e::N_POS), (245, 8));
    let mut g = position(7, "BG_BOT_911");
    for _ in 1..10 {
        g.with_recruit(0, |c| c.give_card(0, cid("BG_BOT_911")).unwrap());
    }
    for h in 0..10 {
        for pos in 0..7 {
            let a = a::A_PLAY + h * N_POS + pos;
            assert!(legal_mask(&g, 0)[a]);
            assert_eq!(
                engine_action(&g, 0, a).unwrap(),
                Action::Magnetize {
                    hand: h,
                    board: pos
                }
            );
            let mut copy = g.clone();
            let host = &copy.players[0].board[pos];
            let expected = (host.atk + 2, host.hp + 4);
            apply(&mut copy, 0, a).unwrap();
            assert_eq!(copy.players[0].board.len(), 7);
            let host = &copy.players[0].board[pos];
            assert_eq!((host.atk, host.hp), expected);
        }
    }
    assert!(engine_action(&g, 0, N_ACTIONS).is_err());
    g.players[0].hand[9].locked = 1;
    assert!(
        legal_mask(&g, 0)[a::A_PLAY + 9 * N_POS..]
            .iter()
            .all(|b| !b)
    );
    g.players[0].ended_turn = true;
    assert!(legal_mask(&g, 0).iter().all(|b| !b));
}

#[test]
fn seven_shop_slots_shift_later_actions_without_changing_engine_semantics() {
    let g = position(1, "BG36_345");
    let engine_mask = g.legal_mask(0);
    let agent_mask = legal_mask(&g, 0);
    for (action, &legal) in agent_mask.iter().enumerate().take(a::A_PLAY) {
        let expected = if action < a::A_SELL {
            action
        } else {
            action + 1
        };
        let resolved = engine_action(&g, 0, action).unwrap();
        assert_eq!(resolved.encode(), expected);
        assert_eq!(legal, engine_mask[expected]);
        assert_ne!(resolved, Action::Buy(7));
    }
    assert_eq!(engine_action(&g, 0, a::A_BUY + 6).unwrap(), Action::Buy(6));
    assert_eq!(engine_action(&g, 0, a::A_SELL).unwrap(), Action::Sell(0));
    assert_eq!(
        engine_action(&g, 0, a::A_CHOOSE + 15).unwrap(),
        Action::Choose(15)
    );
}

#[test]
fn magnetic_minion_can_append_or_play_before_an_incompatible_host() {
    for (races, pos) in [(Races::MECH, 1), (Races::BEAST, 0)] {
        let mut g = position(1, "BG_BOT_911");
        g.players[0].board[0].races = races;
        let a = a::A_PLAY + pos;
        assert_eq!(
            engine_action(&g, 0, a).unwrap(),
            Action::Play { hand: 0, pos }
        );
        assert!(legal_mask(&g, 0)[a]);
        apply(&mut g, 0, a).unwrap();
        assert_eq!(g.players[0].board.len(), 2);
        assert!(g.players[0].board[pos].is_card("BG_BOT_911"));
    }
    let mut g = position(7, "BG_BOT_911");
    g.players[0].board[6].races = Races::BEAST;
    assert!(!legal_mask(&g, 0)[a::A_PLAY + 6]);
    assert!(apply(&mut g, 0, a::A_PLAY + 6).is_err());
}

#[test]
fn special_magnetic_hosts_reuse_engine_rules() {
    for (card, race) in [
        ("BG_DEEP_015", Races::UNDEAD),
        ("BG31_859", Races::ELEMENTAL),
        ("BG25_807t2", Races::DEMON),
    ] {
        let mut g = position(7, card);
        g.players[0].board[6].races = race;
        let a = a::A_PLAY + 6;
        assert!(legal_mask(&g, 0)[a]);
        assert_eq!(
            engine_action(&g, 0, a).unwrap(),
            Action::Magnetize { hand: 0, board: 6 }
        );
        apply(&mut g, 0, a).unwrap();
        assert_eq!(g.players[0].board.len(), 7);
    }
}

#[test]
fn ordinary_minions_and_spells_keep_engine_legality() {
    for card in ["BG36_345", "TB_BaconShop_Triples_01"] {
        for n in 0..=7 {
            let g = position(n, card);
            let mask = legal_mask(&g, 0);
            for pos in 0..7 {
                assert_eq!(mask[a::A_PLAY + pos], g.legal_mask(0)[e::A_PLAY + pos]);
            }
        }
    }
}
