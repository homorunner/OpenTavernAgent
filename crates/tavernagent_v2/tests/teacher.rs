use tavernagent_v2::{heroes::select_heroes, rules::Budgets, teacher};

fn bot_action(game: &mut taverngym::Game, seat: usize) -> taverngym::actions::Action {
    let saved = game.rng.clone();
    let action = taverngym::bot::choose_action(game, seat);
    game.rng = saved;
    action
}
use taverngym::Game;

#[test]
fn matches_upstream_no_composition_games_and_preserves_rng() {
    // Recorded with upstream 5075b273 teacher, empty comps, bonuses zero, same engine/budgets.
    let cases: &[(u64, [usize; 4], [u8; 8])] = &[
        (73100000, [0, 3, 4, 5], [1, 6, 4, 2, 3, 5, 8, 7]),
        (73100001, [4, 7, 2, 1], [8, 6, 2, 5, 4, 1, 7, 3]),
        (73100002, [4, 3, 5, 1], [5, 2, 8, 4, 1, 7, 6, 3]),
        (73100003, [6, 7, 5, 2], [7, 8, 4, 1, 6, 5, 3, 2]),
        (73100004, [1, 5, 6, 7], [8, 3, 7, 5, 4, 2, 6, 1]),
        (73100005, [2, 6, 3, 5], [6, 8, 3, 5, 1, 4, 2, 7]),
        (73100006, [5, 7, 4, 3], [6, 7, 8, 5, 4, 1, 3, 2]),
        (73100007, [7, 1, 6, 5], [6, 2, 8, 3, 1, 7, 4, 5]),
        (73100008, [7, 6, 0, 3], [3, 7, 8, 6, 2, 5, 4, 1]),
        (73100009, [5, 7, 6, 1], [8, 2, 1, 4, 6, 3, 5, 7]),
        (73100010, [6, 1, 5, 2], [8, 4, 6, 7, 5, 2, 3, 1]),
        (73100011, [3, 4, 2, 0], [5, 8, 1, 2, 4, 6, 3, 7]),
        (73100012, [2, 0, 6, 7], [1, 3, 5, 8, 4, 7, 2, 6]),
        (73100013, [4, 1, 2, 7], [6, 1, 2, 8, 3, 7, 5, 4]),
        (73100014, [7, 0, 2, 5], [4, 5, 1, 6, 8, 3, 7, 2]),
        (73100015, [2, 7, 0, 1], [6, 3, 2, 1, 5, 4, 8, 7]),
    ];
    for &(seed, teachers, expected) in cases {
        let mut game = Game::new(seed, Budgets::default().config().unwrap());
        select_heroes(&mut game, seed).unwrap();
        let mut actions = 0;
        while !game.is_over() {
            let seats: Vec<usize> = if game.needs_action(0) {
                vec![0]
            } else {
                (1..8).collect()
            };
            let mut acted = false;
            for seat in seats {
                if !game.needs_action(seat) {
                    continue;
                }
                let before = serde_json::to_value(&game.rng).unwrap();
                let action = if teachers.contains(&seat) {
                    let a = teacher::choose_action(&mut game, seat);
                    assert_eq!(a, teacher::choose_action(&mut game, seat));
                    assert_eq!(a, tavernagent::teacher::choose_action(&mut game, seat));
                    a
                } else {
                    bot_action(&mut game, seat)
                };
                assert_eq!(before, serde_json::to_value(&game.rng).unwrap());
                assert!(
                    game.legal_mask(seat)[action.encode()],
                    "seed={seed} seat={seat} action={action:?}"
                );
                game.apply(seat, action).unwrap();
                actions += 1;
                assert!(actions < 100_000);
                acted = true;
            }
            assert!(acted);
        }
        assert_eq!(
            game.players.iter().map(|p| p.placement).collect::<Vec<_>>(),
            expected
        );
        assert!(!game.turn_limit_end);
        assert_eq!(game.truncated_combats, 0);
    }
}
