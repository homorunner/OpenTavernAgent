//! vdiag: what the value head believes about Play vs hold at T1/T2, and what the search does.
//! Usage: vdiag <model.npz> [games]
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use tavernagent::heroes::select_heroes;
use tavernagent::mcts::{self, Evaluator, SearchOptions, agent_mask, observe_seat};
use tavernagent::net::PolicyValueNet;
use tavernagent::rules::Budgets;
use tavernagent::selfplay::end_turn_preview;
use taverngym::Game;
use taverngym::actions::{A_BUY, A_END, A_PLAY, A_REFRESH, N_POS, N_SHOP};
use taverngym::state::Item;

type Row = (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, bool);

fn v(net: &PolicyValueNet, g: &Game) -> f32 {
    net.evaluate(&observe_seat(g, 0)).unwrap().1
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = PolicyValueNet::load(std::path::Path::new(&args[1])).unwrap();
    let games: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let opts = SearchOptions {
        sims: 256,
        fpu_reduction: mcts::DEFAULT_FPU_REDUCTION,
        ..SearchOptions::default()
    };
    let mut rows = Vec::new();
    for seed in 0..games {
        let mut game = Game::new(seed + 1000, Budgets::default().config().unwrap());
        select_heroes(&mut game, seed).unwrap();
        if !game.players[0].pending.is_empty() {
            continue;
        }
        // T1: buy the first minion.
        let Some(buy) = (A_BUY..A_BUY + N_SHOP).find(|a| {
            game.legal_mask(0)[*a]
                && matches!(game.players[0].shop[*a - A_BUY].item, Item::Minion(_))
        }) else {
            continue;
        };
        game.apply_index(0, buy).unwrap();
        let v_pos = v(&net, &game);
        if std::env::var("VDIAG_ROWS").is_ok() {
            let hero = game.players[0]
                .hero
                .map(|h| taverngym::cards::db().get(h).name.clone())
                .unwrap_or_default();
            let opp_v: Vec<String> = (1..game.players.len())
                .map(|s| format!("{:+.2}", net.evaluate(&observe_seat(&game, s)).unwrap().1))
                .collect();
            println!(
                "seed {seed:3} V0 {v_pos:+.3}  hero {hero:<28} others {}",
                opp_v.join(" ")
            );
        }
        // Hold: end now.
        let hold = end_turn_preview(&game, 0).unwrap();
        let v_hold = v(&net, &hold);
        if !game.players[0].pending.is_empty() {
            continue;
        }
        // Play then end.
        let mut played = game.clone();
        let play = (A_PLAY..A_PLAY + N_POS)
            .find(|a| played.legal_mask(0)[*a])
            .unwrap();
        played.apply_index(0, play).unwrap();
        if !played.players[0].pending.is_empty() {
            continue;
        }
        let v_played_pos = v(&net, &played);
        let pend = end_turn_preview(&played, 0).unwrap();
        let v_play_end = v(&net, &pend);
        // Search from the post-buy position.
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let r = mcts::search(&game, 0, &net, &opts, &mut rng).unwrap();
        let mask = agent_mask(&game, 0);
        let prior = net.evaluate(&observe_seat(&game, 0)).unwrap().0;
        let fam = |p: &[f32]| {
            let play: f32 = (A_PLAY..A_PLAY + 10 * N_POS).map(|a| p[a]).sum();
            (p[A_END], p[A_REFRESH], play)
        };
        let (pe, pr, pp) = fam(&prior);
        let (ve, vr, vp) = fam(&r.policy);
        rows.push((
            v_pos,
            v_hold,
            v_played_pos,
            v_play_end,
            pe,
            pr,
            pp,
            ve,
            vr,
            vp,
            r.value,
            mask[A_END],
        ));
    }
    let n = rows.len() as f32;
    let m = |f: &dyn Fn(&Row) -> f32| rows.iter().map(f).sum::<f32>() / n;
    println!(
        "n={}  (T1 after Buy, 0 gold left; End is masked by must-spend only when gold>=3)",
        rows.len()
    );
    println!("V(pos after Buy)          {:+.3}", m(&|r| r.0));
    println!("V(End, hold in hand)      {:+.3}", m(&|r| r.1));
    println!("V(pos after Play)         {:+.3}", m(&|r| r.2));
    println!("V(Play then End)          {:+.3}", m(&|r| r.3));
    println!(
        "prior  End/Refresh/Play   {:.2} / {:.2} / {:.2}",
        m(&|r| r.4),
        m(&|r| r.5),
        m(&|r| r.6)
    );
    println!(
        "search End/Refresh/Play   {:.2} / {:.2} / {:.2}   root Q {:+.3}",
        m(&|r| r.7),
        m(&|r| r.8),
        m(&|r| r.9),
        m(&|r| r.10)
    );
    let sd = |f: &dyn Fn(&Row) -> f32| {
        let mu = m(f);
        (rows.iter().map(|r| (f(r) - mu).powi(2)).sum::<f32>() / n).sqrt()
    };
    println!(
        "across-games std of V(pos after Buy) {:.3}; mean |V(Play+End) - V(End)| {:.4}",
        sd(&|r| r.0),
        m(&|r| (r.3 - r.1).abs())
    );
    let pref = rows.iter().filter(|r| r.3 > r.1).count();
    println!("games where V(Play+End) > V(End): {}/{}", pref, rows.len());
}
