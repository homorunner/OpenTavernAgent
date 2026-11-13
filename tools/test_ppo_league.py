"""Statistical and scheduling invariants for checkpoint league evaluation."""
import unittest
import json
import tempfile
from pathlib import Path
from collections import Counter
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from league_elo import ratings_with_bootstrap, validate_games
from ppo_league import League, schedule, average_ratings, game_exclusion_reason


class LeagueTests(unittest.TestCase):
    def test_final_budget_only_applies_to_training_endpoints(self):
        league = object.__new__(League)
        league.args = SimpleNamespace(games_per_model=512, final_games_per_model=2048, target_iteration=12000)
        league.sources = {'candidate': Path('unused')}
        league.models = {'early': {'run': 'candidate', 'iteration': 500},
                         'final': {'run': 'candidate', 'iteration': 12000},
                         'reference': {'run': 'frozen', 'iteration': 12000}}
        self.assertEqual([league.required_games(n) for n in league.models], [512, 2048, 512])
        names = list('abcdefgh')
        counts = {name: 512 for name in names}
        deficits = {name: counts[name] - (2048 if name == 'a' else 512) for name in names}
        self.assertTrue(all('a' in t for t in schedule(names, deficits, 128, 1, 'b')))

    def test_seed_means_include_partial_steps_and_preserve_bootstrap_covariance(self):
        names = ['a0', 'b0', 'c0', 'd0', 'e0', 'a1', 'candidate']
        models = {name: {'run': run, 'iteration': step} for name, run, step in
                  zip(names, ['a', 'b', 'c', 'd', 'e', 'a', 'candidate'], [3000] * 5 + [3500, 500])}
        elo = np.array([900., 950., 1000., 1050., 1100., 1200., 1300.])
        # Perfectly correlated perturbations must not shrink by sqrt(number of seeds).
        boots = np.array([elo - 10, elo, elo + 10])
        counts = np.array([100, 200, 300, 400, 500, 600, 700])
        result = average_ratings(names, elo, boots, counts, models, {'baseline': list('abcde')})
        points = result['baseline']['points']
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0]['iteration'], 3000)
        self.assertEqual(points[0]['elo'], 1000.)
        self.assertEqual(points[0]['se'], 10.)
        self.assertEqual(points[0]['seeds'], 5)
        self.assertEqual((points[0]['games_min'], points[0]['games_max']), (100, 500))
        self.assertEqual(points[1]['iteration'], 3500)
        self.assertEqual(points[1]['elo'], 1200.)
        self.assertEqual(points[1]['se'], 10.)
        self.assertEqual(points[1]['seeds'], 1)
        self.assertEqual(points[1]['expected_seeds'], 5)
        self.assertEqual(points[1]['available_members'], ['a'])
        # The same point updates when a second seed receives a rating.
        models['candidate'] = {'run': 'b', 'iteration': 3500}
        updated = average_ratings(names, elo, boots, counts, models, {'baseline': list('abcde')})
        point = updated['baseline']['points'][1]
        self.assertEqual(point['elo'], 1250.)
        self.assertEqual(point['seeds'], 2)
        self.assertEqual(point['se'], 10.)
        self.assertEqual(point['available_members'], ['a', 'b'])
        self.assertEqual(average_ratings([], elo[:0], boots[:, :0], counts[:0], {},
                                        {'baseline': list('abcde')})['baseline']['points'], [])


    def test_frozen_reference_can_exceed_candidate_iteration_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / 'candidate'
            candidates.mkdir()
            (candidates / 'model.iter4000.npz').write_bytes(b'excluded')
            reference = root / 'model.iter5000.npz'
            reference.write_bytes(b'frozen reference')
            league = League.__new__(League)
            league.excluded_models = set()
            league.root = root
            league.sources = {'candidate': candidates}
            league.references = {'baseline': reference}
            league.models = {}
            league.args = SimpleNamespace(interval=500, target_iteration=3000)
            state = {'iteration': 5000, 'options': {'seed': 4, 'hidden': 256,
                     'blocks': 6, 'max_turns': 40, 'budgets': {}}}
            with patch('ppo_league.checkpoint_state', return_value=state):
                league.discover()
            self.assertEqual(list(league.models), ['baseline_s5000'])
            self.assertEqual(league.models['baseline_s5000']['iteration'], 5000)
            league.models = {}
            league.excluded_models = {'baseline_s5000'}
            with patch('ppo_league.checkpoint_state', return_value=state):
                league.discover()
            self.assertEqual(league.models, {})
            self.assertTrue(reference.exists())


    def test_scheduler_connects_new_cohort_and_balances_seats(self):
        names = [f'm{i}' for i in range(24)]
        counts = Counter({name: 256 for name in names[:16]})
        tables = schedule(names, counts, 512, 71, names[0])
        self.assertEqual(tables, schedule(names, counts, 512, 71, names[0]))
        self.assertIn(names[0], tables[0])
        for table in tables:
            self.assertEqual(len(set(table)), 8)
            counts.update(table)
        self.assertGreaterEqual(min(counts.values()), 256)
        validate_games([(table, list(range(1, 9))) for table in tables], names)
        for name in names:
            seats = [table.index(name) for table in tables if name in table]
            self.assertGreater(len(set(seats)), 5)

    def test_symmetric_rankings_and_anchored_bootstrap(self):
        names = list('abcdefgh')
        games = []
        for _ in range(8):
            for shift in range(8):
                games.append((names, [(i + shift) % 8 + 1 for i in range(8)]))
        elo, se, counts = ratings_with_bootstrap(games, names, bootstrap=8, workers=1, anchor='a')
        np.testing.assert_allclose(elo, 1000, atol=1e-8)
        self.assertEqual(se[0], 0)
        self.assertTrue(np.all(se[1:] > 0))
        np.testing.assert_array_equal(counts, 64)
        repeat = ratings_with_bootstrap(games, names, bootstrap=8, workers=2, anchor='a')
        for first, second in zip((elo, se, counts), repeat):
            np.testing.assert_allclose(first, second, atol=1e-12)
        sampled = ratings_with_bootstrap(games, names, bootstrap=8, workers=1, anchor='a', return_samples=True)
        for first, second in zip(repeat, sampled[:3]):
            np.testing.assert_allclose(first, second, atol=1e-12)
        np.testing.assert_allclose(np.std(sampled[3], axis=0, ddof=1), sampled[1])

    def test_unidentifiable_or_invalid_rankings_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disconnected'):
            validate_games([(['a', 'b'], [1, 2]), (['c', 'd'], [1, 2])], list('abcd'))
        for game in [(['a', 'a'], [1, 2]), (['a', 'b'], [1, 1]), (['a', 'b'], [0, 1])]:
            with self.assertRaises(ValueError):
                validate_games([game], ['a', 'b'])

    def test_recovery_ignores_partial_and_never_counts_completed_games_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            league = League.__new__(League)
            league.excluded_models = set()
            league.root = Path(directory)
            batch = league.root / 'batch-00000'
            batch.mkdir()
            names = list('abcdefgh')
            (batch / 'plan.json').write_text(json.dumps({'seed': 123, 'tables': [names]}))
            row = {'game': 0, 'seed': 123, 'models': names, 'placements': list(range(1, 9)),
                   'turn_limit_end': False, 'truncated_combats': 0}
            (batch / 'games.jsonl').write_text(json.dumps(row) + '\n')
            partial = league.root / 'batch-00001'
            partial.mkdir()
            (partial / 'games.jsonl.partial').write_text(json.dumps(row))
            for _ in range(2):
                league.recover()
                self.assertEqual(league.batch, 1)
                self.assertEqual(league.counts, Counter(names))
            (batch / 'games.jsonl').write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                league.recover()
            row['truncated_combats'] = 1
            (batch / 'games.jsonl').write_text(json.dumps(row) + '\n')
            league.recover()
            self.assertEqual(league.batch, 1)
            self.assertEqual(league.counts, Counter())
            self.assertEqual(len(league.excluded), 1)
            self.assertTrue((league.root / 'excluded-games.json').exists())
            row['truncated_combats'] = 0
            (batch / 'games.jsonl').write_text(json.dumps(row) + '\n')
            league.excluded_models = {'a'}
            league.recover()
            self.assertEqual(league.counts, Counter())
            self.assertEqual(league.excluded[0]['reason'], 'excluded_model')
            self.assertEqual(game_exclusion_reason(row, {'a'}), 'excluded_model')
            self.assertIsNone(game_exclusion_reason(row, {'absent'}))
            self.assertTrue((batch / 'games.jsonl').exists())



if __name__ == '__main__':
    unittest.main()
