"""Statistical and scheduling invariants for checkpoint league evaluation."""
import unittest
import json
import tempfile
from pathlib import Path
from collections import Counter
import numpy as np
from league_elo import ratings_with_bootstrap, validate_games
from ppo_league import League, schedule


class LeagueTests(unittest.TestCase):
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

    def test_unidentifiable_or_invalid_rankings_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disconnected'):
            validate_games([(['a', 'b'], [1, 2]), (['c', 'd'], [1, 2])], list('abcd'))
        for game in [(['a', 'a'], [1, 2]), (['a', 'b'], [1, 1]), (['a', 'b'], [0, 1])]:
            with self.assertRaises(ValueError):
                validate_games([game], ['a', 'b'])

    def test_recovery_ignores_partial_and_never_counts_completed_games_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            league = League.__new__(League)
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


if __name__ == '__main__':
    unittest.main()
