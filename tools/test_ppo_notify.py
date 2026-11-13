import tempfile
import unittest
from pathlib import Path
import json
import subprocess
import sys
from unittest.mock import Mock
from ppo_notify import Feishu, events, read_env, image_card, source_script, rating_series


class NotificationTests(unittest.TestCase):
    def test_rating_plot_replaces_baseline_seeds_with_one_mean_series(self):
        data = {'ratings': {'a': {'run': 'a', 'iteration': 3000, 'elo': 900},
                            'b': {'run': 'b', 'iteration': 3000, 'elo': 1100},
                            'a_next': {'run': 'a', 'iteration': 3500, 'elo': 1200},
                            'c': {'run': 'candidate', 'iteration': 500, 'elo': 800}},
                'rating_groups': {'baseline': {'members': ['a', 'b'],
                    'points': [{'iteration': 3000, 'elo': 1000, 'se': 10, 'seeds': 2, 'expected_seeds': 2},
                               {'iteration': 3500, 'elo': 1200, 'se': 12, 'seeds': 1, 'expected_seeds': 2}]}}}
        series = rating_series(data)
        self.assertEqual(set(series), {'baseline (available-seed mean)', 'candidate'})
        self.assertEqual(series['baseline (available-seed mean)'], data['rating_groups']['baseline']['points'])
        # No rated checkpoints means no mean point, rather than individual curves.
        data['rating_groups']['baseline']['points'] = []
        self.assertEqual(set(rating_series(data)), {'candidate'})

    def test_read_env_does_not_execute_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'env'
            path.write_text("export FEISHU_APP_SECRET='$(not-a-command)' # comment\nFEISHU_DOMAIN=feishu\n")
            values = read_env(path)
            self.assertEqual(values['FEISHU_APP_SECRET'], '$(not-a-command)')
            self.assertEqual(values['FEISHU_DOMAIN'], 'feishu')

    def test_failures_are_debounced_and_recovery_is_reported(self):
        failures, previous = {}, {'gpu': 'running'}
        for _ in range(2):
            current, changed = events({'gpu': {'error': 'TimeoutExpired'}}, failures, previous)
            self.assertEqual(changed, {})
        current, changed = events({'gpu': {'error': 'TimeoutExpired'}}, failures, previous)
        self.assertEqual(changed, {'gpu': 'unreachable'})
        _, changed = events({'gpu': {'error': 'TimeoutExpired'}}, failures, current)
        self.assertEqual(changed, {})
        _, changed = events({'gpu': {'status': {'state': 'running'}}}, failures, current)
        self.assertEqual(changed, {'gpu': 'running'})
        _, changed = events({'gpu': {'status': {'state': 'complete'}}}, failures, previous)
        self.assertEqual(changed, {'gpu': 'complete'})

    def test_three_image_card_has_no_header_or_text_blocks(self):
        card=image_card(['scripted-key','teacher-key','elo-key'])
        self.assertNotIn('header',card)
        self.assertEqual([e['tag'] for e in card['elements']],['img']*3)
        self.assertEqual([e['img_key'] for e in card['elements']],['scripted-key','teacher-key','elo-key'])
        with self.assertRaises(ValueError):
            image_card(['only-one'])

    def test_failed_image_upload_never_sends_a_partial_or_text_card(self):
        client=Feishu.__new__(Feishu)
        client.login=Mock()
        client.upload=Mock(side_effect=['scripted-key',RuntimeError('upload failed')])
        client.request=Mock()
        pending={'images':[{'kind':kind,'path':kind+'.png'} for kind in ['scripted','teacher','elo']]}
        with self.assertRaises(RuntimeError):
            client.send(pending)
        client.request.assert_not_called()

    def test_eval_reader_deduplicates_archives_and_excludes_other_protocols(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a,b=root/'first',root/'continued'
            a.mkdir();b.mkdir()
            row={'iteration':500,'opponent':'scripted','games':1024,'mean_placement':1.5,
                 'standard_error':.02,'seed':10000000,'model_seed':123,'sample':False}
            (a/'eval-00500-scripted.json').write_text(json.dumps(row))
            (b/'eval-00500-scripted.json').write_text(json.dumps({**row,'games':2048}))
            (a/'eval-00500-scripted.command.json').write_text('[]')
            (a/'eval-00500-teacher.json').write_text(json.dumps({**row,'opponent':'teacher','sample':True}))
            (a/'eval-00600-scripted.json').write_text(json.dumps({**row,'iteration':600,'seed':20000000}))
            source={'files':{},'eval_runs':[{'name':'test','training_seed':123,'directories':[str(a),str(b)]}]}
            process=subprocess.run([sys.executable,'-'],input=source_script(source),capture_output=True,text=True,check=True)
            rows=json.loads(process.stdout)['evaluations']
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['games'],2048)
            self.assertEqual(rows[0]['run'],'test')
            source['eval_runs'][0]['exclude_iterations'] = [500]
            process=subprocess.run([sys.executable,'-'],input=source_script(source),capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(process.stdout)['evaluations'], [])
            self.assertTrue((b/'eval-00500-scripted.json').exists())



if __name__ == '__main__':
    unittest.main()
