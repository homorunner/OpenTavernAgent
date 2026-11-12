#!/usr/bin/env python3
"""Send periodic PPO progress and checkpoint-league Elo cards to an existing Feishu chat.

Runtime locations, SSH arguments and the credential-file path come from a private
JSON config. Credentials are read at send time, never copied into reports or logs.
Requires matplotlib to render exactly three plots: scripted, teacher and arena Elo.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_env(path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r'\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*)', line)
        if match:
            tokens = shlex.split(match[2], comments=True)
            result[match[1]] = ' '.join(tokens)
    return result


def source_script(source):
    files = {key: str(path) for key, path in source['files'].items()}
    runs = source.get('eval_runs', [])
    return ("import json,pathlib,re\nfiles=" + repr(files) + "\nruns=" + repr(runs) + "\n" + r"""
result={}
for key,name in files.items():
    p=pathlib.Path(name)
    result[key]=json.loads(p.read_text()) if p.exists() else None
points={}
for run in runs:
    for directory in run['directories']:
        for path in sorted(pathlib.Path(directory).glob('eval-*.json')):
            match=re.fullmatch(r'eval-(\d+)-(scripted|teacher)\.json',path.name)
            if not match:
                continue
            row=json.loads(path.read_text())
            if row['iteration']!=int(match[1]) or row['opponent']!=match[2]:
                raise ValueError('evaluation filename and metadata differ')
            if 'training_seed' in run and row.get('model_seed')!=run['training_seed']:
                raise ValueError('evaluation training seed differs')
            if row.get('sample',False) or row.get('seed')!=run.get('evaluation_seed',10000000):
                continue
            point={key:row[key] for key in ['iteration','opponent','games','mean_placement','standard_error']}
            point['run']=run['name']
            key=(point['run'],point['opponent'],point['iteration'])
            if key not in points or point['games']>=points[key]['games']:
                points[key]=point
result['evaluations']=list(points.values())
print(json.dumps(result))
""")


def collect_source(source):
    process = subprocess.run([*source['ssh'], 'python3 -'], input=source_script(source), text=True,
                             capture_output=True, timeout=30)
    if process.returncode:
        raise RuntimeError(f'read failed (exit {process.returncode})')
    return json.loads(process.stdout)


def collect(config):
    from concurrent.futures import ThreadPoolExecutor
    result = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {name: pool.submit(collect_source, source) for name, source in config['sources'].items()}
        for name, task in tasks.items():
            try:
                result[name] = task.result()
            except Exception as error:
                result[name] = {'error': type(error).__name__}
    return result


def events(snapshot, failures, previous):
    """Failures need three failed polls; explicit failure/completion is immediate."""
    current = {}
    for name, data in snapshot.items():
        status = data.get('status') or {}
        if data.get('error') or not status:
            failures[name] = failures.get(name, 0) + 1
            if failures[name] >= 3:
                current[name] = 'unreachable'
            elif name in previous:
                current[name] = previous[name]
            continue
        failures[name] = 0
        phase = status.get('state', 'unknown')
        if phase in ['failed', 'stopped', 'complete']:
            current[name] = phase
        else:
            stamp = status.get('updated_utc')
            age = (time.time() - datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()) if stamp else 0
            current[name] = 'stale' if age > 900 else 'running'
    changes = {name: state for name, state in current.items()
               if state != previous.get(name) and (state != 'running' or previous.get(name) in ['failed', 'stopped', 'stale', 'unreachable'])}
    return current, changes


PLOT_KINDS = ('scripted', 'teacher', 'elo')
CARD_FORMAT = 'three-plots-v1'


def series_label(name):
    return (name.replace('no_repeat_freeze_seed', 'Freeze blocked / ')
            .replace('control_seed', 'Control / ').replace('gpu384_seed', 'GPU384 / ')
            .replace('00000000', '00M'))


def plot_evaluation(snapshot, opponent, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = defaultdict(dict)
    for data in snapshot.values():
        for row in data.get('evaluations', []):
            if row['opponent'] == opponent:
                groups[row['run']][row['iteration']] = row
    legend_rows = (len(groups) + 3) // 4
    legend_height = max(1.0, .22 * legend_rows + .25)
    height = 5.5 + legend_height
    fig, axes = plt.subplots(1, 2, figsize=(14, height), gridspec_kw={'width_ratios': [1, 1.4]})
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h']
    zoom_rows = []
    for i, (name, points) in enumerate(sorted(groups.items())):
        rows = sorted(points.values(), key=lambda row: row['iteration'])
        for ax, selected in [(axes[0], rows), (axes[1], [r for r in rows if r['iteration'] >= 500])]:
            if not selected:
                continue
            ax.errorbar([r['iteration'] for r in selected], [r['mean_placement'] for r in selected],
                        yerr=[1.96*r['standard_error'] for r in selected],
                        marker=markers[i % len(markers)], markersize=2.5, capsize=0,
                        linewidth=1.1, alpha=.8, linestyle=['-', '--', '-.', ':'][i//5 % 4],
                        label=series_label(name))
        zoom_rows.extend(r for r in rows if r['iteration'] >= 500)
    axes[0].set_title('Full training history')
    axes[0].set_ylim(.9, 8.15)
    axes[1].set_title('Detail: iteration ≥ 500, placement ≤ 2.5')
    if zoom_rows:
        low=min(r['mean_placement']-1.96*r['standard_error'] for r in zoom_rows)
        high=max(r['mean_placement']+1.96*r['standard_error'] for r in zoom_rows)
        margin=max(.03,(high-low)*.08)
        axes[1].set_ylim(max(.9,min(low-margin,1.0)),min(2.5,max(1.1,high+margin)))
    for ax in axes:
        ax.set_xlabel('Training iteration')
        ax.set_ylabel('Mean placement (lower is better)')
        ax.grid(alpha=.25)
    if groups:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
    else:
        axes[0].text(.5,.5,'No completed evaluation data',transform=axes[0].transAxes,ha='center')
    timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    title='Scripted bot evaluation' if opponent=='scripted' else 'Teacher bot evaluation'
    fig.suptitle(f'{title} — against 7 bots\n{timestamp} | Error bars: 95% game-sampling intervals',fontsize=13)
    fig.tight_layout(rect=(0,legend_height/height,1,1))
    fig.savefig(path,dpi=135)
    plt.close(fig)


def plot_ratings(data, path):
    ratings = data.get('ratings') or {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for name, row in ratings.items():
        groups[row['run']].append(row)
    legend_height = max(1.0, .22 * ((len(groups) + 3) // 4) + .25)
    height = 6.0 + legend_height
    fig, ax = plt.subplots(figsize=(14, height))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h']
    for i, (name, rows) in enumerate(sorted(groups.items())):
        rows.sort(key=lambda r: r['iteration'])
        ax.errorbar([r['iteration'] for r in rows], [r['elo'] for r in rows],
                    yerr=[1.96*r['se'] for r in rows], marker=markers[i % len(markers)],
                    linestyle=['-', '--', '-.', ':'][i//5 % 4], capsize=2, markersize=4, label=series_label(name))
    meta = data.get('ratings_meta') or {}
    counts = [r['games'] for r in ratings.values()]
    timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    count_label=f'{min(counts)}–{max(counts)} games/model' if counts else 'No completed ratings'
    ax.set_title(f"Arena Elo: {len(ratings)} checkpoints, {meta.get('games', '?')} games\n{timestamp} | {count_label}; error bars: 95% whole-game bootstrap")
    ax.set_xlabel('Training iteration')
    ax.set_ylabel('Elo (fixed anchor = 1000)')
    ax.grid(alpha=.25)
    if groups:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=7, ncol=4)
    fig.tight_layout(rect=(0,legend_height/height,1,1))
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return True


def make_plots(snapshot, root, identifier):
    plots=[]
    for kind in PLOT_KINDS:
        path=root/f'{identifier}-{kind}.png'
        if kind=='elo':
            plot_ratings(snapshot.get('league',{}),path)
        else:
            plot_evaluation(snapshot,kind,path)
        plots.append({'kind':kind,'path':str(path)})
    return plots


def image_card(keys):
    if len(keys)!=3:
        raise ValueError('notification requires exactly three images')
    titles=['Scripted bot evaluation','Teacher bot evaluation','Arena Elo']
    return {'config':{'wide_screen_mode':True},'elements':[
        {'tag':'img','img_key':key,'alt':{'tag':'plain_text','content':title}}
        for key,title in zip(keys,titles)]}


class Feishu:
    def __init__(self, env_file):
        env = read_env(env_file)
        self.app_id = env['FEISHU_APP_ID']
        self.secret = env['FEISHU_APP_SECRET']
        self.destination = env['FEISHU_HOME_CHANNEL']
        if not self.app_id or not self.secret or not self.destination.startswith('oc_'):
            raise ValueError('Feishu app credentials and a default chat must be configured')
        self.base = {'feishu': 'https://open.feishu.cn', 'lark': 'https://open.larksuite.com'}[env.get('FEISHU_DOMAIN', 'feishu')]
        self.token = None

    def request(self, route, payload, content_type='application/json; charset=utf-8', authorized=True):
        headers = {'Content-Type': content_type}
        if authorized:
            headers['Authorization'] = 'Bearer ' + self.token
        request = urllib.request.Request(self.base + route, data=payload, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Feishu HTTP status {error.code}') from None
        except urllib.error.URLError:
            raise RuntimeError('Feishu network error') from None
        if result.get('code', 0) != 0:
            raise RuntimeError(f"Feishu API code {result['code']}")
        return result

    def login(self):
        result = self.request('/open-apis/auth/v3/tenant_access_token/internal',
                              json.dumps({'app_id': self.app_id, 'app_secret': self.secret}).encode(), authorized=False)
        self.token = result['tenant_access_token']

    def upload(self, path):
        boundary = 'tavernagent-' + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image_type"\r\n\r\nmessage\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="elo.png"\r\nContent-Type: image/png\r\n\r\n').encode()
        body += path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        result = self.request('/open-apis/im/v1/images', body, f'multipart/form-data; boundary={boundary}')
        return result['data']['image_key']

    def send(self, pending):
        plots=pending['images']
        if [p['kind'] for p in plots]!=list(PLOT_KINDS):
            raise ValueError('notification plots must be scripted, teacher, elo in that order')
        self.login()
        # Upload all three before publishing. A failed upload retries the whole card;
        # it must never send a partial card or a text fallback.
        keys=[self.upload(Path(plot['path'])) for plot in plots]
        card=image_card(keys)
        payload = {'receive_id': self.destination, 'msg_type': 'interactive', 'uuid': pending['uuid'],
                   'content': json.dumps(card, ensure_ascii=False)}
        result = self.request('/open-apis/im/v1/messages?receive_id_type=chat_id', json.dumps(payload, ensure_ascii=False).encode())
        return {'message_id': result['data']['message_id'], 'image_sent': True, 'images_sent': 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config['out']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if config.get('interval_seconds', 1800) < 60 or config.get('poll_seconds', 60) < 10:
        parser.error('notification interval must be >= 60 seconds, polling >= 10 seconds')
    lock = (root / 'notifier.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'last_sent': 0, 'events': {}, 'failures': {}}
    pending_path = root / 'pending.json'
    while True:
        snapshot = collect(config)
        save(root / 'latest-snapshot.json', snapshot)
        current, changed = events(snapshot, state.setdefault('failures', {}), state.get('events', {}))
        due = time.time() - state.get('last_sent', 0) >= config.get('interval_seconds', 1800)
        if args.dry_run:
            plots=make_plots(snapshot,root,'preview')
            save(root/'preview-images.json',plots)
            print(json.dumps({'images':plots}))
            return
        if pending_path.exists():
            old=json.loads(pending_path.read_text())
            if old.get('card_format')!=CARD_FORMAT:
                pending_path.rename(root/f'pending-legacy-{int(time.time())}.json')
        due=due or state.get('card_format')!=CARD_FORMAT
        if pending_path.exists() or due or changed:
            if not pending_path.exists():
                identifier = uuid.uuid4().hex
                plots=make_plots(snapshot,root,identifier)
                pending = {'uuid':identifier,'images':plots,'card_format':CARD_FORMAT,
                           'events':current,'created':time.time()}
                save(pending_path,pending)
            pending = json.loads(pending_path.read_text())
            try:
                receipt = Feishu(Path(config['credentials_env'])).send(pending)
                state.update(last_sent=time.time(), events=pending['events'], last_receipt=receipt, error=None, card_format=CARD_FORMAT)
                save(root / 'last-sent.json', {**pending, **receipt, 'sent_at': state['last_sent']})
                save(state_path, state)
                pending_path.unlink()
                print(json.dumps({'sent': True, 'images_sent': receipt['images_sent'], 'at': state['last_sent']}), flush=True)
            except Exception as error:
                state['error'] = str(error) if isinstance(error, RuntimeError) else type(error).__name__
                save(state_path, state)
                print(json.dumps({'sent': False, 'error': state['error']}), flush=True)
                if args.once:
                    raise SystemExit(1)
        state.update(last_checked=time.time(), next_regular=state.get('last_sent', 0) + config.get('interval_seconds', 1800))
        save(state_path, state)
        if args.once:
            return
        if (current and len(current) == len(config['sources']) and all(s == 'complete' for s in current.values())
                and state.get('events') == current and not pending_path.exists()):
            state['state'] = 'complete'
            save(state_path, state)
            return
        time.sleep(config.get('poll_seconds', 60))


if __name__ == '__main__':
    main()
