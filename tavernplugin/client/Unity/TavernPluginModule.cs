// TavernPlugin: the in-game half of the tavernplugin module.
//
// It samples the game's own state while a match is running, and asks the local
// bridge server (TavernGym) for
//   * the win probability of the combat that is about to happen,
//   * a recommended warband order during the recruit phase,
//   * an alignment check between the live game and the simulator.
//
// Every network call runs on a worker thread; OnGUI only shows cached text, so a
// slow or missing server can never stall the game.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using TavernPlugin;
using UnityEngine;

namespace HSLibrary
{
    public sealed class TavernPluginModule : MonoBehaviour
    {
        // ---- configuration ---------------------------------------------------
        public string Host = "127.0.0.1";
        public int Port = 8765;
        public bool Visible = true;
        public int CombatSims = 512;
        public int AdviseSims = 128;
        public float PollSeconds = 0.20f;

        // ---- state shared with the worker ------------------------------------
        private readonly object _lock = new object();
        private BridgeClient _client;
        private PowerLogTailer _tailer;
        private PowerLogParser _parser;
        private Thread _worker;
        private volatile bool _running;

        private Snapshot _snapshot;
        private string _snapshotJson = "";
        private string _lastKey = "";
        private string _serverInfo = "not contacted";
        private bool _serverUp;
        private ServerConfig _config;
        private string _launchLine = "";
        private string _combatLine = "";
        private string _adviceLine = "";
        private string _alignLine = "";
        private string _statusLine = "starting";
        private int _posts;
        private int _errors;
        private int _nextPollMs;
        private int _nextHealthMs;
        private int _nextSampleMs;
        private bool _sourceIsGame;
        private static readonly object _markLock = new object();

        public void Start()
        {
            // Canary first: a failure anywhere below must still leave a trace.
            // The game's Mono swallows exceptions from Start() into Player.log,
            // which is often empty, so write our own breadcrumbs.
            GameLog.Write("[tavern] Start enter");
            try
            {
                _parser = new PowerLogParser();
                _tailer = new PowerLogTailer();
                _client = new BridgeClient(Host, Port);
                GameLog.Write("[tavern] client ok");
                _config = ServerConfig.Load();
                _config.Port = Port;          // the inspector overrides the file
                GameLog.Write("[tavern] config ok");
                _running = true;
                _worker = new Thread(WorkerLoop);
                _worker.IsBackground = true;
                _worker.Name = "TavernPlugin";
                _worker.Start();
                GameLog.Write("[tavern] worker started; server " + _client.BaseUrl);
            }
            catch (Exception exc)
            {
                GameLog.Write("[tavern] Start FAILED: " + exc);
                SetStatus("start failed: " + exc.GetType().Name);
            }
        }

        public void Update()
        {
            if (Input.GetKeyDown(KeyCode.F8)) Visible = !Visible;
            if (Input.GetKeyDown(KeyCode.F9)) DumpNow();

            // The live source is the game itself; the log is only a fallback for
            // when no match is loaded. Sampling happens here because GameState is
            // not thread safe, and 4 Hz is plenty for an overlay.
            int now = Environment.TickCount;
            if (now < _nextSampleMs) return;
            _nextSampleMs = now + 250;

            Snapshot snap;
            string error;
            if (GameStateSource.Capture(out snap, out error))
            {
                string json = snap.ToJson();
                lock (_lock)
                {
                    _snapshot = snap;
                    _snapshotJson = json;
                    _sourceIsGame = true;
                }
            }
            else if (error != null && error.Length > 0 && error.IndexOf("no GameState") < 0)
            {
                SetStatus("state: " + error);
            }
        }

        private void DumpNow()
        {
            string error;
            bool ok = GameStateDump.Capture(out error);
            if (!ok)
            {
                GameLog.Write("[dump] failed: " + error);
                SetStatus("dump failed: " + error);
                return;
            }
            string baseName = Path.GetFileNameWithoutExtension(GameStateDump.LastPath);
            string folder = Path.GetDirectoryName(GameStateDump.LastPath);
            string png = Path.Combine(folder, baseName + ".png");
            StartCoroutine(CaptureScreenshot(png));

            // the exact payload the server sees, next to the raw probe dump
            try
            {
                string snapshotJson;
                lock (_lock) { snapshotJson = _snapshotJson; }
                if (!string.IsNullOrEmpty(snapshotJson))
                {
                    using (var stream = new FileStream(Path.Combine(folder, baseName + ".snapshot.json"),
                                                       FileMode.Create, FileAccess.Write, FileShare.Read))
                    using (var writer = new StreamWriter(stream, new UTF8Encoding(false)))
                    {
                        writer.Write(snapshotJson);
                    }
                }
            }
            catch (Exception exc)
            {
                GameLog.Write("[dump] snapshot copy failed: " + exc.GetType().Name);
            }
            GameLog.Write("[dump] wrote " + GameStateDump.LastPath + " + " + png);
            SetStatus("dump -> " + Path.GetFileName(GameStateDump.LastPath));
        }

        // Screenshot next to the dump so the JSON can be checked against pixels.
        private System.Collections.IEnumerator CaptureScreenshot(string path)
        {
            yield return new WaitForEndOfFrame();
            try
            {
                var tex = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
                tex.Apply();
                byte[] png = tex.EncodeToPNG();
                Destroy(tex);
                using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
                {
                    stream.Write(png, 0, png.Length);
                }
                GameLog.Write("[dump] screenshot " + path + " (" + png.Length + " bytes)");
            }
            catch (Exception exc)
            {
                GameLog.Write("[dump] screenshot failed: " + exc);
            }
        }

        public void OnDestroy()
        {
            try
            {
                _running = false;
                if (_tailer != null) _tailer.Dispose();
                if (_client != null) _client.Dispose();
                ServerLauncher.Stop();
            }
            catch (Exception exc)
            {
                // teardown must not throw on the way out
                GameLog.Write("[tavern] shutdown failed: " + exc.GetType().Name);
            }
        }

        public void OnApplicationQuit()
        {
            OnDestroy();
        }

        // ---- worker ----------------------------------------------------------

        private void WorkerLoop()
        {
            bool fromStart = false;   // follow the live game, not the whole history

            // Both of these belong off the main thread: the probe touches the file
            // system, and the launcher waits for a port. Together they used to stall
            // Start() for several seconds.
            SelfCheck.Run();
            GameLog.Write("[tavern] self-check: " + SelfCheck.Failures + " failure(s)");

            // Start the bridge server once, off the main thread: Process.Start plus
            // waiting for the port to open can take a few seconds.
            string launch = ServerLauncher.EnsureRunning(_config);
            lock (_lock) { _launchLine = "server: " + launch; }
            Log("server launcher: " + launch);

            while (_running)
            {
                try
                {
                    if (_tailer.Path == null && !_tailer.FollowNewest(fromStart))
                    {
                        PowerLogLocator.EnsureLogConfig();
                        SetStatus("waiting for Power.log (start a Battlegrounds game)");
                        Thread.Sleep(1000);
                        continue;
                    }

                    List<string> lines = _tailer.ReadLines();
                    if (lines.Count > 0)
                    {
                        for (int i = 0; i < lines.Count; i++) _parser.Feed(lines[i]);
                        Snapshot snap = _parser.Build();
                        string json = snap.ToJson();
                        lock (_lock)
                        {
                            if (!_sourceIsGame)
                            {
                                _snapshot = snap;
                                _snapshotJson = json;
                            }
                        }
                    }

                    // Unity's Time is main-thread only, so the worker uses the OS clock
                    int now = Environment.TickCount;
                    if (now >= _nextPollMs)
                    {
                        _nextPollMs = now + (int)(PollSeconds * 1000f);
                        if (now >= _nextHealthMs)
                        {
                            _nextHealthMs = now + 5000;
                            ProbeServer();
                        }
                        MaybeQuery();
                    }
                }
                catch (Exception exc)
                {
                    lock (_lock) { _errors++; }
                    SetStatus("worker error: " + exc.GetType().Name + ": " + exc.Message);
                    GameLog.Write("[tavern] worker error: " + exc);
                    Thread.Sleep(500);
                }
                Thread.Sleep(120);
            }
        }

        private void ProbeServer()
        {
            string info;
            bool up = _client.Ping(out info);
            lock (_lock)
            {
                _serverUp = up;
                _serverInfo = info;
            }
        }

        // Ask the server only when the state really changed.
        private void MaybeQuery()
        {
            Snapshot snap;
            string json;
            lock (_lock)
            {
                snap = _snapshot;
                json = _snapshotJson;
            }
            if (snap == null) { SetStatus("no game state yet"); return; }
            if (!_serverUp) { SetStatus("server unreachable: " + _serverInfo); return; }

            string key = Key(snap);
            if (key == _lastKey) return;
            _lastKey = key;

            if (snap.Phase == "combat")
            {
                string body = WithSims(json, CombatSims);
                BridgeReply reply = _client.Post("/combat", body);
                if (!reply.Ok) { Fail(reply); return; }
                double win = reply.Num("win"), tie = reply.Num("tie"), loss = reply.Num("loss");
                double dmg = reply.Num("avg_damage_taken");
                double lethal = reply.Num("lethal_risk");
                lock (_lock)
                {
                    _posts++;
                    _combatLine = string.Format(CultureInfo.InvariantCulture,
                        "win {0:0}%  tie {1:0}%  loss {2:0}%   dmg {3:0.0}{4}   ({5} sims, {6} ms)",
                        win * 100, tie * 100, loss * 100, dmg,
                        lethal > 0.02 ? string.Format("  lethal {0:0}%", lethal * 100) : "", reply.Num("n"), reply.ElapsedMs);
                    _alignLine = Align(json);
                }
                SetStatus("combat");
            }
            else if (snap.Phase == "recruit")
            {
                BridgeReply reply = _client.Post("/advise", WithSims(json, AdviseSims));
                lock (_lock)
                {
                    _posts++;
                    if (!reply.Ok)
                    {
                        _adviceLine = "advice unavailable: " + reply.Error;
                    }
                    else
                    {
                        _adviceLine = BestAdvice(reply);
                    }
                }
                SetStatus("recruit");
            }
            else
            {
                SetStatus("waiting for a Battlegrounds game");
            }
        }

        private string Align(string json)
        {
            BridgeReply reply = _client.Post("/align", json);
            if (!reply.Ok) return "align: " + reply.Error;
            bool aligned = JsonLite.Bool(reply.Body, "aligned", false);
            List<object> diffs = JsonLite.Arr(reply.Body, "diffs");
            int n = diffs == null ? 0 : diffs.Count;
            return aligned ? "aligned with TavernGym" : ("align: " + n + " difference(s)");
        }

        private static string BestAdvice(BridgeReply reply)
        {
            List<object> recs = JsonLite.Arr(reply.Body, "recommendations");
            if (recs == null || recs.Count == 0) return JsonLite.Str(reply.Body, "note", "no recommendation");
            var sb = new StringBuilder();
            for (int i = 0; i < Math.Min(3, recs.Count); i++)
            {
                var rec = recs[i] as Dictionary<string, object>;
                if (rec == null) continue;
                if (i > 0) sb.Append("   |   ");
                sb.Append(i == 0 ? "> " : "  ");
                sb.Append(JsonLite.Str(rec, "text"));
                sb.Append(' ');
                sb.Append((JsonLite.Num(rec, "win") * 100).ToString("0", CultureInfo.InvariantCulture));
                sb.Append('%');
            }
            return sb.ToString();
        }

        private static string WithSims(string json, int sims)
        {
            // the snapshot is an object; inject "n" without re-serialising it
            if (string.IsNullOrEmpty(json) || json[0] != '{') return json;
            return "{\"n\":" + sims.ToString(CultureInfo.InvariantCulture) + "," + json.Substring(1);
        }

        private static string Key(Snapshot s)
        {
            var sb = new StringBuilder();
            sb.Append(s.Phase).Append('|').Append(s.Turn).Append('|').Append(s.LocalPlayer);
            Append(sb, s.Me);
            Append(sb, s.Them);
            return sb.ToString();
        }

        private static void Append(StringBuilder sb, SideInfo side)
        {
            sb.Append('|');
            for (int i = 0; i < side.Board.Count; i++)
            {
                MinionInfo m = side.Board[i];
                sb.Append(m.Card).Append(':').Append(m.Atk).Append('/').Append(m.Hp)
                  .Append(m.Taunt ? "T" : "").Append(m.DivineShield ? "D" : "").Append(m.Poisonous ? "P" : "")
                  .Append(',');
            }
        }


        private void Fail(BridgeReply reply)
        {
            lock (_lock) { _errors++; }
            SetStatus("server error: " + reply.Error);
            GameLog.Write("[tavern] server error: " + reply.Error);
        }

        private void SetStatus(string text)
        {
            lock (_lock) { _statusLine = text; }
        }

        private static void Log(string message)
        {
            GameLog.Write("[tavern] " + message);
        }

        // ---- overlay ---------------------------------------------------------

        public void OnGUI()
        {
            if (!Visible) return;

            GUI.Box(new Rect(Screen.width - 490, 60, 470, 176), "TavernPlugin (F8)");

            Snapshot snap;
            string status, combat, advice, align, info, launch;
            bool up;
            int posts, errors;
            lock (_lock)
            {
                snap = _snapshot; status = _statusLine; combat = _combatLine; advice = _adviceLine;
                align = _alignLine; info = _serverInfo; up = _serverUp; posts = _posts; errors = _errors;
                launch = _launchLine;
            }

            string[] rows =
            {
                launch.Length > 0 ? launch : "server: starting",
                up ? "health: " + info : "health: offline (" + info + ")",
                "state: " + (snap == null ? "-" : snap.Summary()),
                "phase: " + status,
                combat.Length > 0 ? "combat: " + combat : "",
                advice.Length > 0 ? "advice: " + advice : "",
                align,
                string.Format(CultureInfo.InvariantCulture, "queries {0}  errors {1}", posts, errors),
            };

            float y = 88f;
            for (int i = 0; i < rows.Length; i++)
            {
                if (rows[i].Length == 0) continue;
                GUI.Label(new Rect(Screen.width - 480, y, 456, 20), rows[i]);
                y += 18f;
            }
        }
    }
}







