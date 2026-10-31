// tavernreplay: offline harness for tavernplugin.
//
//   tavernreplay snapshot <Power.log> [--out snapshots.jsonl] [--every 200]
//   tavernreplay summary  <Power.log>
//   tavernreplay post     snapshots.jsonl [--url http://127.0.0.1:8765] [--limit 5]
//   tavernreplay server   [--server-dir DIR] [--port 8765] [--seconds 5]
//
// "snapshot" runs the live-game parser over a recorded log and writes one JSON
// snapshot per --every parsed lines (plus one whenever the phase changes), which
// is exactly what the in-game client posts to the bridge.
// "post" feeds those snapshots to a running server and prints the replies; it is
// the end-to-end check that client and server agree.
// "server" exercises ServerLauncher - the same code the game calls on startup -
// so auto-start can be tested without launching Hearthstone.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using TavernPlugin;

namespace TavernPlugin.Replay
{
    internal static class Program
    {
        private static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.Error.WriteLine("usage: tavernreplay <snapshot|summary|post|server> [file] [options]");
                return 2;
            }

            string mode = args[0];
            if (mode == "server") return Server(args);

            if (args.Length < 2)
            {
                Console.Error.WriteLine("usage: tavernreplay <snapshot|summary|post> <file> [options]");
                return 2;
            }

            string path = args[1];
            string outPath = null;
            string url = "http://127.0.0.1:8765";
            int every = 200;
            int limit = int.MaxValue;

            for (int i = 2; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--out": outPath = args[++i]; break;
                    case "--every": every = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                    case "--url": url = args[++i]; break;
                    case "--limit": limit = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                    default:
                        Console.Error.WriteLine("unknown option " + args[i]);
                        return 2;
                }
            }

            if (!File.Exists(path))
            {
                Console.Error.WriteLine("no such file: " + path);
                return 2;
            }

            switch (mode)
            {
                case "snapshot": return Snapshot(path, outPath, every);
                case "summary": return Summary(path);
                case "post": return Post(path, url, limit);
                default:
                    Console.Error.WriteLine("unknown mode " + mode);
                    return 2;
            }
        }

        // The same call the game makes in TavernPluginModule.Start().
        private static int Server(string[] args)
        {
            string dir = null;
            int port = 8765;
            int seconds = 5;
            for (int i = 1; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--server-dir": dir = args[++i]; break;
                    case "--port": port = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                    case "--seconds": seconds = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                    default:
                        Console.Error.WriteLine("unknown option " + args[i]);
                        return 2;
                }
            }

            var cfg = new ServerConfig { AutoStart = true, Port = port };
            if (dir != null) cfg.ServerDir = Path.GetFullPath(dir);

            Console.WriteLine("launcher  EnsureRunning(port " + port + (dir != null ? ", dir " + cfg.ServerDir : "") + ")");
            string status = ServerLauncher.EnsureRunning(cfg);
            Console.WriteLine("launcher  " + status);
            Console.WriteLine("launcher  pid=" + ServerLauncher.Pid + " started_by_us=" + ServerLauncher.StartedByUs);
            if (!string.IsNullOrEmpty(ServerLauncher.LastError))
                Console.WriteLine("launcher  last_error=" + ServerLauncher.LastError);

            using (var client = new BridgeClient("127.0.0.1", port))
            {
                string info;
                Console.WriteLine("health    " + (client.Ping(out info) ? "up" : "down") + " - " + info);

                if (seconds > 0)
                {
                    Console.WriteLine("sleeping  " + seconds + "s");
                    System.Threading.Thread.Sleep(seconds * 1000);
                }
                BridgeReply again = client.Get("/health");
                Console.WriteLine("health    still up=" + again.Ok + " taverngym=" + again.Str("taverngym", "?"));
            }

            Console.WriteLine("stopping  (this is what the game does on quit)");
            ServerLauncher.Stop();
            System.Threading.Thread.Sleep(1500);
            using (var probe = new BridgeClient("127.0.0.1", port))
            {
                string info;
                Console.WriteLine("after     " + (probe.Ping(out info) ? "STILL UP (bad)" : "down (ok)") + " - " + info);
            }
            return 0;
        }

        private static int Summary(string path)
        {            var parser = new PowerLogParser { LogFile = path };
            long lines = 0;
            string lastPhase = "";
            foreach (string line in File.ReadLines(path))
            {
                lines++;
                parser.Feed(line);
                if (lines % 20000 == 0) Console.Error.WriteLine("  " + lines + " lines");
            }
            Snapshot snap = parser.Build();
            Console.WriteLine("lines          " + lines);
            Console.WriteLine("games          " + parser.Games);
            Console.WriteLine("phase          " + snap.Phase);
            Console.WriteLine("turn           " + snap.Turn);
            Console.WriteLine("local player   " + snap.LocalPlayer + (snap.LocalPlayerConfident ? "" : " (guessed)"));
            Console.WriteLine("me             " + snap.Summary());
            Console.WriteLine("them player    " + snap.Them.PlayerId);
            Console.WriteLine("shop slots     " + snap.Shop.Count);
            Console.WriteLine();
            Console.WriteLine(snap.ToJson());
            GC.KeepAlive(lastPhase);
            return 0;
        }

        private static int Snapshot(string path, string outPath, int every)
        {
            var parser = new PowerLogParser { LogFile = path };
            TextWriter writer = outPath == null ? Console.Out : new StreamWriter(outPath, false, new UTF8Encoding(false));
            string lastPhase = "";
            long lines = 0;
            int emitted = 0;
            try
            {
                foreach (string line in File.ReadLines(path))
                {
                    lines++;
                    parser.Feed(line);
                    if (lines % every != 0) continue;
                    Snapshot snap = parser.Build();
                    if (snap.Phase != lastPhase || emitted == 0)
                    {
                        lastPhase = snap.Phase;
                        writer.WriteLine(snap.ToJson());
                        emitted++;
                    }
                }
                Snapshot final = parser.Build();
                writer.WriteLine(final.ToJson());
                emitted++;
            }
            finally
            {
                if (outPath != null) writer.Close();
            }
            Console.Error.WriteLine("parsed " + lines + " lines, wrote " + emitted + " snapshots");
            return 0;
        }

        private static int Post(string path, string url, int limit)
        {
            var uri = new Uri(url);
            using (var client = new TavernPlugin.BridgeClient(uri.Host, uri.Port))
            {
                int sent = 0, ok = 0;
                foreach (string line in File.ReadLines(path))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    if (sent >= limit) break;
                    sent++;
                    Dictionary<string, object> snap = JsonLite.ParseObject(line);
                    string phase = JsonLite.Str(snap, "game");
                    Console.WriteLine("--- #" + sent + " " + Short(line));

                    BridgeReply combat = client.Post("/combat", line);
                    Console.WriteLine("  /combat  " + Describe(combat));

                    BridgeReply align = client.Post("/align", line);
                    Console.WriteLine("  /align   " + Describe(align));

                    if (align.Ok && JsonLite.Bool(align.Body, "aligned", false)) ok++;
                }
                Console.WriteLine("posted " + sent + ", aligned " + ok);
                return 0;
            }
        }

        private static string Describe(BridgeReply reply)
        {
            if (!reply.Ok) return "ERROR " + reply.Error;
            if (reply.Body.ContainsKey("win"))
            {
                return string.Format(CultureInfo.InvariantCulture, "win {0:P0} tie {1:P0} loss {2:P0} dmg {3:0.0} ({4} sims, {5} ms)",
                    reply.Num("win"), reply.Num("tie"), reply.Num("loss"), reply.Num("avg_damage_taken"),
                    reply.Num("n"), reply.ElapsedMs);
            }
            if (reply.Body.ContainsKey("aligned"))
            {
                List<object> diffs = JsonLite.Arr(reply.Body, "diffs");
                List<object> issues = JsonLite.Arr(reply.Body, "issues");
                return string.Format(CultureInfo.InvariantCulture, "aligned={0} diffs={1} issues={2}",
                    JsonLite.Bool(reply.Body, "aligned"), diffs == null ? 0 : diffs.Count, issues == null ? 0 : issues.Count);
            }
            return "ok (" + reply.ElapsedMs + " ms)";
        }

        private static string Short(string json)
        {
            return json.Length <= 160 ? json : json.Substring(0, 160) + "...";
        }
    }
}
