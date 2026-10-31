// Starts the bridge server together with the game, so the overlay is useful
// without a second terminal.
//
// Launching goes through NativeProcess (CreateProcessW) on purpose: classic BCL
// Process.Kill() reports MissingMethodException inside Hearthstone's Mono, so the
// whole System.Diagnostics.Process surface is treated as unavailable.
//
//   python <server_dir>\server.py --port 8765 --parent-pid <game pid>
//
// The server watches that pid and exits on its own when the game dies, so a
// crash cannot leave python running forever.
//
// Configuration lives in %USERPROFILE%\Documents\TavernPlugin\config.json.

using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace TavernPlugin
{
    public sealed class ServerConfig
    {
        public bool AutoStart = true;
        public int Port = 8765;
        public string Python = "python";
        public string ServerDir = "";
        public readonly List<string> ExtraArgs = new List<string>();

        public static string DefaultPath()
        {
            string docs = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            return Path.Combine(docs, "TavernPlugin", "config.json");
        }

        public static ServerConfig Load(string path = null)
        {
            if (path == null) path = DefaultPath();
            var cfg = new ServerConfig();
            try
            {
                if (!File.Exists(path))
                {
                    cfg.Save(path);
                    return cfg;
                }
                string text = ReadAll(path);
                Dictionary<string, object> json = JsonLite.ParseObject(text);
                if (json != null)
                {
                    cfg.AutoStart = JsonLite.Bool(json, "auto_start", cfg.AutoStart);
                    cfg.Port = (int)JsonLite.Num(json, "port", cfg.Port);
                    cfg.Python = JsonLite.Str(json, "python", cfg.Python);
                    cfg.ServerDir = JsonLite.Str(json, "server_dir", cfg.ServerDir);
                    List<object> extra = JsonLite.Arr(json, "extra_args");
                    if (extra != null)
                        foreach (object a in extra)
                            if (a is string) cfg.ExtraArgs.Add((string)a);
                }
            }
            catch (Exception)
            {
                // a broken config must never stop the plugin: fall back to defaults
            }
            return cfg;
        }

        public void Save(string path = null)
        {
            if (path == null) path = DefaultPath();
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                var w = new JsonWriter();
                w.BeginObject();
                w.Field("auto_start", AutoStart);
                w.Field("port", Port);
                w.Field("python", Python);
                w.Field("server_dir", ServerDir);
                w.Key("extra_args").BeginArray();
                foreach (string a in ExtraArgs) w.Value(a);
                w.EndArray();
                w.EndObject();
                WriteAll(path, w.ToString());
            }
            catch (Exception)
            {
            }
        }

        // File.ReadAllText / WriteAllText overloads differ between the profiles this
        // runs on, so go through StreamReader/StreamWriter explicitly.
        private static string ReadAll(string path)
        {
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var reader = new StreamReader(stream, Encoding.UTF8))
            {
                return reader.ReadToEnd();
            }
        }

        private static void WriteAll(string path, string text)
        {
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
            using (var writer = new StreamWriter(stream, new UTF8Encoding(false)))
            {
                writer.Write(text);
            }
        }
    }

    // Locates, launches and stops the Python bridge. All failures are reported, never thrown.
    public static class ServerLauncher
    {
        private static readonly object _lock = new object();

        public static string LastError { get; private set; }
        public static int Pid { get; private set; }
        public static bool StartedByUs { get; private set; }
        public static string ServerDir { get; private set; }

        // Make sure a server answers on the configured port. Returns a short status
        // line for the overlay; never throws.
        public static string EnsureRunning(ServerConfig cfg)
        {
            try
            {
                if (cfg == null) cfg = ServerConfig.Load();
                lock (_lock)
                {
                    if (PortOpen("127.0.0.1", cfg.Port, 400))
                    {
                        StartedByUs = false;
                        return "already running on " + cfg.Port;
                    }
                    if (!cfg.AutoStart)
                    {
                        LastError = "auto_start is off in config.json";
                        return "not started (auto_start off)";
                    }

                    ServerDir = ResolveServerDir(cfg);
                    if (ServerDir == null)
                    {
                        LastError = "server.py not found (set server_dir in " + ServerConfig.DefaultPath() + ")";
                        return "not started: server.py not found";
                    }

                    string script = Path.Combine(ServerDir, "server.py");
                    string args = Quote(script)
                                  + " --port " + cfg.Port
                                  + " --parent-pid " + GamePid();
                    for (int i = 0; i < cfg.ExtraArgs.Count; i++) args += " " + Quote(cfg.ExtraArgs[i]);

                    string usedPython = null;
                    string lastError = null;
                    foreach (string python in PythonCandidates(cfg.Python))
                    {
                        string error;
                        int pid = NativeProcess.Start(python, args, ServerDir, out error);
                        if (pid == 0)
                        {
                            lastError = error;
                            continue;
                        }
                        usedPython = python;
                        Pid = pid;
                        StartedByUs = true;
                        GameLog.Write("[launcher] started " + python + " as pid " + pid);
                        break;
                    }

                    if (Pid == 0)
                    {
                        LastError = "cannot start python (" + (lastError ?? "not found") + ")";
                        return "not started: python not found";
                    }
                    GameLog.Write("[launcher] using python " + usedPython + ", dir " + ServerDir);

                    // give it a moment: the first request would otherwise race the startup
                    for (int i = 0; i < 40; i++)
                    {
                        if (PortOpen("127.0.0.1", cfg.Port, 250)) return "auto-started (pid " + Pid + ")";
                        Thread.Sleep(250);
                    }
                    LastError = "server started but the port never opened";
                    return "auto-started, port " + cfg.Port + " not open yet";
                }
            }
            catch (Exception exc)
            {
                LastError = exc.GetType().Name + ": " + exc.Message;
                GameLog.Write("[launcher] FAILED " + exc);
                return "auto-start failed: " + exc.Message;
            }
        }

        // Stop the server if this session started it (the game is closing).
        public static void Stop()
        {
            try
            {
                lock (_lock)
                {
                    if (!StartedByUs || Pid == 0) return;
                    GameLog.Write("[launcher] stopping pid " + Pid);
                    NativeProcess.Kill(Pid);
                    StartedByUs = false;
                    Pid = 0;
                }
            }
            catch (Exception)
            {
            }
        }

        public static int GamePid()
        {
            try
            {
                return System.Diagnostics.Process.GetCurrentProcess().Id;
            }
            catch (Exception)
            {
                return 0;
            }
        }

        // Python interpreters to try, in order. The config value always wins.
        private static List<string> PythonCandidates(string configured)
        {
            var list = new List<string>();
            if (!string.IsNullOrEmpty(configured)) list.Add(configured);

            // the game's PATH may not contain python, so probe the usual install roots
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            var roots = new List<string>();
            if (!string.IsNullOrEmpty(local)) roots.Add(Path.Combine(local, "Programs", "Python"));
            roots.Add(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Python"));
            roots.Add(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Python"));
            roots.Add("C:\\Program Files\\Python313");
            roots.Add("C:\\Program Files\\Python312");

            foreach (string root in roots)
            {
                try
                {
                    if (File.Exists(Path.Combine(root, "python.exe")))
                    {
                        string direct = Path.Combine(root, "python.exe");
                        if (!list.Contains(direct)) list.Add(direct);
                        continue;
                    }
                    if (!Directory.Exists(root)) continue;
                    foreach (string dir in Directory.GetDirectories(root))
                    {
                        string exe = Path.Combine(dir, "python.exe");
                        if (File.Exists(exe) && !list.Contains(exe)) list.Add(exe);
                    }
                }
                catch (Exception)
                {
                }
            }

            list.Add("python");
            list.Add("py");
            return list;
        }

        private static string ResolveServerDir(ServerConfig cfg)
        {
            var candidates = new List<string>();
            if (!string.IsNullOrEmpty(cfg.ServerDir)) candidates.Add(cfg.ServerDir);

            string here = "";
            try { here = Path.GetDirectoryName(typeof(ServerLauncher).Assembly.Location); }
            catch (Exception) { }
            if (!string.IsNullOrEmpty(here))
            {
                candidates.Add(Path.Combine(here, "tavernplugin", "server"));
                candidates.Add(Path.Combine(here, "server"));
            }

            string docs = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            candidates.Add(Path.Combine(docs, "TavernPlugin", "server"));

            foreach (string dir in candidates)
            {
                try
                {
                    if (File.Exists(Path.Combine(dir, "server.py"))) return Path.GetFullPath(dir);
                }
                catch (Exception)
                {
                }
            }
            return null;
        }

        private static bool PortOpen(string host, int port, int timeoutMs)
        {
            // TcpClient.Close() is stripped from the game's mscorlib, so the socket is
            // released through IDisposable (and left to the GC if even that fails).
            // A synchronous Connect on loopback returns immediately either way.
            TcpClient client = null;
            try
            {
                client = new TcpClient();
                client.Connect(host, port);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
            finally
            {
                if (client != null)
                {
                    try { ((IDisposable)client).Dispose(); } catch (Exception) { }
                }
            }
        }

        private static string Quote(string s)
        {
            return "\"" + s + "\"";
        }
    }
}

