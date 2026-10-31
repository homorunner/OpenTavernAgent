// Startup capability probe for Hearthstone's Mono.
//
// The game ships a stripped mscorlib: a member the game itself never calls can be
// missing, and a method body that mentions one fails at JIT time — the whole method
// throws before its first statement runs. That failure mode is why this exists: when
// a patch removes something, the log should name it instead of surfacing as a
// half-dead overlay.
//
// Two rules keep the cost near zero (it runs on the worker thread, after Start):
//   * only probe what the shipped code actually calls — the catalogue of known-bad
//     members belongs in docs/ALIGNMENT.md §4, not in a startup test;
//   * no network round trips (connecting to a closed port costs seconds on Windows);
//     transport problems surface through the bridge's own error path anyway.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

namespace TavernPlugin
{
    public static class SelfCheck
    {
        public static int Failures { get; private set; }

        public static void Run()
        {
            Failures = 0;
            GameLog.Write("[check] --- runtime capabilities ---");

            // logging and files (GameLog, dumps, config). The probe file is written
            // first so the reads stay on a few bytes instead of the real log.
            string probeFile = GameLog.FilePath + ".probe";
            Probe("File.Exists", delegate { File.Exists(GameLog.FilePath); });
            Probe("FileStream+StreamWriter", delegate
            {
                using (var s = new FileStream(probeFile, FileMode.Create, FileAccess.Write, FileShare.ReadWrite))
                using (var w = new StreamWriter(s, new UTF8Encoding(false)))
                {
                    w.Write("probe");
                }
            });
            Probe("File.ReadAllText(1)", delegate { File.ReadAllText(probeFile); });
            Probe("StreamReader(Stream,Encoding)", delegate
            {
                using (var s = new FileStream(probeFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var r = new StreamReader(s, Encoding.UTF8))
                {
                    r.ReadToEnd();
                }
            });
            Probe("FileStream.Length/Seek/Read", delegate
            {
                using (var fs = new FileStream(probeFile, FileMode.Open, FileAccess.Read,
                                               FileShare.ReadWrite | FileShare.Delete))
                {
                    long length = fs.Length;
                    fs.Seek(0, SeekOrigin.Begin);
                    var buffer = new byte[16];
                    fs.Read(buffer, 0, buffer.Length);
                    if (length < 0) throw new Exception("impossible");
                }
            });            Probe("Directory.CreateDirectory", delegate
            {
                Directory.CreateDirectory(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                                                       "TavernPlugin", "probe"));
            });
            Probe("Directory.GetDirectories", delegate
            {
                Directory.GetDirectories(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments));
            });
            Probe("Directory.GetFiles", delegate
            {
                Directory.GetFiles(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "*.log");
            });
            Probe("Path.GetFileName/GetFullPath", delegate
            {
                Path.GetFullPath(Path.GetFileName(GameLog.FilePath));
            });

            // parsing and formatting (PowerLogParser, Snapshot, BridgeClient)
            Probe("Regex(named group)", delegate { new Regex("a(?<b>c)").IsMatch("ac"); });
            Probe("string.Substring", delegate { "abc".Substring(1, 1); });
            Probe("stringBuilder", delegate { new StringBuilder().Append("x").Append(1).ToString(); });
            Probe("int.Parse(NumberStyles)", delegate
            {
                int.Parse("42", NumberStyles.Integer, CultureInfo.InvariantCulture);
            });
            Probe("double.ToString(R,ci)", delegate { (1.5).ToString("R", CultureInfo.InvariantCulture); });
            Probe("string.Format(ci)", delegate { string.Format(CultureInfo.InvariantCulture, "{0}", 1); });
            Probe("DateTime.ToString(fmt,ci)", delegate
            {
                DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture);
            });
            Probe("Encoding.UTF8.GetBytes/GetString", delegate
            {
                Encoding.UTF8.GetString(Encoding.UTF8.GetBytes("x"));
            });
            Probe("Dictionary.TryGetValue", delegate
            {
                var map = new Dictionary<string, string>(StringComparer.Ordinal);
                map["a"] = "b";
                string value;
                map.TryGetValue("a", out value);
            });
            Probe("List.Sort(Comparison)", delegate
            {
                var list = new List<int> { 2, 1 };
                list.Sort(delegate (int a, int b) { return a.CompareTo(b); });
            });

            // threading and interop (TavernPluginModule, ServerLauncher, NativeProcess)
            Probe("Thread", delegate
            {
                var t = new Thread(delegate () { });
                t.IsBackground = true;
                t.Start();
                t.Join();
            });
            Probe("Marshal.SizeOf", delegate { System.Runtime.InteropServices.Marshal.SizeOf(typeof(int)); });
            Probe("IDisposable.Dispose", delegate
            {
                IDisposable d = new MemoryStream();
                d.Dispose();
            });

            GameLog.Write("[check] --- end, " + Failures + " failure(s) ---");
            try { File.Delete(probeFile); }
            catch (Exception) { }
        }

        private static void Probe(string name, Action action)
        {
            try
            {
                action();
            }
            catch (Exception exc)
            {
                Failures++;
                GameLog.Write("[check] FAIL " + name + " -> " + exc.GetType().Name + ": " + exc.Message);
            }
        }
    }
}

