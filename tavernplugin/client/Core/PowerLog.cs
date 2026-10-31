// Finds Hearthstone's Power.log and tails it.
//
// Hearthstone writes one directory per session:
//   <install>\Logs\Hearthstone_2026_10_31_17_07_27\Power.log
// and renames the file to Power_old.log when it starts a new one. Reading needs
// the log enabled in %LOCALAPPDATA%\Blizzard\Hearthstone\log.config:
//   [Power]
//   LogLevel=1
//   FilePrinting=true
//   Verbose=true

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TavernPlugin
{
    public static class PowerLogLocator
    {
        public static IEnumerable<string> CandidateRoots()
        {
            string programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
            if (!string.IsNullOrEmpty(programFilesX86))
                yield return Path.Combine(programFilesX86, "Hearthstone", "Logs");

            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            if (!string.IsNullOrEmpty(programFiles))
                yield return Path.Combine(programFiles, "Hearthstone", "Logs");

            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            if (!string.IsNullOrEmpty(local))
                yield return Path.Combine(local, "Blizzard", "Hearthstone", "Logs");
        }

            public static string FindNewest()
        {
            foreach (string root in CandidateRoots())
            {
                string found = FindIn(root);
                if (found != null) return found;
            }
            return null;
        }

        public static string FindIn(string root)
        {
            try
            {
                if (!Directory.Exists(root)) return null;
                // session folders are named Hearthstone_yyyy_MM_dd_HH_mm_ss, so the
                // ordinal maximum is the newest session. Written as a loop: the LINQ
                // OrderByDescending(..., IComparer) overload is stripped from the
                // game's mscorlib (MissingMethodException at runtime).
                string newest = null;
                foreach (string dir in Directory.GetDirectories(root, "Hearthstone_*"))
                {
                    string name = Path.GetFileName(dir);
                    if (newest == null || Text.CompareOrdinal(name, Path.GetFileName(newest)) > 0) newest = dir;
                }
                if (newest == null) return null;

                string power = Path.Combine(newest, "Power.log");
                if (File.Exists(power)) return power;
                string old = Path.Combine(newest, "Power_old.log");
                if (File.Exists(old)) return old;

                // a restart keeps the directory but renames the file
                string best = null;
                foreach (string file in Directory.GetFiles(newest, "Power*.log"))
                    if (best == null || Text.CompareOrdinal(file, best) > 0) best = file;
                return best;
            }
            catch (IOException) { return null; }
            catch (UnauthorizedAccessException) { return null; }
        }

        // Write the log.config stanza that makes Power.log appear.
        public static bool EnsureLogConfig()
        {
            try
            {
                string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                if (string.IsNullOrEmpty(local)) return false;
                string dir = Path.Combine(local, "Blizzard", "Hearthstone");
                if (!Directory.Exists(dir)) return false;
                string path = Path.Combine(dir, "log.config");
                string text = File.Exists(path) ? File.ReadAllText(path) : "";
                if (Text.IndexOf(text, "[Power]") >= 0)
                    return true;
                File.AppendAllText(path, Environment.NewLine + "[Power]" + Environment.NewLine +
                                         "LogLevel=1" + Environment.NewLine +
                                         "FilePrinting=true" + Environment.NewLine +
                                         "ConsolePrinting=false" + Environment.NewLine +
                                         "ScreenPrinting=false" + Environment.NewLine +
                                         "Verbose=true" + Environment.NewLine);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }
    }

    // Tails a growing UTF-8 log file. The game keeps the file open for writing, so
    // it is opened with FileShare.ReadWrite|Delete; a failed read leaves the offset
    // alone and is retried on the next poll.
    public sealed class PowerLogTailer : IDisposable
    {
        private readonly List<string> _pending = new List<string>();
        private string _path;
        private long _offset = -1;
        private FileStream _stream;
        private readonly StringBuilder _partial = new StringBuilder();
        private readonly Dictionary<string, long> _sizes = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);

        public string Path { get { return _path; } }
        public long Offset { get { return _offset; } }

        public bool Attach(string path, bool fromStart)
        {
            if (string.IsNullOrEmpty(path)) return false;
            if (Text.EqualsIgnoreCase(path, _path) && _stream != null) return true;
            Close();
            _path = path;
            try
            {
                _stream = new FileStream(path, FileMode.Open, FileAccess.Read,
                                         FileShare.ReadWrite | FileShare.Delete, 64 * 1024, FileOptions.SequentialScan);
                _offset = fromStart ? 0 : _stream.Length;
                _stream.Seek(_offset, SeekOrigin.Begin);
                _partial.Length = 0;
                return true;
            }
            catch (Exception)
            {
                Close();
                return false;
            }
        }

        public bool FollowNewest(bool fromStart)
        {
            string newest = PowerLogLocator.FindNewest();
            if (newest == null) return _stream != null;
            return Attach(newest, fromStart);
        }

        public List<string> ReadLines()
        {
            var lines = new List<string>();
            if (_stream == null) return lines;
            try
            {
                if (_stream.Length < _offset)
                {
                    // the game rotated the file under us
                    _offset = 0;
                    _stream.Seek(0, SeekOrigin.Begin);
                    _partial.Length = 0;
                }
                var buffer = new byte[64 * 1024];
                int read;
                while ((read = _stream.Read(buffer, 0, buffer.Length)) > 0)
                {
                    _offset += read;
                    string chunk = Text.DecodeUtf8(buffer, read);
                    for (int i = 0; i < chunk.Length; i++)
                    {
                        char c = chunk[i];
                        if (c == '\n')
                        {
                            lines.Add(_partial.ToString());
                            _partial.Length = 0;
                        }
                        else if (c != '\r')
                        {
                            _partial.Append(c);
                        }
                    }
                }
            }
            catch (IOException)
            {
                // the game holds a write lock for a moment; try again next poll
            }
            catch (ObjectDisposedException)
            {
                Close();
            }
            return lines;
        }

        public void Close()
        {
            if (_stream != null)
            {
                try { _stream.Dispose(); } catch (Exception) { }
                _stream = null;
            }
            _path = null;
            _offset = -1;
        }

        public void Dispose()
        {
            Close();
        }
    }
}



