using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace TavernPlugin
{
    public static class GameLog
    {
        private static readonly object _lock = new object();

        public static string FilePath
        {
            get
            {
                try
                {
                    return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "tavernplugin.log");
                }
                catch (Exception)
                {
                    return "tavernplugin.log";
                }
            }
        }

        public static void Write(string message)
        {
            try
            {
                lock (_lock)
                {
                    using (var stream = new FileStream(FilePath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
                    using (var writer = new StreamWriter(stream, new UTF8Encoding(false)))
                    {
                        writer.WriteLine(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture)
                                         + " " + message);
                    }
                }
            }
            catch (Exception)
            {
            }
        }
    }
}
