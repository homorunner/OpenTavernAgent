// Ordinal string helpers written against the one string API that always works
// (the indexer and Length).
//
// Hearthstone's mscorlib is missing members the game itself never calls (e.g.
// File.AppendAllText(string, string, Encoding) and Convert.ToString(object) both
// throw MissingMethodException at runtime. String overloads that take a
// StringComparison are exactly the kind of thing a linker would drop, so the
// parser does not depend on them.

using System.Text;

namespace TavernPlugin
{
    public static class Text
    {
        public static bool StartsWith(string s, string prefix)
        {
            if (s == null || prefix == null || s.Length < prefix.Length) return false;
            for (int i = 0; i < prefix.Length; i++)
                if (s[i] != prefix[i]) return false;
            return true;
        }

        public static bool EndsWith(string s, string suffix)
        {
            if (s == null || suffix == null || s.Length < suffix.Length) return false;
            int offset = s.Length - suffix.Length;
            for (int i = 0; i < suffix.Length; i++)
                if (s[offset + i] != suffix[i]) return false;
            return true;
        }

        public static int IndexOf(string s, string needle, int from)
        {
            if (s == null || needle == null) return -1;
            if (from < 0) from = 0;
            int n = needle.Length;
            if (n == 0) return from <= s.Length ? from : -1;
            for (int i = from; i + n <= s.Length; i++)
            {
                int k = 0;
                while (k < n && s[i + k] == needle[k]) k++;
                if (k == n) return i;
            }
            return -1;
        }

        public static int IndexOf(string s, string needle)
        {
            return IndexOf(s, needle, 0);
        }

        // Ordinal comparison without string.CompareOrdinal (which may be stripped).
        public static int CompareOrdinal(string a, string b)
        {
            if (a == null) return b == null ? 0 : -1;
            if (b == null) return 1;
            int n = a.Length < b.Length ? a.Length : b.Length;
            for (int i = 0; i < n; i++)
            {
                if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
            }
            return a.Length.CompareTo(b.Length);
        }

        // ASCII-only case folding; the strings compared are config keys and literals.
        public static bool EqualsIgnoreCase(string a, string b)
        {
            if (a == null || b == null) return a == b;
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
                if (Lower(a[i]) != Lower(b[i])) return false;
            return true;
        }

        public static bool ContainsIgnoreCase(string s, string needle)
        {
            return s != null && needle != null && IndexOfIgnoreCase(s, needle) >= 0;
        }

        private static int IndexOfIgnoreCase(string s, string needle)
        {
            int n = needle.Length;
            if (n == 0) return 0;
            for (int i = 0; i + n <= s.Length; i++)
            {
                int k = 0;
                while (k < n && Lower(s[i + k]) == Lower(needle[k])) k++;
                if (k == n) return i;
            }
            return -1;
        }

        private static char Lower(char c)
        {
            return c >= 'A' && c <= 'Z' ? (char)(c + 32) : c;
        }

            public static string DecodeUtf8(byte[] buffer, int count)
        {
            if (buffer == null || count <= 0) return "";
            if (count >= buffer.Length) return Encoding.UTF8.GetString(buffer);
            var exact = new byte[count];
            for (int i = 0; i < count; i++) exact[i] = buffer[i];
            return Encoding.UTF8.GetString(exact);
        }
    }
}

