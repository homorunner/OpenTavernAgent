// Minimal JSON reader/writer. TavernPlugin must build both inside Unity's Mono
// (.NET Framework 4.7.2, C# 7.3) and in the offline replay tool (net9.0), and it
// must not add NuGet dependencies to the game plugin, so this is hand-rolled.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace TavernPlugin
{
    // Writes JSON without any dependency. Keys keep insertion order.
    public sealed class JsonWriter
    {
        private readonly StringBuilder _sb = new StringBuilder(1024);
        // one flag per open container: true while it has no members yet
        private readonly Stack<bool> _first = new Stack<bool>();
        private bool _afterKey;

        public JsonWriter BeginObject()
        {
            BeforeValue();
            _sb.Append('{');
            _first.Push(true);
            return this;
        }

        public JsonWriter EndObject()
        {
            _sb.Append('}');
            _first.Pop();
            return this;
        }

        public JsonWriter BeginArray()
        {
            BeforeValue();
            _sb.Append('[');
            _first.Push(true);
            return this;
        }

        public JsonWriter EndArray()
        {
            _sb.Append(']');
            _first.Pop();
            return this;
        }

        public JsonWriter Key(string name)
        {
            BeforeValue();
            WriteString(name);
            _sb.Append(':');
            _afterKey = true;
            return this;
        }

        private void BeforeValue()
        {
            if (_afterKey)
            {
                _afterKey = false; // a value right after a key needs no separator
                return;
            }
            if (_first.Count == 0) return;
            if (!_first.Peek()) _sb.Append(',');
            _first.Pop();
            _first.Push(false);
        }

        public JsonWriter Value(string s)
        {
            BeforeValue();
            if (s == null) _sb.Append("null");
            else WriteString(s);
            return this;
        }

        public JsonWriter Value(double d)
        {
            BeforeValue();
            if (double.IsNaN(d) || double.IsInfinity(d)) _sb.Append("null");
            else _sb.Append(d.ToString("R", CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter Value(long i)
        {
            BeforeValue();
            _sb.Append(i.ToString(CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter Value(int i)
        {
            return Value((long)i);
        }

        public JsonWriter Value(bool b)
        {
            BeforeValue();
            _sb.Append(b ? "true" : "false");
            return this;
        }

        public JsonWriter Null()
        {
            return Value((string)null);
        }

            public JsonWriter Field(string key, string value)
        {
            return Key(key).Value(value);
        }

        public JsonWriter Field(string key, double value)
        {
            return Key(key).Value(value);
        }

        public JsonWriter Field(string key, int value)
        {
            return Key(key).Value((long)value);
        }

        public JsonWriter Field(string key, bool value)
        {
            return Key(key).Value(value);
        }

        private void WriteString(string s)
        {
            _sb.Append('"');
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                switch (c)
                {
                    case '"': _sb.Append("\\\""); break;
                    case '\\': _sb.Append("\\\\"); break;
                    case '\b': _sb.Append("\\b"); break;
                    case '\f': _sb.Append("\\f"); break;
                    case '\n': _sb.Append("\\n"); break;
                    case '\r': _sb.Append("\\r"); break;
                    case '\t': _sb.Append("\\t"); break;
                    default:
                        if (c < ' ') _sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else _sb.Append(c);
                        break;
                }
            }
            _sb.Append('"');
        }

        public override string ToString()
        {
            return _sb.ToString();
        }
    }

    // Tiny JSON reader: objects become Dictionary<string, object>, arrays List<object>,
    // numbers double, plus string/bool/null. Enough for the bridge's replies; not a
    // general-purpose parser.
    public static class JsonLite
    {
        public static object Parse(string text)
        {
            int i = 0;
            object value = ParseValue(text, ref i);
            return value;
        }

        public static Dictionary<string, object> ParseObject(string text)
        {
            return Parse(text) as Dictionary<string, object>;
        }

        private static void SkipWs(string s, ref int i)
        {
            while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
        }

        private static object ParseValue(string s, ref int i)
        {
            SkipWs(s, ref i);
            if (i >= s.Length) throw new FormatException("unexpected end of JSON");
            char c = s[i];
            if (c == '{') return ParseObjectBody(s, ref i);
            if (c == '[') return ParseArrayBody(s, ref i);
            if (c == '"') return ParseString(s, ref i);
            if (c == 't') { Expect(s, ref i, "true"); return true; }
            if (c == 'f') { Expect(s, ref i, "false"); return false; }
            if (c == 'n') { Expect(s, ref i, "null"); return null; }
            return ParseNumber(s, ref i);
        }

        private static void Expect(string s, ref int i, string word)
        {
            if (string.CompareOrdinal(s, i, word, 0, word.Length) != 0)
                throw new FormatException("bad literal at " + i);
            i += word.Length;
        }

        private static Dictionary<string, object> ParseObjectBody(string s, ref int i)
        {
            var map = new Dictionary<string, object>(StringComparer.Ordinal);
            i++; // {
            SkipWs(s, ref i);
            if (i < s.Length && s[i] == '}') { i++; return map; }
            while (true)
            {
                SkipWs(s, ref i);
                string key = ParseString(s, ref i);
                SkipWs(s, ref i);
                if (i >= s.Length || s[i] != ':') throw new FormatException("expected ':' at " + i);
                i++;
                map[key] = ParseValue(s, ref i);
                SkipWs(s, ref i);
                if (i >= s.Length) throw new FormatException("unterminated object");
                if (s[i] == ',') { i++; continue; }
                if (s[i] == '}') { i++; return map; }
                throw new FormatException("expected ',' or '}' at " + i);
            }
        }

        private static List<object> ParseArrayBody(string s, ref int i)
        {
            var list = new List<object>();
            i++; // [
            SkipWs(s, ref i);
            if (i < s.Length && s[i] == ']') { i++; return list; }
            while (true)
            {
                list.Add(ParseValue(s, ref i));
                SkipWs(s, ref i);
                if (i >= s.Length) throw new FormatException("unterminated array");
                if (s[i] == ',') { i++; continue; }
                if (s[i] == ']') { i++; return list; }
                throw new FormatException("expected ',' or ']' at " + i);
            }
        }

        private static string ParseString(string s, ref int i)
        {
            if (s[i] != '"') throw new FormatException("expected string at " + i);
            i++;
            var sb = new StringBuilder();
            while (i < s.Length)
            {
                char c = s[i++];
                if (c == '"') return sb.ToString();
                if (c != '\\') { sb.Append(c); continue; }
                if (i >= s.Length) break;
                char e = s[i++];
                switch (e)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u':
                        if (i + 4 <= s.Length)
                        {
                            sb.Append((char)Hex4(s, i));
                            i += 4;
                        }
                        break;
                    default: sb.Append(e); break;
                }
            }
            throw new FormatException("unterminated string");
        }

        private static object ParseNumber(string s, ref int i)
        {
            int start = i;
            while (i < s.Length && ("+-.0123456789eE".IndexOf(s[i]) >= 0)) i++;
            // invariant culture: JSON always uses '.', unlike the host locale
            return double.Parse(s.Substring(start, i - start), NumberStyles.Float, CultureInfo.InvariantCulture);
        }

        private static int Hex4(string s, int at)
        {
            int value = 0;
            for (int k = 0; k < 4; k++)
            {
                char c = s[at + k];
                int digit;
                if (c >= '0' && c <= '9') digit = c - '0';
                else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
                else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
                else return 0;
                value = (value << 4) | digit;
            }
            return value;
        }

        // ---- helpers for reading replies -------------------------------------

        // Convert.ToString(object) is stripped from Hearthstone's mscorlib, so values
        // are converted by type. Objects and arrays have no scalar form here.
        public static double Num(Dictionary<string, object> o, string key, double fallback = 0.0)
        {
            object v;
            if (o == null || !o.TryGetValue(key, out v) || v == null) return fallback;
            if (v is double) return (double)v;
            if (v is bool) return ((bool)v) ? 1.0 : 0.0;
            string text = v as string;
            if (text == null) return fallback;
            double d;
            return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out d) ? d : fallback;
        }

        public static string Str(Dictionary<string, object> o, string key, string fallback = "")
        {
            object v;
            if (o == null || !o.TryGetValue(key, out v) || v == null) return fallback;
            string text = v as string;
            if (text != null) return text;
            if (v is double) return ((double)v).ToString("R", CultureInfo.InvariantCulture);
            if (v is bool) return ((bool)v) ? "true" : "false";
            return fallback;
        }

        public static List<object> Arr(Dictionary<string, object> o, string key)
        {
            object v;
            if (o != null && o.TryGetValue(key, out v)) return v as List<object>;
            return null;
        }

        public static bool Bool(Dictionary<string, object> o, string key, bool fallback = false)
        {
            object v;
            if (o == null || !o.TryGetValue(key, out v) || v == null) return fallback;
            if (v is bool) return (bool)v;
            if (v is double) return (double)v != 0.0;
            string text = v as string;
            return text != null && Text.EqualsIgnoreCase(text, "true");
        }
    }
}


