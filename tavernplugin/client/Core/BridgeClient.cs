// Talks to the bridge server (tavernplugin/server/server.py).
//
// Two hard constraints come from Hearthstone's runtime:
//
//  1. System.Net.Http is unusable (HttpClient.set_Timeout throws
//     MissingMethodException), so this uses HttpWebRequest.
//  2. The game's mscorlib is stripped of members it never calls, and a method body
//     that mentions a missing member fails *at JIT time* — the whole method throws
//     before its first statement runs, so a try/catch around that one statement
//     does not help.
//
// Hence the granularity is the *method*, not the statement: POST lives in its own
// method, and if anything in it is unavailable the request is retried as a GET with
// the JSON in a `body` query parameter, which the server accepts. Only two things
// are caught on purpose — that fallback, and a transport failure that has to become
// an error reply instead of an exception.
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text;

namespace TavernPlugin
{
    public sealed class BridgeReply
    {
        public bool Ok;
        public string Error = "";
        public int Status;
        public int ElapsedMs;
        public Dictionary<string, object> Body;

        public string Str(string key, string fallback = "")
        {
            return JsonLite.Str(Body, key, fallback);
        }

        public double Num(string key, double fallback = 0.0)
        {
            return JsonLite.Num(Body, key, fallback);
        }

        public override string ToString()
        {
            return Ok ? "ok" : Error;
        }
    }

    public sealed class BridgeClient : IDisposable
    {
        private readonly string _base;
        private readonly int _timeoutMs;

        public BridgeClient(string host = "127.0.0.1", int port = 8765, int timeoutMs = 30000)
        {
            _base = "http://" + host + ":" + port;
            _timeoutMs = timeoutMs;
        }

        public string BaseUrl { get { return _base; } }

        public void Dispose()
        {
        }

        public BridgeReply Get(string route)
        {
            return Send(route, null);
        }

        public BridgeReply Post(string route, string json)
        {
            return Send(route, json);
        }

        private BridgeReply Send(string route, string json)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            BridgeReply reply;
            try
            {
                reply = json == null ? GetOnce(route) : PostOrFallback(route, json);
            }
            catch (Exception exc)
            {
                // server not listening, DNS, timeout: report it, do not throw into the
                // game loop
                reply = new BridgeReply { Ok = false, Error = exc.GetType().Name + ": " + exc.Message };
            }
            watch.Stop();
            reply.ElapsedMs = (int)watch.ElapsedMilliseconds;
            return reply;
        }

        // POST, or the same payload as a GET query when POST is unusable here.
        private BridgeReply PostOrFallback(string route, string json)
        {
            try
            {
                return PostOnce(route, json);
            }
            catch (Exception postFailure)
            {
                GameLog.Write("[bridge] POST failed (" + postFailure.GetType().Name +
                              "), falling back to GET: " + postFailure.Message);
                return GetOnce(route + "?body=" + UrlEncode(json));
            }
        }
        private BridgeReply PostOnce(string route, string json)
        {
            HttpWebRequest request = Create(_base + route);
            request.Method = "POST";
            request.Timeout = _timeoutMs;
            request.KeepAlive = false;
            byte[] body = Encoding.UTF8.GetBytes(json);
            request.ContentType = "application/json; charset=utf-8";
            request.ContentLength = body.Length;
            using (Stream stream = request.GetRequestStream())
            {
                stream.Write(body, 0, body.Length);
            }
            return Read(request);
        }

        private BridgeReply GetOnce(string route)
        {
            HttpWebRequest request = Create(_base + route);
            request.Method = "GET";
            request.Timeout = _timeoutMs;
            request.KeepAlive = false;
            return Read(request);
        }

        private static HttpWebRequest Create(string url)
        {
#pragma warning disable SYSLIB0014  // obsolete on .NET 6+, required on Unity's Mono
            return (HttpWebRequest)WebRequest.Create(url);
#pragma warning restore SYSLIB0014
        }

        private BridgeReply Read(HttpWebRequest request)
        {
            var reply = new BridgeReply();
            using (var response = (HttpWebResponse)request.GetResponse())
            {
                reply.Status = (int)response.StatusCode;
                reply.Body = ReadJson(response);
            }
            reply.Ok = JsonLite.Bool(reply.Body, "ok", false);
            if (!reply.Ok) reply.Error = JsonLite.Str(reply.Body, "error", "HTTP " + reply.Status);
            return reply;
        }

        private static Dictionary<string, object> ReadJson(HttpWebResponse response)
        {
            try
            {
                using (Stream stream = response.GetResponseStream())
                {
                    if (stream == null) return null;
                    using (var reader = new StreamReader(stream, Encoding.UTF8))
                    {
                        string text = reader.ReadToEnd();
                        return string.IsNullOrEmpty(text) ? null : JsonLite.ParseObject(text);
                    }
                }
            }
            catch (Exception)
            {
                return null;
            }
        }
        // Percent-encode for the GET fallback (Uri.EscapeDataString may be stripped).
        public static string UrlEncode(string text)
        {
            const string hex = "0123456789ABCDEF";
            var sb = new StringBuilder(text.Length * 3);
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            for (int i = 0; i < bytes.Length; i++)
            {
                byte b = bytes[i];
                bool safe = (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
                            || b == '-' || b == '_' || b == '.' || b == '~';
                if (safe) sb.Append((char)b);
                else
                {
                    sb.Append('%');
                    sb.Append(hex[b >> 4]);
                    sb.Append(hex[b & 0xF]);
                }
            }
            return sb.ToString();
        }

            public bool Ping(out string info)
        {
            BridgeReply reply = Get("/health");
            if (!reply.Ok)
            {
                info = reply.Error;
                return false;
            }
            bool core = JsonLite.Bool(reply.Body, "taverngym", false);
            info = core ? "taverngym ready" : ("server up, " + JsonLite.Str(reply.Body, "error", "taverngym missing"));
            return true;
        }
    }
}


