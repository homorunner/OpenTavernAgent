// ilspec: print the game's types as C#, using the decompiler that ships with ILSpy.
//
//   ilspec list <assembly> <regex>            type names matching the regex
//   ilspec type <assembly> <Full.Type.Name>    decompile one type
//   ilspec members <assembly> <regex>          one-line member list per matching type
//
// The assembly is normally
//   C:\Program Files (x86)\Hearthstone\Hearthstone_Data\Managed\Assembly-CSharp.dll
// and the resolver also looks next to it, so UnityEngine references resolve.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using ICSharpCode.Decompiler;
using ICSharpCode.Decompiler.CSharp;
using ICSharpCode.Decompiler.Metadata;
using ICSharpCode.Decompiler.TypeSystem;

namespace TavernPlugin.IlSpec
{
    internal static class Program
    {
        private static int Main(string[] args)
        {
            if (args.Length < 3)
            {
                Console.Error.WriteLine("usage: ilspec <list|type|members> <assembly> <regex|type name>");
                return 2;
            }

            string mode = args[0];
            string assembly = args[1];
            string query = args[2];

            if (!File.Exists(assembly))
            {
                Console.Error.WriteLine("no such assembly: " + assembly);
                return 2;
            }

            var settings = new DecompilerSettings(LanguageVersion.CSharp10_0)
            {
                ThrowOnAssemblyResolveErrors = false,
                ShowXmlDocumentation = false,
                UsingDeclarations = true,
                RemoveDeadCode = false,
            };
            var resolver = new UniversalAssemblyResolver(assembly, false, null);
            // the game's own references (UnityEngine, Blizzard.T5.*, ...) live next to it
            resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(assembly)));
            var decompiler = new CSharpDecompiler(assembly, resolver, settings);
            var module = decompiler.TypeSystem.MainModule;

            switch (mode)
            {
                case "list":
                {
                    var rx = new Regex(query, RegexOptions.IgnoreCase);
                    foreach (ITypeDefinition type in module.TypeDefinitions)
                    {
                        string full = type.FullName.ToString();
                        if (rx.IsMatch(full)) Console.WriteLine(full);
                    }
                    return 0;
                }
                case "members":
                {
                    var rx = new Regex(query, RegexOptions.IgnoreCase);
                    foreach (ITypeDefinition type in module.TypeDefinitions)
                    {
                        string full = type.FullName.ToString();
                        if (!rx.IsMatch(full)) continue;
                        string baseName = type.DirectBaseTypes
                            .Select(b => b.FullName.ToString())
                            .FirstOrDefault();
                        Console.WriteLine("== " + full + (baseName != null ? " : " + baseName : ""));
                        foreach (IEntity member in type.Members.OrderBy(m => m.Name, StringComparer.Ordinal))
                        {
                            string kind = member is IMethod m && m.IsConstructor ? "ctor"
                                : member is IMethod ? "method"
                                : member is IProperty ? "prop"
                                : member is IField ? "field"
                                : member is IEvent ? "event"
                                : "member";
                            string sig = member is IMethod mm
                                ? "(" + string.Join(", ", mm.Parameters.Select(p => p.Type.Name + " " + p.Name)) + ")"
                                : (member as IMember) != null ? ((IMember)member).ReturnType.Name : "";
                            Console.WriteLine($"   {kind,-6} {member.Name}{sig}");
                        }
                        Console.WriteLine();
                    }
                    return 0;
                }
                case "type":
                {
                    var name = new FullTypeName(query);
                    string code = decompiler.DecompileTypeAsString(name);
                    Console.Out.Write(code);
                    return 0;
                }
                default:
                    Console.Error.WriteLine("unknown mode " + mode);
                    return 2;
            }
        }
    }
}
