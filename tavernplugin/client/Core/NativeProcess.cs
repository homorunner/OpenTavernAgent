using System;
using System.Runtime.InteropServices;
using System.Text;

namespace TavernPlugin
{
    public static class NativeProcess
    {
        private const uint CREATE_NO_WINDOW = 0x08000000;
        private const uint CREATE_UNICODE_ENVIRONMENT = 0x00000400;
        private const uint PROCESS_TERMINATE = 0x0001;

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct STARTUPINFO
        {
            public int cb;
            public string lpReserved;
            public string lpDesktop;
            public string lpTitle;
            public int dwX;
            public int dwY;
            public int dwXSize;
            public int dwYSize;
            public int dwXCountChars;
            public int dwYCountChars;
            public int dwFillAttribute;
            public int dwFlags;
            public short wShowWindow;
            public short cbReserved2;
            public IntPtr lpReserved2;
            public IntPtr hStdInput;
            public IntPtr hStdOutput;
            public IntPtr hStdError;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct PROCESS_INFORMATION
        {
            public IntPtr hProcess;
            public IntPtr hThread;
            public int dwProcessId;
            public int dwThreadId;
        }

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern bool CreateProcessW(
            string lpApplicationName,
            StringBuilder lpCommandLine,
            IntPtr lpProcessAttributes,
            IntPtr lpThreadAttributes,
            bool bInheritHandles,
            uint dwCreationFlags,
            IntPtr lpEnvironment,
            string lpCurrentDirectory,
            ref STARTUPINFO lpStartupInfo,
            out PROCESS_INFORMATION lpProcessInformation);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr handle);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr OpenProcess(uint desiredAccess, bool inheritHandle, int processId);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool TerminateProcess(IntPtr process, uint exitCode);

        public static int Start(string exe, string arguments, string workingDirectory, out string error)
        {
            error = null;
            var si = new STARTUPINFO();
            si.cb = Marshal.SizeOf(typeof(STARTUPINFO));
            PROCESS_INFORMATION pi;

            var commandLine = new StringBuilder();
            commandLine.Append('"').Append(exe).Append('"');
            if (!string.IsNullOrEmpty(arguments)) commandLine.Append(' ').Append(arguments);

            bool ok = CreateProcessW(
                exe,
                commandLine,
                IntPtr.Zero,
                IntPtr.Zero,
                false,
                CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
                IntPtr.Zero,
                workingDirectory,
                ref si,
                out pi);

            if (!ok)
            {
                error = "CreateProcess failed, win32 error " + Marshal.GetLastWin32Error();
                return 0;
            }

            try
            {
                CloseHandle(pi.hThread);
            }
            catch (Exception)
            {
            }
            try
            {
                CloseHandle(pi.hProcess);
            }
            catch (Exception)
            {
            }
            return pi.dwProcessId;
        }

        public static bool Kill(int pid)
        {
            if (pid <= 0) return false;
            IntPtr handle = IntPtr.Zero;
            try
            {
                handle = OpenProcess(PROCESS_TERMINATE, false, pid);
                if (handle == IntPtr.Zero) return false;
                return TerminateProcess(handle, 0);
            }
            catch (Exception)
            {
                return false;
            }
            finally
            {
                if (handle != IntPtr.Zero)
                {
                    try { CloseHandle(handle); } catch (Exception) { }
                }
            }
        }
    }
}
