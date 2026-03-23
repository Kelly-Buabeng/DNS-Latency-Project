"""
DCIT 417 — Network Modelling & Performance Analysis
network_detect.py  —  Auto Network Detection Module

Automatically detects:
    • Current SSID  (Wi-Fi network name)
    • DNS resolver IPs  (from OS resolver config)
    • Default gateway IP

Works on Linux, macOS, and Windows without any extra dependencies.
Falls back gracefully through multiple detection methods.

Usage (imported by dns_measure.py automatically):
    from network_detect import detect_network
    info = detect_network()
    print(info)
    # → NetworkInfo(ssid='UG-WiFi', dns_servers=['10.1.1.1','8.8.8.8'], gateway='10.1.0.1')
"""

import platform
import re
import socket
import struct
import subprocess
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────

@dataclass
class NetworkInfo:
    ssid:           Optional[str]       # Wi-Fi network name, e.g. "UG-WiFi"
    dns_servers:    list[str]           # ordered list of resolver IPs
    gateway:        Optional[str]       # default gateway IP
    interface:      Optional[str]       # active network interface name
    detection_log:  list[str] = field(default_factory=list, repr=False)

    def resolver_label(self) -> str:
        """
        Return a clean, filesystem-safe label derived from the SSID.
        Used as the resolver name in CSV output instead of a hardcoded label.

        Examples:
            'UG-WiFi'      → 'ug_wifi'
            'MTN Home 5G'  → 'mtn_home_5g'
            'eduroam'      → 'eduroam'
            None           → 'unknown_network'
        """
        if not self.ssid:
            return "unknown_network"
        label = self.ssid.lower()
        label = re.sub(r'[^a-z0-9]+', '_', label)   # replace non-alphanumeric with _
        label = label.strip('_')
        return label or "unknown_network"

    def primary_dns(self) -> Optional[str]:
        """Return the first (primary) DNS server IP, or None."""
        return self.dns_servers[0] if self.dns_servers else None

    def __str__(self) -> str:
        lines = [
            f"  SSID          : {self.ssid or '(not detected)'}",
            f"  Resolver label: {self.resolver_label()}",
            f"  DNS servers   : {self.dns_servers or '(none detected)'}",
            f"  Gateway       : {self.gateway or '(not detected)'}",
            f"  Interface     : {self.interface or '(not detected)'}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 3) -> Optional[str]:
    """Run a shell command and return stdout, or None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _log(info: NetworkInfo, msg: str) -> None:
    info.detection_log.append(msg)


# ─────────────────────────────────────────────────────────────────
# SSID DETECTION
# ─────────────────────────────────────────────────────────────────

def _get_ssid_linux(info: NetworkInfo) -> Optional[str]:
    """Try multiple Linux methods to get the connected SSID."""

    # Method 1: nmcli (NetworkManager — most common on Ubuntu/Fedora)
    out = _run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"])
    if out:
        for line in out.splitlines():
            if line.startswith("yes:"):
                ssid = line.split(":", 1)[1]
                _log(info, f"SSID via nmcli: {ssid}")
                return ssid

    # Method 2: iwgetid (wireless-tools)
    out = _run(["iwgetid", "-r"])
    if out:
        _log(info, f"SSID via iwgetid: {out}")
        return out

    # Method 3: iwconfig (older wireless-tools)
    out = _run(["iwconfig"])
    if out:
        m = re.search(r'ESSID:"([^"]+)"', out)
        if m:
            _log(info, f"SSID via iwconfig: {m.group(1)}")
            return m.group(1)

    # Method 4: iw dev (modern iproute2)
    out = _run(["iw", "dev"])
    if out:
        m = re.search(r'ssid (.+)', out)
        if m:
            _log(info, f"SSID via iw dev: {m.group(1).strip()}")
            return m.group(1).strip()

    # Method 5: Read directly from /proc/net/wireless + iw
    out = _run(["iw", "wlan0", "link"])
    if out:
        m = re.search(r'SSID: (.+)', out)
        if m:
            return m.group(1).strip()

    _log(info, "SSID: all Linux methods failed")
    return None


def _get_ssid_macos(info: NetworkInfo) -> Optional[str]:
    """Get SSID on macOS."""
    # macOS Ventura+ moved airport tool; try both paths
    for airport_path in [
        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
        "/usr/sbin/networksetup",
    ]:
        out = _run([airport_path, "-I"])
        if out:
            m = re.search(r'\s+SSID: (.+)', out)
            if m:
                _log(info, f"SSID via airport: {m.group(1).strip()}")
                return m.group(1).strip()

    # networksetup fallback
    out = _run(["networksetup", "-getairportnetwork", "en0"])
    if out and "Current Wi-Fi Network:" in out:
        ssid = out.split("Current Wi-Fi Network:")[1].strip()
        _log(info, f"SSID via networksetup: {ssid}")
        return ssid

    _log(info, "SSID: all macOS methods failed")
    return None


def _get_ssid_windows(info: NetworkInfo) -> Optional[str]:
    """Get SSID on Windows."""
    out = _run(["netsh", "wlan", "show", "interfaces"])
    if out:
        m = re.search(r'SSID\s*:\s*(.+)', out)
        if m:
            ssid = m.group(1).strip()
            _log(info, f"SSID via netsh: {ssid}")
            return ssid
    _log(info, "SSID: Windows netsh method failed")
    return None


def get_ssid(info: NetworkInfo) -> Optional[str]:
    """Dispatch SSID detection to the right OS method."""
    os_name = platform.system()
    if os_name == "Linux":
        return _get_ssid_linux(info)
    elif os_name == "Darwin":
        return _get_ssid_macos(info)
    elif os_name == "Windows":
        return _get_ssid_windows(info)
    return None


# ─────────────────────────────────────────────────────────────────
# DNS SERVER DETECTION
# ─────────────────────────────────────────────────────────────────

def _parse_ips(text: str) -> list[str]:
    """Extract all valid IPv4 addresses from a block of text."""
    candidates = re.findall(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', text)
    valid = []
    for ip in candidates:
        parts = ip.split('.')
        if all(0 <= int(p) <= 255 for p in parts):
            # Exclude obviously wrong things like 0.0.0.0 and 255.255.255.255
            if ip not in ('0.0.0.0', '255.255.255.255'):
                valid.append(ip)
    return valid


def _get_dns_linux(info: NetworkInfo) -> list[str]:
    """Detect DNS servers on Linux."""
    servers = []

    # Method 1: resolvectl (systemd-resolved)
    out = _run(["resolvectl", "status"])
    if out:
        for line in out.splitlines():
            if "DNS Servers" in line or "DNS Server" in line:
                ips = _parse_ips(line)
                servers.extend(ips)
        if servers:
            _log(info, f"DNS via resolvectl: {servers}")
            return list(dict.fromkeys(servers))   # deduplicate, preserve order

    # Method 2: /etc/resolv.conf (universal fallback)
    try:
        with open("/etc/resolv.conf") as f:
            content = f.read()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("nameserver"):
                ips = _parse_ips(line)
                servers.extend(ips)
        if servers:
            _log(info, f"DNS via /etc/resolv.conf: {servers}")
            return list(dict.fromkeys(servers))
    except FileNotFoundError:
        pass

    # Method 3: systemd-resolve --status
    out = _run(["systemd-resolve", "--status"])
    if out:
        for line in out.splitlines():
            if "DNS Servers" in line:
                ips = _parse_ips(line)
                servers.extend(ips)
        if servers:
            _log(info, f"DNS via systemd-resolve: {servers}")
            return list(dict.fromkeys(servers))

    # Method 4: nmcli
    out = _run(["nmcli", "dev", "show"])
    if out:
        for line in out.splitlines():
            if "IP4.DNS" in line:
                ips = _parse_ips(line)
                servers.extend(ips)
        if servers:
            _log(info, f"DNS via nmcli dev show: {servers}")
            return list(dict.fromkeys(servers))

    _log(info, "DNS: all Linux methods failed")
    return []


def _get_dns_macos(info: NetworkInfo) -> list[str]:
    """Detect DNS servers on macOS."""
    servers = []

    # scutil --dns shows full resolver config
    out = _run(["scutil", "--dns"])
    if out:
        for line in out.splitlines():
            if "nameserver[" in line:
                ips = _parse_ips(line)
                servers.extend(ips)
        if servers:
            _log(info, f"DNS via scutil: {servers}")
            return list(dict.fromkeys(servers))

    # Fallback: /etc/resolv.conf (macOS also has this)
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                if line.startswith("nameserver"):
                    servers.extend(_parse_ips(line))
        if servers:
            return list(dict.fromkeys(servers))
    except FileNotFoundError:
        pass

    return []


def _get_dns_windows(info: NetworkInfo) -> list[str]:
    """Detect DNS servers on Windows."""
    servers = []

    out = _run(["ipconfig", "/all"])
    if out:
        for line in out.splitlines():
            if "DNS Servers" in line or "DNS Server" in line:
                ips = _parse_ips(line)
                servers.extend(ips)
    if servers:
        _log(info, f"DNS via ipconfig /all: {servers}")
        return list(dict.fromkeys(servers))

    # PowerShell fallback
    ps = _run(["powershell", "-Command",
               "Get-DnsClientServerAddress -AddressFamily IPv4 | Select-Object -ExpandProperty ServerAddresses"])
    if ps:
        servers = _parse_ips(ps)
        if servers:
            _log(info, f"DNS via PowerShell: {servers}")
            return list(dict.fromkeys(servers))

    return []


def get_dns_servers(info: NetworkInfo) -> list[str]:
    """Dispatch DNS detection to the right OS method."""
    os_name = platform.system()
    if os_name == "Linux":
        return _get_dns_linux(info)
    elif os_name == "Darwin":
        return _get_dns_macos(info)
    elif os_name == "Windows":
        return _get_dns_windows(info)
    return []


# ─────────────────────────────────────────────────────────────────
# GATEWAY DETECTION
# ─────────────────────────────────────────────────────────────────

def _get_gateway_linux(info: NetworkInfo) -> tuple[Optional[str], Optional[str]]:
    """Return (gateway_ip, interface) on Linux."""

    # Method 1: ip route (iproute2)
    out = _run(["ip", "route", "show", "default"])
    if out:
        m = re.search(r'default via (\S+) dev (\S+)', out)
        if m:
            _log(info, f"Gateway via ip route: {m.group(1)} on {m.group(2)}")
            return m.group(1), m.group(2)

    # Method 2: /proc/net/route (always available on Linux)
    try:
        with open("/proc/net/route") as f:
            lines = f.readlines()
        for line in lines[1:]:           # skip header
            fields = line.strip().split()
            if len(fields) >= 3 and fields[1] == "00000000":   # destination = 0.0.0.0
                # Gateway is hex, little-endian
                gw_hex = fields[2]
                gw_int = int(gw_hex, 16)
                gw_ip = socket.inet_ntoa(struct.pack("<I", gw_int))
                iface = fields[0]
                if gw_ip != "0.0.0.0":
                    _log(info, f"Gateway via /proc/net/route: {gw_ip} on {iface}")
                    return gw_ip, iface
    except Exception:
        pass

    # Method 3: route -n
    out = _run(["route", "-n"])
    if out:
        for line in out.splitlines():
            if line.startswith("0.0.0.0"):
                parts = line.split()
                if len(parts) >= 2:
                    _log(info, f"Gateway via route -n: {parts[1]}")
                    return parts[1], parts[7] if len(parts) > 7 else None

    _log(info, "Gateway: all Linux methods failed")
    return None, None


def _get_gateway_macos(info: NetworkInfo) -> tuple[Optional[str], Optional[str]]:
    """Return (gateway_ip, interface) on macOS."""
    out = _run(["netstat", "-rn"])
    if out:
        for line in out.splitlines():
            if line.startswith("default"):
                parts = line.split()
                if len(parts) >= 2:
                    gw = parts[1]
                    iface = parts[-1] if len(parts) > 5 else None
                    _log(info, f"Gateway via netstat: {gw}")
                    return gw, iface
    return None, None


def _get_gateway_windows(info: NetworkInfo) -> tuple[Optional[str], Optional[str]]:
    """Return (gateway_ip, interface) on Windows."""
    out = _run(["ipconfig"])
    if out:
        m = re.search(r'Default Gateway[^\d]*(\d+\.\d+\.\d+\.\d+)', out)
        if m:
            _log(info, f"Gateway via ipconfig: {m.group(1)}")
            return m.group(1), None
    return None, None


def get_gateway(info: NetworkInfo) -> tuple[Optional[str], Optional[str]]:
    """Dispatch gateway detection to the right OS method."""
    os_name = platform.system()
    if os_name == "Linux":
        return _get_gateway_linux(info)
    elif os_name == "Darwin":
        return _get_gateway_macos(info)
    elif os_name == "Windows":
        return _get_gateway_windows(info)
    return None, None


# ─────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────

def detect_network(verbose: bool = False) -> NetworkInfo:
    """
    Auto-detect the current network environment.

    Returns a NetworkInfo object containing:
        • ssid           : the Wi-Fi network name (e.g. "UG-WiFi")
        • dns_servers    : list of resolver IPs from the OS
        • gateway        : default gateway IP
        • interface      : active network interface name
        • resolver_label : filesystem-safe name derived from SSID

    Parameters
    ----------
    verbose : if True, print a detection log after auto-detect completes

    Example
    -------
        from network_detect import detect_network
        info = detect_network(verbose=True)

        # Build resolver config for dns_measure.py
        resolvers = {
            info.resolver_label(): info.primary_dns(),        # e.g. "ug_wifi": "10.1.1.1"
            "google_public":       "8.8.8.8",
            "cloudflare_public":   "1.1.1.1",
        }
    """
    info = NetworkInfo(ssid=None, dns_servers=[], gateway=None, interface=None)

    info.ssid = get_ssid(info)

    dns = get_dns_servers(info)
    info.dns_servers = dns

    gw, iface = get_gateway(info)
    info.gateway = gw
    info.interface = iface

    if verbose:
        print("\n── Network auto-detection ──")
        print(info)
        if info.detection_log:
            print("  Detection log:")
            for entry in info.detection_log:
                print(f"    • {entry}")

    return info


def build_resolver_config(info: NetworkInfo,
                           always_include_public: bool = True) -> dict[str, str]:
    """
    Build the RESOLVERS dict for dns_measure.py from detected network info.

    The current network's DNS server gets the SSID-derived label.
    Public resolvers (Google, Cloudflare) are always appended unless disabled.

    Parameters
    ----------
    info                  : NetworkInfo from detect_network()
    always_include_public : if True, Google 8.8.8.8 and Cloudflare 1.1.1.1
                            are added as comparison baselines

    Returns
    -------
    dict  e.g. {"ug_wifi": "10.1.1.1",
                "google_public": "8.8.8.8",
                "cloudflare_public": "1.1.1.1"}
    """
    resolvers: dict[str, str] = {}

    if info.primary_dns():
        label = info.resolver_label()
        resolvers[label] = info.primary_dns()

        # If there's a secondary DNS, include it too (label it _secondary)
        if len(info.dns_servers) > 1:
            resolvers[f"{label}_secondary"] = info.dns_servers[1]

    if always_include_public:
        resolvers["google_public"]     = "8.8.8.8"
        resolvers["cloudflare_public"] = "1.1.1.1"

    return resolvers


# ─────────────────────────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DCIT 417 — Network Auto-Detection")
    print("=" * 40)

    info = detect_network(verbose=True)

    print("\n── Resolver config for dns_measure.py ──")
    resolvers = build_resolver_config(info)
    if resolvers:
        for name, ip in resolvers.items():
            print(f"  '{name}': '{ip}'")
    else:
        print("  WARNING: No DNS servers detected.")
        print("  Check your network connection and try again.")
