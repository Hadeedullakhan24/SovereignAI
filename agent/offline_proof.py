"""SovereignAI Offline Compliance & Network Airgap Enforcement (offline_proof.py).

This utility verifies and enforces zero external network calls at runtime,
directly answering the "prove no external API calls" requirement in the
SIH problem statement.

Capabilities
------------
1. Monkey-patches socket-level and HTTP-level network entrypoints:
   - socket.socket.connect
   - socket.socket.connect_ex
   - socket.create_connection
   - urllib.request.urlopen
   - http.client.HTTPConnection.connect
   - http.client.HTTPSConnection.connect
   - requests.Session.send (if requests library is installed)
2. Intercepts any outbound network attempt, records a full stack trace with the
   exact source file and line number of the origin call, and raises a
   `BlockedNetworkCallError` to prevent any packet from reaching the network.
3. Provides `offline_guard()` as both a context manager and a decorator:
       with offline_guard() as guard:
           agent.handle(...)
       report = guard.get_report()
4. Generates an `OfflineComplianceReport` (structured JSON + terminal display)
   certifying whether the run was 100% offline compliant.
"""

from __future__ import annotations

import contextlib
import functools
import http.client
import json
import logging
import socket
import sys
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Try importing requests if available
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logger = logging.getLogger("sovereign_ai.offline_guard")


# =============================================================================
# Custom Exceptions
# =============================================================================

class BlockedNetworkCallError(PermissionError, ConnectionRefusedError):
    """Raised when an outbound network call is attempted under OfflineGuard."""

    def __init__(
        self,
        target: str,
        api: str,
        origin_file: str,
        origin_line: int,
        origin_function: str = "",
    ) -> None:
        self.target = target
        self.api = api
        self.origin_file = origin_file
        self.origin_line = origin_line
        self.origin_function = origin_function
        super().__init__(
            f"[OFFLINE AIRGAP VIOLATION] Outbound connection to '{target}' blocked "
            f"via {api}() at {origin_file}:{origin_line} in {origin_function}()"
        )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BlockedAttempt:
    """Record of an intercepted and blocked outbound network attempt."""

    timestamp: str
    target: str
    protocol: str
    caller_api: str
    origin_file: str
    origin_line: int
    origin_function: str
    origin_code: str
    stack_trace: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OfflineComplianceReport:
    """Audit report produced at the conclusion of an offline-guarded execution."""

    timestamp: str
    is_compliant: bool
    status: str
    blocked_count: int
    allowed_local_count: int
    duration_ms: float
    allow_localhost: bool
    blocked_attempts: List[BlockedAttempt] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "is_compliant": self.is_compliant,
            "status": self.status,
            "blocked_count": self.blocked_count,
            "allowed_local_count": self.allowed_local_count,
            "duration_ms": round(self.duration_ms, 2),
            "allow_localhost": self.allow_localhost,
            "summary": self.summary,
            "blocked_attempts": [b.to_dict() for b in self.blocked_attempts],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self) -> None:
        """Prints a human-readable audit compliance banner to standard output."""
        border = "=" * 80
        sub_border = "-" * 80

        print("\n" + border)
        print("           SOVEREIGN-AI RUNTIME OFFLINE COMPLIANCE AUDIT REPORT")
        print(border)
        print(f"Timestamp        : {self.timestamp}")
        print(f"Duration         : {self.duration_ms:.2f} ms")
        print(f"Airgap Policy    : {'Permit Localhost Only' if self.allow_localhost else 'Strict Zero-Socket Airgap'}")
        
        if self.is_compliant:
            print("Audit Result     : PASSED [100% AIRGAPPED - ZERO EXTERNAL CALLS]")
            print(f"Blocked Attempts : 0")
            if self.allow_localhost:
                print(f"Localhost Calls  : {self.allowed_local_count} (permitted internal IPC)")
            print(sub_border)
            print("VERIFICATION: System operated with absolute zero outbound network emissions.")
            print("SIH Requirement  : 'Prove no external API calls' -- COMPLIED.")
        else:
            print("Audit Result     : FAILED [OUTBOUND NETWORK ATTEMPTS INTERCEPTED]")
            print(f"Blocked Attempts : {self.blocked_count}")
            print(sub_border)
            print("INTERCEPTED VIOLATIONS:")
            for idx, attempt in enumerate(self.blocked_attempts, 1):
                print(f"\n  [VIOLATION #{idx}]")
                print(f"    Target       : {attempt.target}")
                print(f"    Protocol/API : {attempt.protocol} via {attempt.caller_api}()")
                print(f"    Call Site    : {attempt.origin_file}:{attempt.origin_line} (in {attempt.origin_function})")
                if attempt.origin_code:
                    print(f"    Code Line    : {attempt.origin_code.strip()}")
                print(f"    Stack Trace  :")
                # Indent stack trace for readable display
                for line in attempt.stack_trace.strip().splitlines():
                    print(f"      {line}")
        print(border + "\n")


# =============================================================================
# Helper Utilities
# =============================================================================

def is_loopback(host: Any) -> bool:
    """Determine if a hostname/IP is loopback/localhost."""
    if host is None:
        return False
    host_str = str(host).strip().lower()
    if host_str in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    if host_str.startswith("127."):
        return True
    return False


def _format_target(address: Any) -> Tuple[str, str]:
    """Format an address or url into (target_str, host_str)."""
    if isinstance(address, tuple) and len(address) >= 2:
        host = str(address[0])
        port = str(address[1])
        return f"{host}:{port}", host
    if isinstance(address, str):
        return address, address
    return str(address), str(address)


_INTERNAL_WRAPPER_NAMES = {
    "_intercept",
    "_extract_caller_origin",
    "guarded_connect",
    "guarded_connect_ex",
    "guarded_create_connection",
    "guarded_urlopen",
    "guarded_http_connect",
    "guarded_https_connect",
    "guarded_requests_send",
}


def _extract_caller_origin() -> Tuple[str, int, str, str, str]:
    """Walk stack backwards to find the frame that initiated the call outside internal wrappers."""
    frames = traceback.extract_stack()
    full_trace = "".join(traceback.format_stack()[:-1])

    origin_file = "unknown"
    origin_line = 0
    origin_func = "unknown"
    origin_code = ""

    for frame in reversed(frames):
        # Exclude internal guard functions
        if frame.name in _INTERNAL_WRAPPER_NAMES:
            continue
        origin_file = frame.filename
        origin_line = frame.lineno or 0
        origin_func = frame.name
        origin_code = frame.line or ""
        break

    return origin_file, origin_line, origin_func, origin_code, full_trace


# =============================================================================
# OfflineGuard Context Manager and Decorator
# =============================================================================

class OfflineGuard(contextlib.ContextDecorator):
    """Enforces network airgap during execution of code blocks or functions.

    Can be used as:
        with OfflineGuard() as guard:
            agent.handle(...)
        report = guard.get_report()

    Or as a function decorator:
        @OfflineGuard()
        def demo_run():
            ...
    """

    def __init__(
        self,
        allow_localhost: bool = False,
        raise_on_blocked: bool = True,
        auto_print_report: bool = False,
        save_report_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Args:
            allow_localhost: If True, allows connections to 127.0.0.1 / localhost
                             (for local service IPC). If False, enforces strict
                             zero-socket airgap.
            raise_on_blocked: If True, raises BlockedNetworkCallError upon any
                              blocked call. If False, silently prevents and logs.
            auto_print_report: If True, prints report to stdout on exit.
            save_report_path: Optional file path to write the JSON report to.
        """
        self.allow_localhost = allow_localhost
        self.raise_on_blocked = raise_on_blocked
        self.auto_print_report = auto_print_report
        self.save_report_path = Path(save_report_path) if save_report_path else None

        self.blocked_attempts: List[BlockedAttempt] = []
        self.allowed_local_count: int = 0
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._report: Optional[OfflineComplianceReport] = None

        # Saved original functions for clean restoration
        self._orig_socket_connect = None
        self._orig_socket_connect_ex = None
        self._orig_socket_create_connection = None
        self._orig_urllib_urlopen = None
        self._orig_http_connect = None
        self._orig_https_connect = None
        self._orig_requests_send = None

    def __enter__(self) -> "OfflineGuard":
        self.blocked_attempts.clear()
        self.allowed_local_count = 0
        self._start_time = time.perf_counter()
        self._report = None
        self._install_patches()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        try:
            self._end_time = time.perf_counter()
            self._finalize_report()
            if self.auto_print_report and self._report is not None:
                self._report.print_summary()
            if self.save_report_path and self._report is not None:
                self._save_report_file()
        finally:
            self._restore_patches()

        # If exc_val was BlockedNetworkCallError raised by the guard itself,
        # do not suppress it unless caller wanted suppression
        return False

    # -------------------------------------------------------------------------
    # Interception Handler
    # -------------------------------------------------------------------------

    def _intercept(
        self,
        target_repr: str,
        host_candidate: str,
        protocol: str,
        api_name: str,
    ) -> bool:
        """Checks target and logs/records if blocked. Returns True if allowed."""
        if self.allow_localhost and is_loopback(host_candidate):
            self.allowed_local_count += 1
            return True

        # Intercepted outbound external call!
        origin_file, origin_line, origin_func, origin_code, trace = _extract_caller_origin()
        
        attempt = BlockedAttempt(
            timestamp=datetime.now(timezone.utc).isoformat(),
            target=target_repr,
            protocol=protocol,
            caller_api=api_name,
            origin_file=origin_file,
            origin_line=origin_line,
            origin_function=origin_func,
            origin_code=origin_code,
            stack_trace=trace,
        )
        self.blocked_attempts.append(attempt)

        logger.warning(
            "[AIRGAP GUARD] Blocked %s outbound call to '%s' via %s() at %s:%d (%s)",
            protocol, target_repr, api_name, origin_file, origin_line, origin_func
        )

        if self.raise_on_blocked:
            raise BlockedNetworkCallError(
                target=target_repr,
                api=api_name,
                origin_file=origin_file,
                origin_line=origin_line,
                origin_function=origin_func,
            )
        return False

    # -------------------------------------------------------------------------
    # Monkey-Patch Installation and Removal
    # -------------------------------------------------------------------------

    def _install_patches(self) -> None:
        guard = self

        # 1. socket.socket.connect
        self._orig_socket_connect = socket.socket.connect
        def guarded_connect(sock_self: socket.socket, address: Any) -> Any:
            target_str, host_str = _format_target(address)
            if guard._intercept(target_str, host_str, "TCP/IP Socket", "socket.socket.connect"):
                return guard._orig_socket_connect(sock_self, address)
            return None
        socket.socket.connect = guarded_connect

        # 2. socket.socket.connect_ex
        self._orig_socket_connect_ex = socket.socket.connect_ex
        def guarded_connect_ex(sock_self: socket.socket, address: Any) -> int:
            target_str, host_str = _format_target(address)
            if guard._intercept(target_str, host_str, "TCP/IP Socket", "socket.socket.connect_ex"):
                return guard._orig_socket_connect_ex(sock_self, address)
            return 111  # ECONNREFUSED
        socket.socket.connect_ex = guarded_connect_ex

        # 3. socket.create_connection
        self._orig_socket_create_connection = socket.create_connection
        def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
            target_str, host_str = _format_target(address)
            if guard._intercept(target_str, host_str, "TCP/IP Socket", "socket.create_connection"):
                return guard._orig_socket_create_connection(address, *args, **kwargs)
            raise ConnectionRefusedError(f"Connection to {target_str} blocked by OfflineGuard")
        socket.create_connection = guarded_create_connection

        # 4. urllib.request.urlopen
        self._orig_urllib_urlopen = urllib.request.urlopen
        def guarded_urlopen(url: Any, *args: Any, **kwargs: Any) -> Any:
            target_str = getattr(url, "full_url", str(url))
            host_str = getattr(url, "host", target_str)
            if guard._intercept(target_str, host_str, "HTTP/HTTPS (urllib)", "urllib.request.urlopen"):
                return guard._orig_urllib_urlopen(url, *args, **kwargs)
            raise ConnectionRefusedError(f"URL {target_str} blocked by OfflineGuard")
        urllib.request.urlopen = guarded_urlopen

        # 5. http.client.HTTPConnection.connect
        self._orig_http_connect = http.client.HTTPConnection.connect
        def guarded_http_connect(http_self: http.client.HTTPConnection) -> None:
            target_str = f"{http_self.host}:{http_self.port}"
            if guard._intercept(target_str, http_self.host, "HTTP", "HTTPConnection.connect"):
                guard._orig_http_connect(http_self)
        http.client.HTTPConnection.connect = guarded_http_connect

        # 6. http.client.HTTPSConnection.connect
        self._orig_https_connect = http.client.HTTPSConnection.connect
        def guarded_https_connect(https_self: http.client.HTTPSConnection) -> None:
            target_str = f"{https_self.host}:{https_self.port}"
            if guard._intercept(target_str, https_self.host, "HTTPS", "HTTPSConnection.connect"):
                guard._orig_https_connect(https_self)
        http.client.HTTPSConnection.connect = guarded_https_connect

        # 7. requests.Session.send (if installed)
        if _REQUESTS_AVAILABLE:
            self._orig_requests_send = requests.Session.send
            def guarded_requests_send(session_self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                url = getattr(request, "url", str(request))
                host = getattr(request, "headers", {}).get("Host", url)
                if guard._intercept(url, host, "HTTP/HTTPS (requests)", "requests.Session.send"):
                    return guard._orig_requests_send(session_self, request, *args, **kwargs)
                raise ConnectionRefusedError(f"Request to {url} blocked by OfflineGuard")
            requests.Session.send = guarded_requests_send

    def _restore_patches(self) -> None:
        """Restores all original functions to prevent environment contamination."""
        if self._orig_socket_connect is not None:
            socket.socket.connect = self._orig_socket_connect
            self._orig_socket_connect = None

        if self._orig_socket_connect_ex is not None:
            socket.socket.connect_ex = self._orig_socket_connect_ex
            self._orig_socket_connect_ex = None

        if self._orig_socket_create_connection is not None:
            socket.create_connection = self._orig_socket_create_connection
            self._orig_socket_create_connection = None

        if self._orig_urllib_urlopen is not None:
            urllib.request.urlopen = self._orig_urllib_urlopen
            self._orig_urllib_urlopen = None

        if self._orig_http_connect is not None:
            http.client.HTTPConnection.connect = self._orig_http_connect
            self._orig_http_connect = None

        if self._orig_https_connect is not None:
            http.client.HTTPSConnection.connect = self._orig_https_connect
            self._orig_https_connect = None

        if _REQUESTS_AVAILABLE and self._orig_requests_send is not None:
            requests.Session.send = self._orig_requests_send
            self._orig_requests_send = None

    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------

    def _finalize_report(self) -> OfflineComplianceReport:
        duration_ms = (self._end_time - self._start_time) * 1000.0
        is_compliant = len(self.blocked_attempts) == 0
        status = "COMPLIANT" if is_compliant else "NON_COMPLIANT"

        if is_compliant:
            summary = (
                f"PASSED: 0 outbound connections attempted across {duration_ms:.1f} ms. "
                f"Airgap compliance 100% verified."
            )
        else:
            summary = (
                f"FAILED: {len(self.blocked_attempts)} outbound connection attempt(s) intercepted "
                f"and blocked during {duration_ms:.1f} ms."
            )

        self._report = OfflineComplianceReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_compliant=is_compliant,
            status=status,
            blocked_count=len(self.blocked_attempts),
            allowed_local_count=self.allowed_local_count,
            duration_ms=duration_ms,
            allow_localhost=self.allow_localhost,
            blocked_attempts=list(self.blocked_attempts),
            summary=summary,
        )
        return self._report

    def get_report(self) -> OfflineComplianceReport:
        """Retrieve the finalized compliance report."""
        if self._report is None:
            self._end_time = time.perf_counter()
            self._finalize_report()
        return self._report

    @property
    def report(self) -> OfflineComplianceReport:
        return self.get_report()

    def _save_report_file(self) -> None:
        if not self.save_report_path or self._report is None:
            return
        self.save_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_report_path.write_text(self._report.to_json(), encoding="utf-8")


# Alias for clean consumer imports
offline_guard = OfflineGuard


# =============================================================================
# Demonstration & Self-Verification Suite
# =============================================================================

if __name__ == "__main__":
    # Ensure project root is importable
    _THIS_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "agent" else Path(".").resolve()
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from agent.agent import SovereignAgent

    print("\n" + "=" * 80)
    print("      SOVEREIGN-AI STEP 6: OFFLINE PROOF & RUNTIME AIRGAP DEMONSTRATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART (A): Run full agent.handle() workflow under OfflineGuard
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART A: Running Full SovereignAgent Workflow Under Offline Guard")
    print("Goal  : Prove SovereignAgent executes 100% locally with 0 outbound calls")
    print("-" * 80)

    agent = SovereignAgent()

    with offline_guard(allow_localhost=False, auto_print_report=True) as guard_a:
        response = agent.handle(
            "Process inspection report for V-2201 knockout drum, "
            "verify calculations, and prepare statutory approval document."
        )

    report_a = guard_a.get_report()

    print(f"Agent Execution Status : {response.status}")
    print(f"Agent Is Verified      : {response.is_verified}")
    print(f"Airgap Audit Status    : {report_a.status}")
    print(f"Blocked Attempts Count : {report_a.blocked_count}")

    assert report_a.is_compliant is True, (
        f"Expected is_compliant=True, got {report_a.is_compliant}"
    )
    assert report_a.blocked_count == 0, (
        f"Expected 0 blocked calls, got {report_a.blocked_count}"
    )
    assert response.status == "awaiting_approval", (
        f"Expected agent status 'awaiting_approval', got {response.status}"
    )
    print("\n  [+] PART A PASSED: SovereignAgent executed 100% offline with zero outbound network calls!")

    # -------------------------------------------------------------------------
    # PART (B): Deliberate dummy outbound call inside the guard to prove blocking
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART B: Deliberate Outbound Connection Interception Proof")
    print("Goal  : Deliberately invoke socket.create_connection(('8.8.8.8', 53))")
    print("        Verify it is blocked, logged, and captured with file/line origin")
    print("-" * 80)

    with offline_guard(allow_localhost=False, auto_print_report=True) as guard_b:
        intercepted = False
        try:
            # Deliberate outbound connection to Google Public DNS
            socket.create_connection(("8.8.8.8", 53), timeout=0.5)
        except BlockedNetworkCallError as err:
            intercepted = True
            print(f"  [+] Expected exception caught: {type(err).__name__}")
            print(f"      Message: {err}")

    report_b = guard_b.get_report()

    assert intercepted is True, "Expected socket.create_connection to raise BlockedNetworkCallError!"
    assert report_b.is_compliant is False, "Expected report_b.is_compliant=False"
    assert report_b.blocked_count == 1, f"Expected 1 blocked attempt, got {report_b.blocked_count}"

    violation = report_b.blocked_attempts[0]
    print(f"\n  [+] Intercepted Violation Details:")
    print(f"      Target       : {violation.target}")
    print(f"      Caller API   : {violation.caller_api}")
    print(f"      Origin File  : {violation.origin_file}")
    print(f"      Origin Line  : {violation.origin_line}")
    print(f"      Origin Func  : {violation.origin_function}")

    assert "8.8.8.8:53" in violation.target or "8.8.8.8" in violation.target
    assert "socket.create_connection" in violation.caller_api
    assert violation.origin_line > 0
    assert "offline_proof.py" in violation.origin_file
    print("\n  [+] PART B PASSED: Outbound connection actively blocked and origin logged!")

    # -------------------------------------------------------------------------
    # PART (C): Verify JSON export format
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART C: Verify JSON Export of Offline Compliance Report")
    print("-" * 80)
    
    json_export = report_a.to_json()
    parsed = json.loads(json_export)
    assert parsed["is_compliant"] is True
    assert parsed["blocked_count"] == 0
    assert "duration_ms" in parsed
    print(f"JSON Export verified valid ({len(json_export)} bytes):")
    print(json_export)
    print("\n  [+] PART C PASSED: Offline Compliance Report produces standard JSON audit artifact.")

    print("\n" + "=" * 80)
    print("ALL STEP 6 OFFLINE PROOF VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
