#!/usr/bin/env python3
"""
overleaf_agent.py — Interactive REPL for live-editing Overleaf papers with an LLM
==================================================================================

HOW TO USE IN AN AGENT SESSION
------------------------------

1. Resolve your session cookie:
   By default the script reads `overleaf_session2` from Firefox's cookie jar.
   You can still pass `--cookie` explicitly to override this.

2. Start a REPL session:

   $ python3 overleaf_agent.py <project_id>

   If the project has multiple documents you will be shown a numbered list
   and asked to pick one:

     [1] main.tex          (doc: 69a81e69bdcdae63f5c4be41, v526)
     [2] references/refs.bib  ...
   Select document [1-2]: 1

   Then type instructions one after another:

     Editing: main.tex (v526, 18069 chars)
     >>> add an easter egg in the conclusion
       Assistant: Added a hidden comment in the conclusion
     Applying 1 OT operation...
     Done. (v527)
     >>> make the abstract shorter
     ...
     >>> exit

3. To skip the file picker, pass --doc-id:
   $ python3 overleaf_agent.py <project_id> \
       --doc-id 69a81e69bdcdae63f5c4be41

HOW IT WORKS
------------
  Session startup:
    - Connects to Overleaf via Socket.io (WebSocket, kept alive in a thread)
    - Joins the project and the selected document
    - Receives the current document content + version number

  On each instruction:
    - Sends current content + instruction to the model
    - The model returns the modified content via tool use
    - Diffs old vs new → minimal insert/delete OT operations
    - Pushes ops via the live WebSocket (applyOtUpdate)
    - Changes appear in Overleaf instantly; version increments

  Background WebSocket thread:
    - Heartbeats keep the connection alive between instructions
    - otUpdateApplied events from other collaborators are applied to the
      local content copy so the model always sees the latest version

USING FROM WITHIN AN AGENT SESSION
---------------------------------
Add to your local agent instructions:

  ## Tools
  - python3 overleaf_agent.py <project_id> [--cookie $OVERLEAF_COOKIE] [--doc-id <id>]
    Opens an interactive REPL to live-edit the Overleaf document.

Default auth source:
  Firefox cookies.sqlite → overleaf_session2

Override sources:
  --cookie
  OVERLEAF_COOKIE

For non-interactive use from an agent pass --doc-id so the file picker is
skipped.

Dependencies:
  pip install requests websocket-client anthropic

Environment (one of):
  ANTHROPIC_API_KEY=sk-ant-...
  ANTHROPIC_BASE_URL=... + ANTHROPIC_AUTH_TOKEN=...   (agent proxy)
  --api-key flag
"""

import argparse
import configparser
import difflib
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import anthropic


# ---------------------------------------------------------------------------
# DNS fallback via getent (handles filtered /etc/resolv.conf)
# ---------------------------------------------------------------------------

@contextmanager
def dns_override_if_needed(url: str):
    """
    If Python's resolver can't resolve the host in url (e.g. because
    /etc/resolv.conf points to filtered public DNS), fall back to
    `getent hosts` which uses the full nsswitch chain (systemd-resolved,
    NSCD, /etc/hosts).  Monkey-patches socket.getaddrinfo for the duration.
    """
    hostname = urlparse(url).hostname
    if not hostname:
        yield
        return

    try:
        socket.getaddrinfo(hostname, 443)
        yield
        return
    except socket.gaierror:
        pass

    try:
        result = subprocess.run(
            ["getent", "hosts", hostname],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            yield
            return
        ip = result.stdout.split()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        yield
        return

    print(f"  [dns] {hostname} → {ip} (via getent fallback)")
    original_getaddrinfo = socket.getaddrinfo

    def patched(host, port, *args, **kwargs):
        if host == hostname:
            return original_getaddrinfo(ip, port, *args, **kwargs)
        return original_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = patched
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


# ---------------------------------------------------------------------------
# Firefox cookie lookup
# ---------------------------------------------------------------------------

def _default_firefox_profile_dir() -> Path | None:
    root = Path.home() / ".mozilla" / "firefox"
    profiles_ini = root / "profiles.ini"
    if not profiles_ini.exists():
        return None

    cfg = configparser.ConfigParser()
    cfg.read(profiles_ini)

    for section in cfg.sections():
        if section.startswith("Install") and cfg.has_option(section, "Default"):
            candidate = root / cfg.get(section, "Default")
            if candidate.exists():
                return candidate

    for section in cfg.sections():
        if not section.startswith("Profile"):
            continue
        if cfg.get(section, "Default", fallback="0") != "1":
            continue
        rel = cfg.get(section, "IsRelative", fallback="1") == "1"
        path = Path(cfg.get(section, "Path", fallback=""))
        candidate = root / path if rel else path
        if candidate.exists():
            return candidate

    dbs = sorted(root.glob("*/cookies.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dbs[0].parent if dbs else None


def load_overleaf_cookie(
    explicit_cookie: str | None = None,
    firefox_profile: str | None = None,
    cookie_db: str | None = None,
) -> str:
    """
    Resolve overleaf_session2, preferring:
      1. explicit_cookie argument
      2. OVERLEAF_COOKIE env var
      3. Firefox cookies.sqlite
    """
    if explicit_cookie:
        return explicit_cookie

    env_cookie = os.environ.get("OVERLEAF_COOKIE")
    if env_cookie:
        return env_cookie

    candidates: list[Path] = []
    if cookie_db:
        candidates.append(Path(cookie_db).expanduser())
    else:
        profile_dir = Path(firefox_profile).expanduser() if firefox_profile else _default_firefox_profile_dir()
        if profile_dir is not None:
            candidates.append(profile_dir / "cookies.sqlite")
        root = Path.home() / ".mozilla" / "firefox"
        candidates.extend(
            p for p in sorted(root.glob("*/cookies.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
            if p not in candidates
        )

    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        print(
            "Error: could not resolve overleaf_session2.\n"
            "Tried:\n"
            "  --cookie\n"
            "  $OVERLEAF_COOKIE\n"
            "  Firefox cookies.sqlite\n"
            "Pass --cookie explicitly or use --firefox-profile / --cookie-db."
        )
        sys.exit(1)

    best_row = None
    best_db = None
    for db_path in candidates:
        tmp_dir = Path(tempfile.mkdtemp(prefix="overleaf-cookie-"))
        tmp_db = tmp_dir / "cookies.sqlite"
        try:
            shutil.copy2(db_path, tmp_db)
            with sqlite3.connect(tmp_db) as conn:
                row = conn.execute(
                    """
                    SELECT value, lastAccessed
                    FROM moz_cookies
                    WHERE name = 'overleaf_session2'
                      AND (host = '.overleaf.com' OR host = 'www.overleaf.com')
                    ORDER BY lastAccessed DESC
                    LIMIT 1
                    """
                ).fetchone()
            if row and (best_row is None or row[1] > best_row[1]):
                best_row = row
                best_db = db_path
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if best_row and best_row[0]:
        return best_row[0]

    print(
        "Error: overleaf_session2 not found in Firefox cookie jars.\n"
        f"Checked: {', '.join(str(p) for p in candidates)}\n"
        "Pass --cookie explicitly if your active browser session is elsewhere."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# OT helpers
# ---------------------------------------------------------------------------

def compute_ot_ops(old: str, new: str) -> list[dict]:
    """
    Diff old → new and return a list of ShareJS ops in reverse position order
    so they can be applied sequentially without shifting offsets.
    """
    if old == new:
        return []

    # Fast path for the common case in editor sync: one contiguous edit inside
    # a large document. This avoids SequenceMatcher's pathological latency on
    # near-identical long strings.
    prefix = 0
    max_prefix = min(len(old), len(new))
    while prefix < max_prefix and old[prefix] == new[prefix]:
        prefix += 1

    old_suffix_idx = len(old)
    new_suffix_idx = len(new)
    while (
        old_suffix_idx > prefix
        and new_suffix_idx > prefix
        and old[old_suffix_idx - 1] == new[new_suffix_idx - 1]
    ):
        old_suffix_idx -= 1
        new_suffix_idx -= 1

    old_mid = old[prefix:old_suffix_idx]
    new_mid = new[prefix:new_suffix_idx]
    if old_mid or new_mid:
        ops = []
        if old_mid:
            ops.append({"p": prefix, "d": old_mid})
        if new_mid:
            ops.append({"p": prefix, "i": new_mid})
        return list(reversed(ops))

    ops = []
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            ops.append({"p": i1, "i": new[j1:j2]})
        elif tag == "delete":
            ops.append({"p": i1, "d": old[i1:i2]})
        elif tag == "replace":
            ops.append({"p": i1, "d": old[i1:i2]})
            ops.append({"p": i1, "i": new[j1:j2]})
    return list(reversed(ops))


def apply_sharejs_ops(content: str, ops: list[dict]) -> str:
    """
    Apply a list of ShareJS ops received from the server to a local string.
    Op format: {"p": position, "i": inserted_text} or {"p": position, "d": deleted_text}
    """
    for op in ops:
        p = op["p"]
        if "i" in op:
            content = content[:p] + op["i"] + content[p:]
        elif "d" in op:
            content = content[:p] + content[p + len(op["d"]):]
    return content


# ---------------------------------------------------------------------------
# Socket.io client with event handler support
# ---------------------------------------------------------------------------

class TrackingSocketIOClient:
    """
    Thin wrapper around SocketIOClient that adds named event handler
    registration and suppresses the default noisy logging.
    """

    def __init__(self, session, project_id: str):
        sys.path.insert(0, str(Path(__file__).parent))
        from overleaf_cli import SocketIOClient as _Base
        self._base = _Base(session, project_id)
        self._handlers: dict[str, callable] = {}
        self._orig_on_close = self._base._on_close
        # Override the base _handle_event to route through our handlers
        self._base._handle_event = self._handle_event
        self._base._on_close = self._on_close

    def on(self, event_name: str, handler: callable):
        self._handlers[event_name] = handler

    def _handle_event(self, name: str, args: list):
        if name in self._handlers:
            try:
                self._handlers[name](args)
            except Exception as e:
                print(f"  [ws] handler error for {name}: {e}")

    def _on_close(self, ws, code, msg):
        self._base._connected.clear()
        self._orig_on_close(ws, code, msg)

    # Delegate everything else to the base client
    def connect(self): return self._base.connect()
    def run_forever(self): return self._base.run_forever()

    def join_project(self) -> dict:
        """
        Send joinProject and collect the response.
        Overleaf may respond via an ack (type 6) OR a joinProjectResponse
        event (type 5); handle whichever arrives first.
        """
        result = {}
        done = threading.Event()

        def on_event(args):
            # event path: args[0] = {"publicId": ..., "project": {rootFolder, ...}}
            if args and isinstance(args[0], dict):
                result.update(args[0].get("project", args[0]))
            done.set()

        def on_ack(data):
            # ack path: data = [null, {rootFolder, ...}]
            if data and len(data) > 1 and data[1]:
                result.update(data[1])
            done.set()

        self._handlers["joinProjectResponse"] = on_event
        self._base._connected.wait(timeout=10)
        self._base.send_event(
            "joinProject", [{"project_id": self._base.project_id}], callback=on_ack
        )
        done.wait(timeout=10)
        self._handlers.pop("joinProjectResponse", None)
        return result

    def join_doc(self, doc_id): return self._base.join_doc(doc_id)
    def leave_doc(self, doc_id): return self._base.leave_doc(doc_id)
    def send_event(self, *a, **kw): return self._base.send_event(*a, **kw)
    def disconnect(self): return self._base.disconnect()
    def is_connected(self): return self._base._running and self._base._connected.is_set()


# ---------------------------------------------------------------------------
# Project doc-tree helpers
# ---------------------------------------------------------------------------

def collect_docs_from_project(project_data: dict) -> dict[str, str]:
    """
    Extract {doc_id: pathname} by walking the rootFolder tree returned by
    the joinProject WebSocket event. Works for any project regardless of
    edit history depth.
    """
    def _recurse(folder: dict, prefix: str) -> dict[str, str]:
        out = {}
        for doc in folder.get("docs", []):
            out[doc["_id"]] = prefix + doc["name"]
        for sub in folder.get("folders", []):
            out.update(_recurse(sub, prefix + sub["name"] + "/"))
        return out

    docs: dict[str, str] = {}
    for root in project_data.get("rootFolder", []):
        docs.update(_recurse(root, ""))
    return docs


def pick_doc_interactively(docs: dict[str, str]) -> tuple[str, str]:
    """
    Show a numbered list of documents and ask the user to pick one.
    docs: {doc_id: pathname}. Returns (doc_id, pathname).
    """
    if not docs:
        print("Error: no documents found in project.")
        sys.exit(1)

    entries = sorted(docs.items(), key=lambda x: x[1])
    print()
    for i, (doc_id, pathname) in enumerate(entries, 1):
        print(f"  [{i}] {pathname:<40} (doc: {doc_id})")
    print()

    while True:
        try:
            choice = int(input(f"Select document [1-{len(entries)}]: ").strip())
            if 1 <= choice <= len(entries):
                return entries[choice - 1]
        except (ValueError, EOFError):
            pass
        print(f"  Enter a number between 1 and {len(entries)}.")


# ---------------------------------------------------------------------------
# Model API
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert LaTeX and academic writing assistant helping edit an Overleaf paper.
You have access to the full document content.

When asked to make changes:
- Use the edit_file tool to return the complete modified file content.
- Preserve all existing LaTeX commands, formatting conventions, and structure
  unless the instruction explicitly asks to change them.
- Be precise about LaTeX syntax.
"""


def make_anthropic_client(api_key: str | None = None) -> tuple[anthropic.Anthropic, str | None]:
    """
    Build an Anthropic client.  Returns (client, base_url).
    base_url is non-None only for the ANTHROPIC_BASE_URL path so that
    dns_override_if_needed knows which host to watch.

    Priority:
      1. --api-key / explicit api_key argument
      2. ANTHROPIC_API_KEY env var
          3. ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN  (e.g. local agent proxy)
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return anthropic.Anthropic(api_key=key), None

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if base_url and auth_token:
        return anthropic.Anthropic(base_url=base_url, auth_token=auth_token), base_url

    print("Error: no Anthropic credentials found.\n"
          "Set one of:\n"
          "  export ANTHROPIC_API_KEY=sk-ant-...\n"
          "  export ANTHROPIC_BASE_URL=... ANTHROPIC_AUTH_TOKEN=...\n"
          "Or pass: --api-key sk-ant-...")
    sys.exit(1)


def ask_model_to_edit(path: str, content: str, instruction: str,
                      model: str = "claude-sonnet-4-6",
                      api_key: str | None = None) -> str | None:
    """Ask the configured model to edit content. Returns new content or None."""
    client, base_url = make_anthropic_client(api_key)

    tools = [{
        "name": "edit_file",
        "description": "Return the complete modified file content",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string",
                            "description": "Complete new file content after edits"},
                "reason":  {"type": "string",
                            "description": "One-sentence summary of what was changed"},
            },
            "required": ["content", "reason"],
        },
    }]

    messages = [{
        "role": "user",
        "content": (
            f"=== {path} ===\n{content}\n\n"
            f"---\n\nInstruction: {instruction}\n\n"
            "Apply the instruction and call edit_file with the complete new content."
        ),
    }]

    # Need enough output tokens to return the full (possibly expanded) document.
    # 8192 is insufficient for documents >~25k chars. Streaming is required by
    # the SDK for max_tokens values that could exceed 10 minutes.
    max_tokens = 32768

    with dns_override_if_needed(base_url or "https://api.anthropic.com"):
        chars = 0
        last_dot = 0
        print("  Streaming", end="", flush=True)

        with client.messages.stream(
            model=model, max_tokens=max_tokens,
            system=SYSTEM_PROMPT, tools=tools, messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta and hasattr(delta, "partial_json"):
                        chars += len(delta.partial_json)
                        if chars - last_dot >= 500:
                            print(".", end="", flush=True)
                            last_dot = chars
            print(f" ({chars} chars)")
            response = stream.get_final_message()

        if response.stop_reason == "max_tokens":
            print(f"  Error: response hit the {max_tokens}-token limit. "
                  "The document may be too large for a single edit.")
            return None

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        for tc in tool_calls:
            if tc.name == "edit_file":
                reason = tc.input.get("reason", "")
                new_content = tc.input.get("content")
                if reason:
                    print(f"  Assistant: {reason}")
                if new_content is None:
                    print("  Error: model did not return file content.")
                    return None
                return new_content

    return None


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

class OverleafREPL:
    """
    Persistent editing session against a single Overleaf document.

    The WebSocket connection stays open in a background thread.
    otUpdateApplied events from other collaborators are applied to the
    local content copy so each model call always sees the current text.
    """

    def __init__(self, project_id: str, session_cookie: str,
                 model: str, api_key: str | None):
        self.project_id = project_id
        self.session_cookie = session_cookie
        self.model = model
        self.api_key = api_key

        self._lock = threading.Lock()
        self._content = ""
        self._version = 0
        self._pending_versions: set[int] = set()  # ops we sent, skip echo

        self.doc_id: str = ""
        self.doc_path: str = ""
        self._ws: TrackingSocketIOClient | None = None

    # -- thread-safe content/version accessors --------------------------------

    def _get_state(self) -> tuple[str, int]:
        with self._lock:
            return self._content, self._version

    def _set_state(self, content: str, version: int):
        with self._lock:
            self._content = content
            self._version = version

    # -- WebSocket event handler (background thread) --------------------------

    def _on_ot_update(self, args: list):
        """
        Receive otUpdateApplied from the server.
        If it's from another user, apply their ops to our local content.
        """
        payload = args[0] if args else {}
        if payload.get("doc") != self.doc_id:
            return

        v = payload.get("v")
        ops = payload.get("op", [])

        with self._lock:
            if v in self._pending_versions:
                # This is the echo of our own op — already counted via ack.
                self._pending_versions.discard(v)
                return
            if ops:
                self._content = apply_sharejs_ops(self._content, ops)
            # v is the version the op was applied at; new version is v+1
            if v is not None:
                self._version = v + 1

    # -- REPL startup ---------------------------------------------------------

    def run(self, doc_id_hint: str | None = None):
        sys.path.insert(0, str(Path(__file__).parent))
        from overleaf_cli import OverleafSession

        s = OverleafSession(self.session_cookie)
        s.get_project_bootstrap(self.project_id)

        # Connect WebSocket — joinProject returns the full doc tree
        self._ws = TrackingSocketIOClient(s, self.project_id)
        self._ws.on("otUpdateApplied", self._on_ot_update)
        self._ws.connect()
        self._ws.run_forever()
        project_data = self._ws.join_project()
        docs = collect_docs_from_project(project_data)

        # Resolve document
        if doc_id_hint:
            self.doc_id = doc_id_hint
            self.doc_path = docs.get(doc_id_hint, doc_id_hint)
        else:
            self.doc_id, self.doc_path = pick_doc_interactively(docs)

        # Join doc — authoritative content + version from server
        lines, version = self._ws.join_doc(self.doc_id)
        self._set_state("\n".join(lines), version)

        content, version = self._get_state()
        print(f"\nEditing: {self.doc_path}  (v{version}, {len(content)} chars)")
        print("Type an instruction, or 'exit' / Ctrl-D to quit.\n")

        try:
            while True:
                try:
                    instruction = input(">>> ").strip()
                except EOFError:
                    print()
                    break
                if not instruction:
                    continue
                if instruction.lower() in ("exit", "quit"):
                    break
                self._handle(instruction)
        finally:
            self._ws.leave_doc(self.doc_id)
            self._ws.disconnect()
            print("Session ended.")

    # -- single instruction ---------------------------------------------------

    def _handle(self, instruction: str):
        current, _ = self._get_state()

        print("Asking model...")
        new_content = ask_model_to_edit(
            self.doc_path, current, instruction,
            model=self.model, api_key=self.api_key,
        )
        if new_content is None or new_content == current:
            print("No changes.")
            return

        ops = compute_ot_ops(current, new_content)
        if not ops:
            print("No changes.")
            return

        print(f"Applying {len(ops)} OT operation(s)...")
        if self._apply_ops(ops, new_content):
            _, v = self._get_state()
            print(f"Done. (v{v})")

    def _apply_ops(self, ops: list[dict], expected_new_content: str) -> bool:
        for i, op in enumerate(ops):
            kind = "insert" if "i" in op else "delete"
            text = op.get("i") or op.get("d", "")
            print(f"  [{i+1}/{len(ops)}] {kind} at {op['p']}: {repr(text[:50])}")

            _, v = self._get_state()

            with self._lock:
                self._pending_versions.add(v)

            update = {
                "doc": self.doc_id,
                "op": [op],
                "v": v,
                "dupIfSource": [],
            }
            done = threading.Event()
            ok = [False]

            def on_ack(data, _ok=ok, _done=done):
                _ok[0] = data and data[0] is None
                _done.set()

            self._ws.send_event("applyOtUpdate", [self.doc_id, update], callback=on_ack)
            done.wait(timeout=10)

            if ok[0]:
                with self._lock:
                    self._version += 1
                    self._pending_versions.discard(v)
            else:
                with self._lock:
                    self._pending_versions.discard(v)
                print(f"  Op {i+1} failed (version conflict). Re-syncing...")
                lines, new_v = self._ws.join_doc(self.doc_id)
                self._set_state("\n".join(lines), new_v)
                print(f"  Re-synced to v{new_v}. Re-run your instruction.")
                return False

        # All ops succeeded — update local content to the expected new state
        with self._lock:
            self._content = expected_new_content
        return True


# ---------------------------------------------------------------------------
# Listen mode
# ---------------------------------------------------------------------------

class OverleafListener:
    """
    Passive sync: joins every live document in a project, writes initial
    content to disk, then applies incoming OT updates in real time.

    No model, no editing — purely Overleaf → local filesystem.
    """

    def __init__(self, project_id: str, session_cookie: str, output_dir: str):
        self.project_id = project_id
        self.session_cookie = session_cookie
        self.output_dir = Path(output_dir)

        self._lock = threading.Lock()
        self._docs: dict[str, dict] = {}  # doc_id → {content, version, path}
        self._ws: TrackingSocketIOClient | None = None

    def _on_ot_update(self, args: list):
        payload = args[0] if args else {}
        doc_id = payload.get("doc")
        v = payload.get("v")
        ops = payload.get("op", [])

        with self._lock:
            if doc_id not in self._docs:
                return
            state = self._docs[doc_id]
            if ops:
                state["content"] = apply_sharejs_ops(state["content"], ops)
            if v is not None:
                state["version"] = v + 1
            out_path = self.output_dir / state["path"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(state["content"], encoding="utf-8")
            label = (state["path"], state["version"])

        print(f"  [{label[0]}] v{label[1]} updated")

    def run(self):
        sys.path.insert(0, str(Path(__file__).parent))
        from overleaf_cli import OverleafSession

        s = OverleafSession(self.session_cookie)
        s.get_project_bootstrap(self.project_id)

        # Connect WebSocket — joinProject returns the full doc tree
        self._ws = TrackingSocketIOClient(s, self.project_id)
        self._ws.on("otUpdateApplied", self._on_ot_update)
        self._ws.connect()
        self._ws.run_forever()
        project_data = self._ws.join_project()
        docs = collect_docs_from_project(project_data)

        if not docs:
            print("Error: no documents found in project.")
            sys.exit(1)

        # Join each doc and write initial content to disk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSyncing {len(docs)} document(s) → {self.output_dir}/\n")
        for doc_id, pathname in sorted(docs.items(), key=lambda x: x[1]):
            lines, version = self._ws.join_doc(doc_id)
            content = "\n".join(lines)
            with self._lock:
                self._docs[doc_id] = {"content": content, "version": version, "path": pathname}
            out_path = self.output_dir / pathname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"  {pathname:<45} v{version}  ({len(content)} chars)")

        print(f"\nListening for changes... (Ctrl-C to stop)\n")

        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            for doc_id in list(self._docs.keys()):
                self._ws.leave_doc(doc_id)
            self._ws.disconnect()
            print("Done.")


# ---------------------------------------------------------------------------
# Bidirectional sync mode
# ---------------------------------------------------------------------------

import time as _time


class OverleafSync:
    """
    Bidirectional sync: Overleaf ↔ local folder.

    - Incoming OT updates from Overleaf → applied to memory + written to disk.
    - Local file changes (detected by polling every 0.5 s) → diffed against
      the last-known Overleaf state → pushed as OT ops via applyOtUpdate.

    Conflict resolution: if an applyOtUpdate is rejected (version mismatch),
    we re-join the doc from the server and overwrite the local file.
    """

    POLL_INTERVAL = 0.5  # seconds between local file checks

    def __init__(self, project_id: str, session_cookie: str, output_dir: str):
        self.project_id = project_id
        self.session_cookie = session_cookie
        self.output_dir = Path(output_dir)

        self._lock = threading.Lock()
        # doc_id → {content, version, path, pending_versions}
        self._docs: dict[str, dict] = {}
        self._ws: TrackingSocketIOClient | None = None
        self._running = False
        self._ws_ready = threading.Event()
        self._watch_thread = None
        self._inotify_proc = None

    # -- incoming from Overleaf -----------------------------------------------

    def _on_ot_update(self, args: list):
        payload = args[0] if args else {}
        doc_id = payload.get("doc")
        v = payload.get("v")
        ops = payload.get("op", [])

        with self._lock:
            if doc_id not in self._docs:
                return
            state = self._docs[doc_id]

            # Two-layer echo suppression:
            # Layer 1 — echo arrives before ack: v is still in pending_versions.
            if v in state["pending_versions"]:
                state["pending_versions"].discard(v)
                return
            # Layer 2 — echo arrives after ack: version was already incremented
            # to v+1 in _push_change, so v < state["version"] means stale.
            if v is not None and v < state["version"]:
                return

            if ops:
                state["content"] = apply_sharejs_ops(state["content"], ops)
            if v is not None:
                state["version"] = v + 1
            # Write to disk while holding the lock so the poll loop always
            # sees memory and file in a consistent state.
            out_path = self.output_dir / state["path"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(state["content"], encoding="utf-8")
            label = (state["path"], state["version"])

        print(f"  [overleaf→local] {label[0]}  v{label[1]}")

    # -- outgoing to Overleaf -------------------------------------------------

    def _watch_files(self):
        """
        Background thread: watch the output directory with inotifywait.
        Fires on close_write (in-place save) and moved_to (vi/emacs atomic
        write-then-rename).  Pushes changes to Overleaf immediately.
        """
        # Build a map from absolute path → doc_id for fast lookup.
        path_to_doc: dict[str, str] = {
            str((self.output_dir / state["path"]).resolve()): doc_id
            for doc_id, state in self._docs.items()
        }

        watch_root = str(self.output_dir.resolve())
        print(f"  [inotify] watching: {watch_root}", flush=True)
        for p in path_to_doc:
            print(f"  [inotify]   tracking: {p}", flush=True)

        cmd = [
            "inotifywait", "-m", "-r",
            "-e", "close_write", "-e", "moved_to",
            "--format", "%w%f",
            watch_root,
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except FileNotFoundError:
            print("  [sync] inotifywait not found, falling back to polling", flush=True)
            self._poll_files_fallback()
            return

        self._inotify_proc = proc
        for line in proc.stdout:
            if not self._running:
                break
            filepath = line.strip()
            print(f"  [inotify] event: {filepath!r}", flush=True)
            doc_id = path_to_doc.get(filepath)
            if doc_id is None:
                print(f"  [inotify] (not a tracked file, skipping)", flush=True)
                continue
            try:
                file_content = Path(filepath).read_text(encoding="utf-8")
            except FileNotFoundError:
                print(f"  [inotify] file gone (mid-write?), skipping", flush=True)
                continue
            with self._lock:
                state = self._docs.get(doc_id)
                if state is None or file_content == state["content"]:
                    print(f"  [inotify] no content change", flush=True)
                    continue
                old_content = state["content"]
            self._push_change(doc_id, old_content, file_content)

        proc.stdout.close()

    def _poll_files_fallback(self):
        """Polling fallback when inotifywait is unavailable."""
        while self._running:
            try:
                changed: list[tuple[str, str, str]] = []
                with self._lock:
                    for doc_id, state in list(self._docs.items()):
                        out_path = self.output_dir / state["path"]
                        try:
                            fc = out_path.read_text(encoding="utf-8")
                        except FileNotFoundError:
                            continue
                        if fc != state["content"]:
                            changed.append((doc_id, state["content"], fc))
                for doc_id, old, new in changed:
                    self._push_change(doc_id, old, new)
            except Exception as e:
                print(f"  [sync] poll error: {e}", flush=True)
            _time.sleep(0.5)

    def _push_change(self, doc_id: str, old_content: str, new_content: str):
        ops = compute_ot_ops(old_content, new_content)
        if not ops:
            return True

        with self._lock:
            # Bail if a remote update already changed the baseline.
            if self._docs[doc_id]["content"] != old_content:
                print(f"  [sync] skipping push (remote update arrived first)", flush=True)
                return False
            pathname = self._docs[doc_id]["path"]
            ws = self._ws

        if not self._ws_ready.wait(timeout=10):
            print(f"  [sync] websocket unavailable, deferring push for {pathname}", flush=True)
            return False
        if ws is None or not ws.is_connected():
            print(f"  [sync] websocket disconnected, deferring push for {pathname}", flush=True)
            return False

        print(f"  [local→overleaf] {pathname}  ({len(ops)} op(s))", flush=True)

        for op in ops:
            with self._lock:
                state = self._docs[doc_id]
                v = state["version"]
                state["pending_versions"].add(v)

            update = {"doc": doc_id, "op": [op], "v": v, "dupIfSource": []}
            done = threading.Event()
            ok = [False]

            def on_ack(data, _ok=ok, _done=done):
                _ok[0] = data and data[0] is None
                _done.set()

            try:
                ws.send_event("applyOtUpdate", [doc_id, update], callback=on_ack)
            except Exception as e:
                with self._lock:
                    self._docs[doc_id]["pending_versions"].discard(v)
                print(f"  [sync] send failed for {pathname}: {e}", flush=True)
                self._ws_ready.clear()
                return False

            done.wait(timeout=10)

            with self._lock:
                # Discard pending and increment version atomically so that
                # the otUpdateApplied echo (which may arrive after the ack)
                # cannot slip through the pending_versions guard.
                self._docs[doc_id]["pending_versions"].discard(v)
                if ok[0]:
                    self._docs[doc_id]["version"] += 1

            if not ok[0]:
                if not ws.is_connected():
                    print(f"  [sync] websocket dropped during push for {pathname}", flush=True)
                    self._ws_ready.clear()
                    return False
                # Version conflict — pull fresh state from server.
                print(f"  [sync] conflict on {pathname}, re-syncing from Overleaf...", flush=True)
                lines, new_v = ws.join_doc(doc_id)
                server_content = "\n".join(lines)
                with self._lock:
                    self._docs[doc_id]["content"] = server_content
                    self._docs[doc_id]["version"] = new_v
                out_path = self.output_dir / pathname
                out_path.write_text(server_content, encoding="utf-8")
                print(f"  [sync] reset to server v{new_v}", flush=True)
                return False  # discard remaining ops for this edit

        # All ops applied — commit new content as the new baseline.
        with self._lock:
            self._docs[doc_id]["content"] = new_content
        return True

    def _disconnect_ws(self):
        self._ws_ready.clear()
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            for doc_id in list(self._docs.keys()):
                ws.leave_doc(doc_id)
        except Exception:
            pass
        try:
            ws.disconnect()
        except Exception:
            pass

    def _connect_and_sync_state(self, initial: bool):
        sys.path.insert(0, str(Path(__file__).parent))
        from overleaf_cli import OverleafSession

        session = OverleafSession(self.session_cookie)
        session.get_project_bootstrap(self.project_id)

        ws = TrackingSocketIOClient(session, self.project_id)
        ws.on("otUpdateApplied", self._on_ot_update)
        ws.connect()
        ws.run_forever()
        project_data = ws.join_project()
        docs = collect_docs_from_project(project_data)
        if not docs:
            raise RuntimeError("no documents found in project")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        pending_local: list[tuple[str, str, str, str]] = []
        with self._lock:
            previous_content = {
                doc_id: state["content"] for doc_id, state in self._docs.items()
            }

        for doc_id, pathname in sorted(docs.items(), key=lambda x: x[1]):
            lines, version = ws.join_doc(doc_id)
            server_content = "\n".join(lines)
            out_path = self.output_dir / pathname
            local_content = None
            if out_path.exists():
                local_content = out_path.read_text(encoding="utf-8")
            with self._lock:
                self._docs[doc_id] = {
                    "content": server_content,
                    "version": version,
                    "path": pathname,
                    "pending_versions": set(),
                }
                prior = previous_content.get(doc_id)
            if (
                not initial
                and prior is not None
                and local_content is not None
                and local_content != prior
            ):
                pending_local.append((doc_id, pathname, server_content, local_content))
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(server_content, encoding="utf-8")
            if initial:
                print(f"  {pathname:<45} v{version}  ({len(server_content)} chars)")

        self._ws = ws
        self._ws_ready.set()

        for doc_id, pathname, server_content, local_content in pending_local:
            print(f"  [sync] replaying local edits after reconnect: {pathname}", flush=True)
            if not self._push_change(doc_id, server_content, local_content):
                print(f"  [sync] deferred local replay for {pathname}", flush=True)

    # -- startup --------------------------------------------------------------

    def run(self):
        self._running = True
        try:
            self._connect_and_sync_state(initial=True)
        except Exception as e:
            print(f"Error: initial sync failed: {e}")
            sys.exit(1)

        print(f"\nBidirectional sync active... (Ctrl-C to stop)\n")
        self._watch_thread = threading.Thread(target=self._watch_files, daemon=True)
        self._watch_thread.start()

        try:
            while self._running:
                _time.sleep(1)
                ws = self._ws
                if ws is not None and ws.is_connected():
                    continue
                print("  [sync] websocket disconnected, reconnecting...", flush=True)
                self._disconnect_ws()
                while self._running:
                    try:
                        self._connect_and_sync_state(initial=False)
                        print("  [sync] reconnect complete", flush=True)
                        break
                    except Exception as e:
                        print(f"  [sync] reconnect failed: {e}", flush=True)
                        _time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._running = False
            if self._inotify_proc:
                self._inotify_proc.terminate()
            self._disconnect_ws()
            print("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live Overleaf ↔ local sync and model editing REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # zero-arg sync from a git repo with an Overleaf remote
              python3 overleaf_agent.py --sync

              # interactive file picker → REPL
              python3 overleaf_agent.py 69a0589216d96e98bc76b8c6 \\
                  --cookie "s:abc123..."

              # skip file picker
              python3 overleaf_agent.py 69a0589216d96e98bc76b8c6 \\
                  --cookie "s:abc123..." --doc-id 69a81e69bdcdae63f5c4be41

              # passive mirror: Overleaf → local
              python3 overleaf_agent.py 69a0589216d96e98bc76b8c6 \\
                  --cookie "s:abc123..." --listen

              # bidirectional sync: Overleaf ↔ local
              python3 overleaf_agent.py 69a0589216d96e98bc76b8c6 \\
                  --cookie "s:abc123..." --sync
        """),
    )
    parser.add_argument("project_id", nargs="?", default=None,
                        help="Overleaf project ID (from the URL); auto-detected from git remote if omitted)")
    parser.add_argument("--cookie", default=None,
                        help="overleaf_session2 cookie value (overrides Firefox/default lookup)")
    parser.add_argument("--firefox-profile", default=None,
                        help="Firefox profile directory to read cookies.sqlite from")
    parser.add_argument("--cookie-db", default=None,
                        help="Path to a specific Firefox cookies.sqlite database")
    parser.add_argument("--listen", action="store_true",
                        help="Passive mode: Overleaf → local folder, read-only")
    parser.add_argument("--sync", action="store_true",
                        help="Bidirectional mode: Overleaf ↔ local folder")
    parser.add_argument("--output-dir", default=None,
                        help="Local folder for --listen/--sync (default: current working directory)")
    parser.add_argument("--doc-id", default=None,
                        help="Document ID for REPL mode (omit to pick interactively)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Model name (default: claude-sonnet-4-6)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (default: $ANTHROPIC_API_KEY)")

    args = parser.parse_args()

    if args.project_id is None:
        import re as _re
        import subprocess as _sp
        try:
            remotes_out = _sp.check_output(
                ["git", "remote", "-v"], stderr=_sp.PIPE, text=True
            )
        except Exception as exc:
            parser.error(f"No project_id given and 'git remote -v' failed: {exc}")
        m = _re.search(r"overleaf\.com/([a-f0-9]{24})", remotes_out)
        if not m:
            parser.error(
                f"No project_id given and no Overleaf remote found.\n"
                f"git remotes:\n{remotes_out.strip() or '  (none)'}\n"
                f"Add one with: git remote add origin https://git.overleaf.com/<project_id>"
            )
        args.project_id = m.group(1)
        remote_url = _re.search(r"\S+overleaf\S+", remotes_out).group(0)  # type: ignore[union-attr]
        print(f"[auto] project_id={args.project_id} (from git remote {remote_url})", flush=True)

    if args.output_dir is None:
        import os as _os
        args.output_dir = _os.getcwd()

    session_cookie = load_overleaf_cookie(
        explicit_cookie=args.cookie,
        firefox_profile=args.firefox_profile,
        cookie_db=args.cookie_db,
    )

    if args.sync:
        OverleafSync(
            project_id=args.project_id,
            session_cookie=session_cookie,
            output_dir=args.output_dir,
        ).run()
    elif args.listen:
        OverleafListener(
            project_id=args.project_id,
            session_cookie=session_cookie,
            output_dir=args.output_dir,
        ).run()
    else:
        OverleafREPL(
            project_id=args.project_id,
            session_cookie=session_cookie,
            model=args.model,
            api_key=args.api_key,
        ).run(doc_id_hint=args.doc_id)


if __name__ == "__main__":
    main()
