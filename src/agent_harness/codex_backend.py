"""Codex app-server backend with role-explicit, fail-closed model provenance.

Implements the harness interface of the agent-harness study (paper App. "Agent-harness study
on SpreadsheetBench-2", "Harness and models"): one app-server process per ROLE (actor /
verifier), each thread and turn carries the role's model and reasoning effort, the served
model and effort are checked on every thread start (a different model, a missing or
different effort, a timeout or a closed stream raise HardStop and nothing is sealed). Every
turn is logged with its transport record and charged to a per-arm usage ledger; usage or
rate limits stop the run with SystemExit so it can resume later.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time

from src.agent_harness import HardStop
from src.utils.io import atomic_write, read_json

LIMIT_PATTERNS = (
    "rate limit",
    "rate_limit",
    "usage limit",
    "usage_limit",
    "429",
    "overloaded",
    "limit reached",
    "out of extra usage",
    "resets at",
    "quota",
    "usage_limit_reached",
    "insufficient_quota",
    "too many requests",
    "session limit",
    "resets ",
)
DEFAULT_LIMITS = {"turns": 6000, "turn_timeout_s": 25 * 60, "idle_timeout_s": 600}
CLIENT_NAME = "ucoe_agent_harness"


def model_matches(served, target):
    """Exact model id or its dated snapshot (target + '-' + 8 digits); never another generation."""
    return bool(served) and (served == target or re.fullmatch(re.escape(target) + r"-\d{8}", str(served)) is not None)


def app_server_cmd(model, effort):
    return [
        "codex",
        "app-server",
        "--stdio",
        "-c",
        'model="%s"' % model,
        "-c",
        'model_reasoning_effort="%s"' % effort,
        "-c",
        "features.multi_agent=false",
        "-c",
        "features.multi_agent_v2=false",
        "-c",
        "features.plugins=false",
        "-c",
        "features.skill_search=false",
        "-c",
        "features.memories=false",
        "-c",
        "features.external_agent_memory_import=false",
    ]


def audit(root, event, **kw):
    os.makedirs(os.path.join(root, "private", "audit"), exist_ok=True)
    with open(os.path.join(root, "private", "audit", "events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "event": event, **kw}, default=str) + "\n")


def transport_ref(r):
    return {
        "log": r.get("log"),
        "thread_id": r.get("thread_id"),
        "turn_id": r.get("turn_id"),
        "served": r.get("served"),
        "backend": r.get("backend"),
    }


class StdioTransport:
    def __init__(self, cmd):
        e = dict(os.environ)
        e.pop("OPENAI_API_KEY", None)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=e,
            bufsize=1,
            start_new_session=True,
        )
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for ln in self.proc.stdout:
            ln = ln.strip()
            if ln:
                try:
                    self.q.put(json.loads(ln))
                except json.JSONDecodeError:
                    self.q.put({"_raw": ln})
        self.q.put(None)

    def send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout):
        return self.q.get(timeout=timeout)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass


class AppServer:
    """One Codex app-server process per ROLE; every thread/turn carries the role's model and effort; served values are
    checked (HardStop on mismatch)."""

    def __init__(self, role, model, effort, root, limits=None, transport=None):
        self.role, self.model, self.effort, self.root = role, model, effort, root
        self.limits = dict(DEFAULT_LIMITS, **(limits or {}))
        self.t = transport or StdioTransport(app_server_cmd(self.model, self.effort))
        self.next_id, self.loaded, self.usage, self.rate_limits = 1, set(), [], None
        self.server_info = self.request(
            "initialize",
            {"clientInfo": {"name": CLIENT_NAME, "title": "UCoE agent harness (%s)" % role, "version": "1"}},
        )
        self.t.send({"jsonrpc": "2.0", "method": "initialized"})

    def request(self, method, params, timeout=120):
        rid = self.next_id
        self.next_id += 1
        self.t.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        t0 = time.time()
        while True:
            try:
                msg = self.t.recv(max(1, timeout - (time.time() - t0)))
            except queue.Empty:
                raise HardStop(
                    "transport_error",
                    "%s app-server unresponsive: no reply to %s within %ss" % (self.role, method, timeout),
                )
            if msg is None:
                raise HardStop("transport_error", "%s app-server closed the stream" % self.role)
            if msg.get("id") == rid and ("result" in msg or "error" in msg):
                if "error" in msg:
                    raise HardStop("transport_error", "%s %s: %s" % (self.role, method, msg["error"]))
                return msg["result"]
            self._notify(msg)

    def _notify(self, msg):
        if "id" in msg and "method" in msg:
            audit(self.root, "server_request_declined", role=self.role, method=msg.get("method"))
            self.t.send({"jsonrpc": "2.0", "id": msg["id"], "result": {"decision": "decline"}})
            return
        m = msg.get("method", "")
        if m == "thread/tokenUsage/updated":
            self.usage.append(msg.get("params"))
        elif m == "account/rateLimits/updated":
            self.rate_limits = msg.get("params")

    def _pin(self, r, where):
        model, eff = r.get("model"), r.get("reasoningEffort")
        served = {
            "model": model,
            "effort": (str(eff) if eff is not None else None),
            "modelProvider": r.get("modelProvider"),
        }
        if model != self.model:
            audit(
                self.root,
                "served_model_mismatch",
                role=self.role,
                where=where,
                requested=[self.model, self.effort],
                served=served,
            )
            raise HardStop(
                "model_fallback",
                "%s/%s: served model=%s != requested %s (no silent fallback)" % (self.role, where, model, self.model),
            )
        if eff is None:
            raise HardStop("missing_effort", "%s/%s: server did not report reasoningEffort" % (self.role, where))
        if str(eff) != self.effort:
            audit(
                self.root,
                "served_effort_mismatch",
                role=self.role,
                where=where,
                requested=[self.model, self.effort],
                served=served,
            )
            raise HardStop(
                "provenance_mismatch", "%s/%s: served effort=%s != requested %s" % (self.role, where, eff, self.effort)
            )
        return served

    def _thread_params(self, cwd, ephemeral):
        return {
            "cwd": cwd,
            "model": self.model,
            "approvalPolicy": "never",
            "sandbox": "workspace-write",
            "ephemeral": ephemeral,
            "config": {
                "model_reasoning_effort": self.effort,
                "features": {
                    "multi_agent": False,
                    "multi_agent_v2": False,
                    "plugins": False,
                    "skill_search": False,
                    "memories": False,
                },
            },
        }

    def thread_start(self, cwd, ephemeral=True):
        r = self.request("thread/start", self._thread_params(cwd, ephemeral))
        self.loaded.add(r["thread"]["id"])
        return {
            "served": self._pin(r, "thread/start"),
            "requested": {"model": self.model, "effort": self.effort},
            "thread": r["thread"],
        }

    def thread_delete(self, thread_id):
        return self.request("thread/delete", {"threadId": thread_id})

    def turn(self, thread_id, text, cwd, timeout=None):
        timeout = timeout or self.limits["turn_timeout_s"]
        r = self.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": text}],
                "cwd": cwd,
                "model": self.model,
                "effort": self.effort,
                "approvalPolicy": "never",
                "sandboxPolicy": {"type": "workspaceWrite", "writableRoots": [cwd], "networkAccess": False},
            },
        )
        turn_id = r["turn"]["id"]
        texts, t0, status, err, usage_before = [], time.time(), None, None, len(self.usage)
        idle = self.limits["idle_timeout_s"]
        while True:
            try:
                msg = self.t.recv(max(1, min(idle, timeout - (time.time() - t0))))
            except queue.Empty:
                try:
                    self.request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id}, timeout=15)
                except Exception:  # noqa: BLE001
                    pass
                status, err = "timeout", {"message": "no app-server message for %ds" % int(time.time() - t0)}
                break
            if msg is None:
                raise HardStop("transport_error", "%s app-server closed during turn" % self.role)
            if "id" in msg and "method" in msg:
                self._notify(msg)
                continue
            m, pr = msg.get("method", ""), msg.get("params") or {}
            self._notify(msg)
            if m == "item/completed" and pr.get("threadId") == thread_id:
                it = pr.get("item") or {}
                if it.get("type") == "agentMessage" and it.get("text"):
                    texts.append(it["text"])
            elif m == "turn/completed" and (pr.get("turn") or {}).get("id") == turn_id:
                status, err = pr["turn"].get("status"), pr["turn"].get("error")
                break
            if time.time() - t0 > timeout:
                try:
                    self.request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id}, timeout=30)
                except Exception:  # noqa: BLE001
                    pass
                status = "timeout"
                break
        return {
            "turn_id": turn_id,
            "status": status,
            "error": err,
            "text": "\n".join(texts)[-4000:],
            "seconds": round(time.time() - t0, 1),
            "token_usage": (self.usage[-1] or {}).get("tokenUsage") if len(self.usage) > usage_before else None,
            "rate_limits": self.rate_limits,
        }

    def close(self):
        self.t.close()


class CodexBackend:
    """Role-explicit transport: `roles` = {"actor": {"model", "effort"}, "verifier": {...}}; `transport_factory(role)`
    injects a mock transport."""

    def __init__(self, root, roles, limits=None, transport_factory=None):
        self.root, self.roles = root, {r: dict(v) for r, v in roles.items()}
        self.limits = dict(DEFAULT_LIMITS, **(limits or {}))
        self.factory = transport_factory
        self.inst = {}

    def _p(self, *a):
        return os.path.join(self.root, *a)

    def server(self, role):
        if role not in self.inst:
            tr = self.factory(role) if self.factory else None
            self.inst[role] = AppServer(
                role, self.roles[role]["model"], self.roles[role]["effort"], self.root, self.limits, tr
            )
        return self.inst[role]

    def restart_server(self, role):
        srv = self.inst.pop(role, None)
        if srv is not None:
            try:
                srv.close()
            except Exception:  # noqa: BLE001
                pass
        audit(self.root, "app_server_restart", role=role)
        return self.server(role)

    def close(self):
        for role in list(self.inst):
            try:
                self.inst.pop(role).close()
            except Exception:  # noqa: BLE001
                pass

    def ledger(self, arm, role, kind, rec):
        path = self._p("state", "usage_%s.json" % arm)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        led = read_json(path, None) or {"turns": 0, "usd": 0.0, "by_role": {}}
        led["turns"] += 1
        led["usd"] += float(rec.get("cost_usd") or 0)
        by = led["by_role"].setdefault(
            role, {"turns": 0, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
        )
        by["turns"] += 1
        by["seconds"] += float(rec.get("seconds") or 0)
        by["usd"] += float(rec.get("cost_usd") or 0)
        tu = (rec.get("token_usage") or {}).get("last") or {}
        by["input_tokens"] += int(tu.get("inputTokens") or rec.get("input_tokens") or 0)
        by["output_tokens"] += int(tu.get("outputTokens") or rec.get("output_tokens") or 0)
        atomic_write(path, json.dumps(led, indent=1, default=str))
        with open(self._p("state", "turns_%s.jsonl" % arm), "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "arm": arm,
                        "role": role,
                        "kind": kind,
                        **{
                            k: rec.get(k)
                            for k in (
                                "thread_id",
                                "turn_id",
                                "status",
                                "seconds",
                                "token_usage",
                                "requested",
                                "served",
                                "cost_usd",
                                "log",
                            )
                        },
                        "t": time.time(),
                    },
                    default=str,
                )
                + "\n"
            )
        if led["turns"] >= self.limits["turns"]:
            raise SystemExit("[ucoe-agent %s] turn cap %d reached; rerun later to resume" % (arm, self.limits["turns"]))

    def turn(self, arm, role, kind, cwd, text, log, thread_id=None, keep=False, served=None):
        """One turn in a fresh ephemeral thread (or in `thread_id`); the record is written to `log` and charged to the
        arm's ledger.
        A wedged server (timeout) is restarted once with a fresh thread; a second timeout is a HardStop."""
        srv = self.server(role)
        started = None
        if thread_id is None:
            try:
                started = srv.thread_start(cwd, ephemeral=True)
            except HardStop as e:
                if e.code != "transport_error":
                    raise
                audit(self.root, "thread_start_failed_restart", role=role, err=str(e)[:200])
                srv = self.restart_server(role)
                started = srv.thread_start(cwd, ephemeral=True)
            thread_id = started["thread"]["id"]
        r = srv.turn(thread_id, text, cwd)
        if r["status"] == "timeout":  # wedged server: restart, fresh thread, retry once
            audit(self.root, "turn_timeout_retry", role=role, kind=kind)
            srv = self.restart_server(role)
            started = srv.thread_start(cwd, ephemeral=True)
            thread_id = started["thread"]["id"]
            r = srv.turn(thread_id, text, cwd)
            r["retried_after_restart"] = True
        r.update(
            {
                "thread_id": thread_id,
                "kind": kind,
                "role": role,
                "requested": dict(self.roles[role]),
                "served": (started or {}).get("served") or served,
                "log": log,
                "backend": "codex",
            }
        )
        os.makedirs(os.path.dirname(log), exist_ok=True)
        atomic_write(log, json.dumps(r, indent=1, default=str))
        self.ledger(arm, role, kind, r)
        low = json.dumps(r.get("error") or "").lower() + (r.get("text") or "").lower()[-600:]
        if r["status"] == "failed" and any(p in low for p in LIMIT_PATTERNS):
            raise SystemExit(
                "[ucoe-agent] %s usage/rate limit (%s): %s; rerun later to resume (nothing sealed)"
                % (role, kind, str(r.get("error"))[:300])
            )
        if not keep:
            try:
                srv.thread_delete(thread_id)
            except Exception:  # noqa: BLE001
                pass
        if r["status"] == "timeout":
            raise HardStop("timeout", "%s turn %s timed out twice" % (role, kind))
        if r["status"] != "completed":
            raise HardStop(
                "transport_error",
                "%s turn %s ended with status %s: %s" % (role, kind, r["status"], str(r.get("error"))[:200]),
            )
        return r
