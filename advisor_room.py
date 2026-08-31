"""In-process 'advisor room' for the Chameleon docs assistant.

Thin wrapper around chi-edge-advisor so web_rag can (1) gate on whether a
question is edge-resource-shaped and (2) get a rendered recommendation to inject
as a context section. Fires only when ADVISOR_ENABLED is set.
"""
_ROOM = None

def _load():
    import os
    from advisor.artifacts.embeddings import get_embedder
    from advisor.artifacts.store import ArtifactStore
    from advisor.artifacts.router import RetrievalRouter
    from advisor.availability.base import get_backend
    from advisor.inventory.catalog import InventoryCache
    from advisor.reason.reasoner import Reasoner
    emb = get_embedder()
    assert emb.__class__.__name__ == "BgeEmbedder", "advisor not on bge path: %s" % emb.name
    store = ArtifactStore(embedder=emb).load()
    # "blazar" needs CHAMELEON_RC_GLOB pointing at openrc files and reports real
    # reservation state; "reference_api" has no CHI@Edge inventory at all and is
    # kept as the default so benchmark collection is unchanged.
    backend = os.environ.get("ADVISOR_AVAILABILITY", "reference_api")
    return {"store": store, "router": RetrievalRouter(store),
            "backend": get_backend(backend),
            "inventory": InventoryCache().load(), "reasoner": Reasoner()}

def get_room():
    global _ROOM
    if _ROOM is None:
        _ROOM = _load()
    return _ROOM

def classify_top(question: str) -> float:
    scores = get_room()["router"].classify(question)
    return max(scores.values()) if scores else 0.0

def advise(question: str) -> str:
    r = get_room()
    retrieval = r["router"].route(question)
    try:
        availability = r["backend"].list_devices()
    except Exception:
        availability = []
    rec = r["reasoner"].recommend(question, availability, r["inventory"], retrieval)
    return _render(rec, availability)


def _device_line(rec, availability) -> str:
    """State the named device and how long it is actually free.

    Blazar exposes available_hours / free_until_utc computed from allocations.
    When a device is free and those are None it means nothing is booked behind
    it, so the honest phrasing is "no reservation queued" rather than inventing
    a number or silently omitting the window.
    """
    dev = next((d for d in (availability or []) if d.device_uid == rec.device_name), None)
    if dev is None:
        return "- device: none confirmed free at request time"
    if dev.available_hours is not None:
        window = "free for the next %.1f h" % dev.available_hours
    elif dev.free_now:
        window = "free now, no reservation queued behind it"
    else:
        window = "NOT free - do not reserve this one"
    return "- device: %s (%s, checked live against Blazar)" % (dev.device_uid, window)


def _render(rec, availability=()) -> str:
    return "\n".join([
        "A CHI@Edge resource recommendation for this workload "
        "(grounded_by=%s, produced_by=%s):" % (rec.grounded_by, rec.produced_by),
        "- machine_type: %s%s" % (rec.machine_type, (" [%s]" % rec.device_name) if rec.device_name else ""),
        _device_line(rec, availability),
        "- architecture: %s   gpu: %s" % (rec.architecture, rec.gpu),
        "- image: %s" % rec.image,
        "- device_profiles: %s" % (rec.device_profiles or []),
        "- lease: count=%s duration_hours=%s platform_version=%s runtime=%s exposed_ports=%s" % (
            rec.count, rec.duration_hours, rec.platform_version, rec.runtime, rec.exposed_ports),
        "- reasoning: %s" % (rec.reasoning or "").strip(),
        "",
        _render_code(rec),
        _further_reading(rec),
    ])


# Verified Trovi share pages. Only artifacts we have actually confirmed a share
# URL for appear here; the rest fall back to their source repo, because guessing
# a share id would send users to a page that does not exist.
_TROVI = {
    "serve-edge-chi":
        "https://chameleoncloud.org/experiment/share/a1662022-9017-45b1-9b96-31705ca20358",
}


def _further_reading(rec) -> str:
    """Point the user at the artifacts this recommendation actually came from."""
    from advisor.artifacts.registry import ARTIFACTS_BY_ID

    lines = []
    for aid in (rec.grounded_by or []):
        meta = ARTIFACTS_BY_ID.get(aid)
        if not meta:
            continue
        if aid in _TROVI:
            lines.append("- %s - Trovi: %s" % (meta.title, _TROVI[aid]))
        elif getattr(meta, "repo", ""):
            repo = meta.repo if meta.repo.startswith("http") else "https://" + meta.repo
            lines.append("- %s - source: %s" % (meta.title, repo))
    if not lines:
        return ""
    return "\n".join(
        ["", "Further reading - the artifacts this recommendation is grounded in:"]
        + lines)


def _render_code(rec) -> str:
    """Ready-to-run python-chi for this recommendation.

    Every call here appears in a grounding artifact - lease_duration,
    add_device_reservation, create_lease, get_device_reservation and
    create_container in A1, device_profiles in A2/A3, runtime in A4 - so the
    snippet is assembled from documented usage rather than invented. Blazar
    reserves by machine type, not by individual device, which is why
    device_name only appears as a comment.
    """
    kwargs = [
        '    image="%s",' % rec.image,
        "    exposed_ports=%s," % (list(rec.exposed_ports) if rec.exposed_ports else []),
        "    reservation_id=lease.get_device_reservation(lease_id),",
        "    platform_version=%s," % rec.platform_version,
    ]
    if rec.device_profiles:
        kwargs.append("    device_profiles=%s," % list(rec.device_profiles))
    if rec.runtime:
        kwargs.append('    runtime="%s",' % rec.runtime)

    free_note = ("# %s is free right now; Blazar reserves by type, so the lease "
                 "asks for\n# machine_name and you may land on any free device of "
                 "that type.\n" % rec.device_name) if rec.device_name else ""

    return "\n".join([
        "```python",
        free_note + "import chi",
        "from chi import container, lease",
        "",
        'chi.use_site("CHI@Edge")',
        'chi.set("project_name", "<your project name>")',
        "",
        'machine_name = "%s"' % rec.machine_type,
        "start, end = lease.lease_duration(hours=%g)" % rec.duration_hours,
        "",
        "reservations = []",
        "lease.add_device_reservation(reservations, count=%s, machine_name=machine_name)"
        % rec.count,
        'container_lease = lease.create_lease(f"advisor-{machine_name}", reservations)',
        'lease_id = container_lease["id"]',
        "lease.wait_for_active(lease_id)",
        "",
        "my_container = container.create_container(",
        '    f"advisor-{machine_name}".replace("_", "-"),',
        *kwargs,
        ")",
        "```",
    ])


# Matched as substrings, so stems are used wherever a word inflects:
# "raspberr" covers raspberry/raspberries, "device_profile" covers the plural.
_EDGE_CUES = ("edge", "chi@edge", "chi edge", "container", "raspberr", "picamera",
              "pi camera", "sense hat", "sensehat", "sensor", "camera", "gpio",
              "jetson", "coral", "xavier", "orin", "tpu", "on-device", "iot",
              "arm64", "device profile", "device_profile")


def has_cue(question: str) -> bool:
    """Cheap prefilter: does the question mention edge hardware at all?"""
    return any(c in question.lower() for c in _EDGE_CUES)


_JUDGE_PROMPT = """You decide whether a question is about obtaining or using edge \
computing hardware on Chameleon's CHI@Edge testbed.

CHI@Edge provides small edge devices: Raspberry Pi 4 and 5, NVIDIA Jetson (Nano, \
Xavier NX, Orin), and Google Coral Edge TPU. Typical work: cameras and computer \
vision, sensors, GPIO, on-device inference, containers on ARM.

Answer YES if the question is about choosing, reserving, configuring or using that \
kind of hardware - including when the user describes only a use case ("I want to \
work with cameras") without naming any device.

Answer NO for bare-metal Chameleon nodes, FPGAs, general networking, SSH keys, \
account or allocation questions, or anything CHI@Edge devices do not provide.

Reply with exactly one word: YES, NO, or UNSURE if it genuinely could be either.

Question: %s"""


def judge(question: str, prompt: str = None) -> "str | None":
    """Ask the LLM a YES/NO question about the user's question.

    ``prompt`` defaults to the edge-intent check; pass _AVAIL_PROMPT to ask the
    availability-intent question instead. Returns None when the model cannot be
    reached, so callers fall back rather than silently answering "no".
    """
    import json
    import os
    import urllib.request

    # Resolve the same way rag.py does: LLM_* first, TEJAS_* as fallback, then the
    # real gateway. Deliberately NO key default - the old "ollama" placeholder made
    # a missing key 401 silently, and judge() returning None degrades to keyword
    # matching with nothing in the UI to say so.
    base = (os.environ.get("LLM_API_BASE") or os.environ.get("TEJAS_BASE_URL")
            or "https://ai.tejas.tacc.utexas.edu/v1").rstrip("/")
    model = (os.environ.get("LLM_MODEL") or os.environ.get("TEJAS_MODEL")
             or "Meta-Llama-3.3-70B-Instruct")
    key = os.environ.get("LLM_API_KEY") or os.environ.get("TEJAS_API_KEY") or ""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": (prompt or _JUDGE_PROMPT) % question}],
        "temperature": 0,
        "max_tokens": 5,
    }).encode()
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            answer = json.load(resp)["choices"][0]["message"]["content"]
    except Exception as err:
        print("advisor judge unreachable:", err)
        return None
    reply = answer.strip().upper()
    for verdict in ("UNSURE", "YES", "NO"):
        if reply.startswith(verdict):
            return verdict.lower()
    return "unsure"


def _judge_enabled() -> bool:
    import os
    return os.environ.get("ADVISOR_JUDGE", "false").lower() in {"1", "true", "yes", "on"}


def explain(question: str, gate: float) -> "tuple[str, str]":
    """Decide 'fire' | 'offer' | 'skip', with the reason.

    A single threshold has to trade off missing edge questions against firing on
    unrelated ones, so there are two. Above `gate` the signal is strong enough to
    act on unasked. Below ADVISOR_GATE_LOW there is nothing worth raising. In
    between - or when the judge says it genuinely cannot tell - the user is asked
    instead of being guessed at.
    """
    import os

    gate_low = float(os.environ.get("ADVISOR_GATE_LOW", "0.25"))
    cue = has_cue(question)
    score = classify_top(question)

    # Deterministic path, unchanged: this is what the benchmark measured.
    if cue and score >= gate:
        return "fire", "tag score %.2f >= gate %.2f" % (score, gate)

    verdict = judge(question) if _judge_enabled() else None
    if verdict == "yes":
        return "fire", "LLM judge: edge-related (keyword path missed it)"
    if verdict == "no":
        return "skip", "LLM judge: not edge-related"
    if verdict == "unsure":
        return "offer", "LLM judge: unsure, asking you"

    # Judge off or unreachable: fall back to the numeric band.
    if cue or score >= gate_low:
        return "offer", ("weak signal (tag score %.2f, cue %s), asking you"
                         % (score, "yes" if cue else "no"))
    return "skip", "no edge signal (tag score %.2f, no cue)" % score


def should_fire(question: str, gate: float) -> bool:
    """Fire for CHI@Edge resource questions: an edge cue AND tag score >= gate,
    or - when ADVISOR_JUDGE is set - an LLM intent check that catches use-case
    phrasing the keyword path misses ("I want to work with cameras").

    The cue guard prevents general SSH/networking questions (which share ssh/
    floating-ip vocabulary with the edge_ssh artifact) from spuriously firing.
    ADVISOR_JUDGE defaults off, so the benchmarked behaviour is unchanged.
    """
    return explain(question, gate)[0] == "fire"


_AVAIL_PROMPT = """Does this question ask what hardware is currently available, \
free, busy or reservable on Chameleon's CHI@Edge testbed?

Answer YES for questions about current status or inventory - "what devices are \
available right now", "are any Jetsons free", "what can I reserve today", "show me \
what is up".

Answer NO when the question only asks what to reserve for a workload, or how to \
configure or use something, without asking about current state.

Reply with exactly one word: YES or NO.

Question: %s"""


def wants_availability(question: str) -> bool:
    """Is this a 'what is available right now' question?

    Falls back to a keyword test when the judge is disabled or unreachable, so
    the capability still works without an LLM.
    """
    if _judge_enabled():
        verdict = judge(question, _AVAIL_PROMPT)
        if verdict in ("yes", "no"):
            return verdict == "yes"
    q = question.lower()
    return any(w in q for w in ("available", "availability", "free right now",
                                "what is up", "status", "in use", "busy"))


def list_availability() -> str:
    """Render current device availability per machine type.

    free_now is None => the backend does not know; those are counted as unknown
    rather than free, because offering an unknown device as free is the failure
    mode this whole advisor exists to avoid.
    """
    from datetime import datetime, timezone

    backend = get_room()["backend"]
    try:
        devices = backend.list_devices()
    except Exception as err:
        return "Availability lookup failed (backend=%s): %s" % (backend.name, err)
    if not devices:
        return ("Backend '%s' returned no devices, so current availability is "
                "unknown." % backend.name)

    rows = {}
    for d in devices:
        free, down, unknown, total = rows.get(d.machine_type, (0, 0, 0, 0))
        rows[d.machine_type] = (
            free + (1 if d.free_now is True else 0),
            down + (1 if d.reservable is False else 0),
            unknown + (1 if d.free_now is None else 0),
            total + 1,
        )

    provenance = ("live reservation state" if backend.reports_live_state
                  else "static inventory - NO live state")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = ["CHI@Edge device availability (%s, backend=%s, read %s):"
           % (provenance, backend.name, stamp),
           "%-32s %6s %6s %8s %6s" % ("device_type", "free", "down", "unknown", "total")]
    for mt, (free, down, unknown, total) in sorted(rows.items(),
                                                   key=lambda kv: -kv[1][3]):
        out.append("%-32s %6d %6d %8d %6d" % (mt, free, down, unknown, total))
    out.append("%-32s %6d %6d %8d %6d" % (
        "TOTAL",
        sum(v[0] for v in rows.values()), sum(v[1] for v in rows.values()),
        sum(v[2] for v in rows.values()), sum(v[3] for v in rows.values())))
    return "\n".join(out)
