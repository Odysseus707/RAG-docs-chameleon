"""In-process 'advisor room' for the Chameleon docs assistant.

Thin wrapper around chi-edge-advisor so web_rag can (1) gate on whether a
question is edge-resource-shaped and (2) get a rendered recommendation to inject
as a context section. Fires only when ADVISOR_ENABLED is set.
"""
_ROOM = None

def _load():
    from advisor.artifacts.embeddings import get_embedder
    from advisor.artifacts.store import ArtifactStore
    from advisor.artifacts.router import RetrievalRouter
    from advisor.availability.base import get_backend
    from advisor.inventory.catalog import InventoryCache
    from advisor.reason.reasoner import Reasoner
    emb = get_embedder()
    assert emb.__class__.__name__ == "BgeEmbedder", "advisor not on bge path: %s" % emb.name
    store = ArtifactStore(embedder=emb).load()
    return {"store": store, "router": RetrievalRouter(store),
            "backend": get_backend("reference_api"),
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
    return _render(rec)

def _render(rec) -> str:
    return "\n".join([
        "A CHI@Edge resource recommendation for this workload "
        "(grounded_by=%s, produced_by=%s):" % (rec.grounded_by, rec.produced_by),
        "- machine_type: %s%s" % (rec.machine_type, (" [%s]" % rec.device_name) if rec.device_name else ""),
        "- architecture: %s   gpu: %s" % (rec.architecture, rec.gpu),
        "- image: %s" % rec.image,
        "- device_profiles: %s" % (rec.device_profiles or []),
        "- lease: count=%s duration_hours=%s platform_version=%s runtime=%s exposed_ports=%s" % (
            rec.count, rec.duration_hours, rec.platform_version, rec.runtime, rec.exposed_ports),
        "- reasoning: %s" % (rec.reasoning or "").strip(),
    ])


_EDGE_CUES = ("edge", "chi@edge", "chi edge", "container", "raspberry", "picamera",
              "pi camera", "sense hat", "sensehat", "sensor", "camera", "gpio",
              "jetson", "on-device", "iot", "arm64", "device profile")

def should_fire(question: str, gate: float) -> bool:
    """Fire only for CHI@Edge resource questions: an edge cue AND tag score >= gate.
    The cue guard prevents general SSH/networking questions (which share ssh/
    floating-ip vocabulary with the edge_ssh artifact) from spuriously firing.
    """
    if not any(c in question.lower() for c in _EDGE_CUES):
        return False
    return classify_top(question) >= gate
