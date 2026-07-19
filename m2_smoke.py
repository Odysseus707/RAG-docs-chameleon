import time
from rag import load_vectorstore, load_parents, create_llm_chain, build_context
vs = load_vectorstore(); parents = load_parents(); chain = create_llm_chain()
qs = [
 "How do I reserve a bare metal node?",
 "How do I use object storage on Chameleon?",
 "What GPU hardware is available?",
 "How do I create a custom disk image?",
 "How do I connect to a node with a floating IP over SSH?",
]
for q in qs:
    t=time.time()
    srcs, ctx, _ = build_context(q, vs, parents)
    ans = chain.invoke({"question": q, "context": ctx, "history": []}).content
    print("\n=== Q: %s  (%.1fs, %d sources) ===" % (q, time.time()-t, len(srcs)))
    print((ans or "").strip()[:420])
    print("SOURCES:", srcs[:3])
