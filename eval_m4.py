import re, advisor_room as ar
GATE = 1.0
docs = ["How do I reserve a bare metal node?","How do I use object storage on Chameleon?",
 "What is a lease and how long can it last?","How do I connect to a node via SSH with a floating IP?",
 "How do I create a custom disk image?","How do I use Jupyter notebooks on Chameleon?",
 "What GPU hardware is available?","How do I set up a private network between nodes?"]
edge = [("capture photos with a pi camera every 10 minutes at the edge","edge-picamera-image"),
 ("read temperature and humidity from a sense hat on an edge device","edge_sensehat_image"),
 ("run cpu image classification inference on an edge device","edge-cpu-inference"),
 ("i need ssh access into a chi@edge container","edge_ssh_image")]
ar.get_room(); dp=0
print("== DOCS (should_fire must be False) ==")
for q in docs:
    fire=ar.should_fire(q,GATE); ok=not fire; dp+=ok
    print("%-4s fire=%s  %s"%("PASS" if ok else "FAIL", fire, q[:52]))
print("== EDGE (should_fire True + correct artifact) ==")
ep=0
for q,exp in edge:
    fire=ar.should_fire(q,GATE); r=ar.advise(q) if fire else ""
    gb=(re.search(r"grounded_by=\[([^\]]*)\]", r) or [None,""])[1] if r else ""
    prod=(re.search(r"produced_by=(\w+)", r) or [None,"-"])[1] if r else "-"
    ok=fire and (exp in gb); ep+=ok
    print("%-4s fire=%s by=%s expect=%s"%("PASS" if ok else "FAIL", fire, prod, exp))
print("SUMMARY docs=%d/8 edge=%d/4 -> %s"%(dp,ep,"EVAL_PASS" if (dp==8 and ep==4) else "EVAL_PARTIAL"))
