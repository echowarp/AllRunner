"""Work-conserving spill rule of llmt-router (no network)."""
import importlib.util, time
import os as _os
sp = importlib.util.spec_from_file_location("r", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "llmt-router.py"))
r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
import tempfile as _tf
r.DRAIN_DIR = _tf.mkdtemp()   # hermetic: a drain file of a real router on this host must not count
def mk():
    a = r.Backend({"name": "arc", "url": "http://127.0.0.1:1", "cap": 16, "prefill": 2800, "dec": [83, 74, 63, 57]})
    g = r.Backend({"name": "rtx5090", "url": "http://127.0.0.1:2", "cap": 6, "prefill": 5000, "dec": [231, 203, 139, 137], "prefer_until": 2})
    for b in (a, g): b.healthy = True
    R = r.Router([a, g]); R.spill_kv = 0.9; return R, a, g
chain = [(b"x", 100), (b"y", 20000)]
now = time.monotonic()
R, a, g = mk(); g.eng_t = a.eng_t = now
order, reason, _ = R.choose(chain); assert order[0] is g and reason != "spill", (order[0].name, reason); print("ok   idle: 5090 first")
g.eng_waiting = 1; order, reason, _ = R.choose(chain); assert order[0] is a and reason == "spill", (order[0].name, reason); print("ok   5090 queue -> spill to arc")
g.eng_waiting = 0; g.eng_kv = 0.95; order, reason, _ = R.choose(chain); assert order[0] is a and reason == "spill"; print("ok   5090 KV 95% -> spill to arc")
a.eng_waiting = 2; order, reason, _ = R.choose(chain); assert order[0] is g, order[0].name; print("ok   both saturated -> cost model order")
a.eng_waiting = 0; g.eng_t = now - 10; order, reason, _ = R.choose(chain); assert reason != "spill"; print("ok   stale engine numbers ignored")
R.spill_kv = 0.0; g.eng_t = time.monotonic(); order, reason, _ = R.choose(chain); assert reason != "spill"; print("ok   spill_kv 0 = off")
print("ALL PASS")
