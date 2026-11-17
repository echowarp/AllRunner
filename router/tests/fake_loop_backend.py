"""Manual loop-stop harness, part 1: a fake OpenAI SSE backend (synthetic text only, no GPU).
/v1/chat/completions streams a repeating sentence after 300 events if the messages contain 'loop', else varied
random words (1,500 events). Also answers /tokenize and /health. Appends one JSON line per request to LOG: whether the
client (the router) disconnected early.
  python3 fake_loop_backend.py PORT LOG"""
import asyncio, json, random, sys, time
PORT = int(sys.argv[1]); LOG = sys.argv[2]
async def handle(r, w):
    head = await r.readuntil(b"\r\n\r\n")
    hl = head.decode().split("\r\n"); path = hl[0].split()[1]
    n = next((int(x.split(":")[1]) for x in hl if x.lower().startswith("content-length")), 0)
    body = json.loads((await r.readexactly(n)) or b"{}") if n else {}
    if not path.startswith("/v1/chat"):
        out = json.dumps({"count": 10, "tokens": [1] * 10, "max_model_len": 196608}).encode()
        w.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(out), out)); await w.drain(); w.close(); return
    loop = "loop" in json.dumps(body.get("messages"))
    w.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
    rnd = random.Random(1); sent = 0; aborted = False
    vocab = [f"w{i}" for i in range(5000)]
    try:
        for i in range(20000):
            if loop and i > 300:
                txt = "the same sentence repeats again and again forever "
            else:
                txt = " ".join(rnd.choice(vocab) for _ in range(8)) + " "
            ev = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "m",
                  "choices": [{"index": 0, "delta": {"content": txt}, "finish_reason": None}]}
            d = b"data: " + json.dumps(ev).encode() + b"\n\n"
            w.write(b"%x\r\n%s\r\n" % (len(d), d)); await w.drain(); sent += 1
            await asyncio.sleep(0.0005)
            if not loop and i >= 1500: break
        fin = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "m",
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": sent}}
        d = b"data: " + json.dumps(fin).encode() + b"\n\ndata: [DONE]\n\n"
        w.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(d), d)); await w.drain()
    except (ConnectionError, OSError):
        aborted = True
    open(LOG, "a").write(json.dumps({"loop": loop, "sent_events": sent, "client_disconnected": aborted}) + "\n")
    w.close()
async def main():
    s = await asyncio.start_server(handle, "127.0.0.1", PORT); await s.serve_forever()
asyncio.run(main())
