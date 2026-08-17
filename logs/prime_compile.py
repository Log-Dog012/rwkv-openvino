import time, sys, openvino as ov
IR = sys.argv[1]; DEV = sys.argv[2] if len(sys.argv) > 2 else "CPU"
def flag(m):
    try: open("logs/prime_progress.log","a").write(m + "\n")
    except Exception: pass
try:
    core = ov.Core()
    props = {}
    if DEV == "CPU":
        props[ov.properties.cache_dir] = "out/ov_cache"
        props[ov.properties.inference_num_threads] = 8
    if props: core.set_property(DEV, props)
    flag(f"START dev={DEV} {time.strftime('%H:%M:%S')}")
    t0 = time.time(); m = core.read_model(IR)
    flag(f"READ ok {time.time()-t0:.1f}s")
    t0 = time.time(); c = core.compile_model(m, DEV)
    flag(f"COMPILE ok dev={DEV} {time.time()-t0:.1f}s {time.strftime('%H:%M:%S')}")
    # 跑一次推理确认 compiled 可用
    import numpy as np
    ins = {i.any_name: i for i in c.inputs}
    req = c.create_infer_request()
    feed = {}
    for k in ins:
        if k == "idx": feed[k] = np.array([0], np.int64)
        elif k == "s_att_x" or k == "s_ffn": feed[k] = np.zeros((61, 4096), np.float16)
        elif k == "s_kv": feed[k] = np.zeros((61, 64, 64, 64), np.float16)
    req.infer(feed)
    flag("INFER ok")
except Exception:
    import traceback
    open("logs/prime_crash.log","w").write(traceback.format_exc())
    flag("CRASH")
