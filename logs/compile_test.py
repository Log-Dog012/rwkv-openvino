import time, sys, openvino as ov
IR = sys.argv[1]
DEV = sys.argv[2] if len(sys.argv) > 2 else "CPU"
def flag(m):
    try: open("logs/compile_progress.log","a").write(m + "\n")
    except Exception: pass
try:
    core = ov.Core()
    flag(f"START dev={DEV} {time.strftime('%H:%M:%S')}")
    t0 = time.time(); m = core.read_model(IR)
    flag(f"READ ok {time.time()-t0:.1f}s")
    t0 = time.time(); c = core.compile_model(m, DEV)
    flag(f"COMPILE ok dev={DEV} {time.time()-t0:.1f}s")
except Exception:
    import traceback
    open("logs/compile_crash.log","w").write(traceback.format_exc())
    flag("CRASH")
