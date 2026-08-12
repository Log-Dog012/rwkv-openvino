#!/usr/bin/env python3
"""GGUF 头部解析器：dump 张量名 -> 形状 -> ggml 量化类型（混合精度证据）。

用法: python3 scripts/gguf_inspect.py models/xxx.gguf [--tensor-filter rwkv.blocks.0]
"""
import argparse, struct

# ggml 类型枚举（GGUF v3 常用）
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "IQ1_M", 26: "IQ4_K", 27: "BF16",
}
VAL_TYPES = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",8:"str",9:"array",10:"u64",11:"i64",12:"f64"}

def read_str(buf, off):
    n = struct.unpack_from("<Q", buf, off)[0]; off += 8
    return buf[off:off+n].decode("utf-8", "replace"), off+n

def read_val(buf, off, vt):
    if vt == 8:
        val, off = read_str(buf, off)
    elif vt == 6:
        val = struct.unpack_from("<f", buf, off)[0]; off += 4
    elif vt == 10:
        val = struct.unpack_from("<Q", buf, off)[0]; off += 8
    elif vt == 11:
        val = struct.unpack_from("<q", buf, off)[0]; off += 8
    elif vt == 12:
        val = struct.unpack_from("<d", buf, off)[0]; off += 8
    elif vt == 0:
        val = buf[off]; off += 1
    elif vt == 1:
        val = struct.unpack_from("<b", buf, off)[0]; off += 1
    elif vt == 2:
        val = struct.unpack_from("<H", buf, off)[0]; off += 2
    elif vt == 3:
        val = struct.unpack_from("<h", buf, off)[0]; off += 2
    elif vt == 4:
        val = struct.unpack_from("<I", buf, off)[0]; off += 4
    elif vt == 5:
        val = struct.unpack_from("<i", buf, off)[0]; off += 4
    elif vt == 7:
        val = bool(buf[off]); off += 1
    elif vt == 9:  # array
        et = struct.unpack_from("<I", buf, off)[0]; off += 4
        cnt = struct.unpack_from("<Q", buf, off)[0]; off += 8
        items = []
        for _ in range(cnt):
            it, off = read_val(buf, off, et)
            items.append(it)
        val = (et, cnt, items)
    else:
        val = "?"
    return val, off

def read_kv(buf, off):
    key, off = read_str(buf, off)
    vtype = struct.unpack_from("<I", buf, off)[0]; off += 4
    val, off = read_val(buf, off, vtype)
    return key, vtype, val, off

def main():
    A = argparse.ArgumentParser()
    A.add_argument("gguf")
    A.add_argument("--tensor-filter", default="")
    A = A.parse_args()
    import mmap
    fh = open(A.gguf, "rb")
    buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)  # 全文件内存映射，不占 RSS

    assert buf[:4] == b"GGUF", "not a GGUF"
    ver = struct.unpack_from("<I", buf, 4)[0]
    n_tensors = struct.unpack_from("<Q", buf, 8)[0]
    n_kv = struct.unpack_from("<Q", buf, 16)[0]
    print(f"GGUF v{ver}: tensors={n_tensors} metadata_kv={n_kv}")

    off = 24
    for i in range(n_kv):
        key, vt, val, off = read_kv(buf, off)
        if key in ("general.architecture", "general.name", "rwkv.embedding_length",
                   "rwkv.block_count", "rwkv.head_count", "rwkv.attention.head_size"):
            print(f"  meta[{i}] {key} = {val}")
        elif key.startswith("tokenizer"):
            pass
    # 跳过其余 metadata 到 tensor info
    # 重新读 KV 全量以正确定位 tensor_info 起点
    # （上面已读完所有 KV，off 现在指向 tensor_info_count）
    # 注：上面读了全部 KV（含 tokenizer），off 已就位
    # GGUF 无独立 tensor_info_count：数量 = 头部 tensor_count
    n_tinfo = n_tensors
    print(f"tensor_infos={n_tinfo} (info starts at off={off})")

    from collections import Counter
    cnt = Counter()
    tensors = []
    for i in range(n_tinfo):
        name, off = read_str(buf, off)
        nd = struct.unpack_from("<I", buf, off)[0]; off += 4
        dims = list(struct.unpack_from("<" + "Q"*nd, buf, off)); off += 8*nd  # GGUF v3 dims 是 u64
        gtype = struct.unpack_from("<I", buf, off)[0]; off += 4
        toff = struct.unpack_from("<Q", buf, off)[0]; off += 8
        tensors.append((name, dims, gtype))
        cnt[GGML_TYPES.get(gtype, f"T{gtype}")] += 1
        if not A.tensor_filter or A.tensor_filter in name:
            print(f"  {name} {dims} {GGML_TYPES.get(gtype, 'T'+str(gtype))} @{toff}")
    print("=== dtype 分布（全模型）===")
    for k, v in cnt.most_common():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
