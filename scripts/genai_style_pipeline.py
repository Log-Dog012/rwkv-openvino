#!/usr/bin/env python3
"""GenAI 风格的自写 pipeline：用分块执行器驱动我们的 IR + openvino_tokenizers IR 编解码。

目标（易用性补齐的「自写驱动」路线）:
- 接口对标 openvino_genai.LLMPipeline：generate(prompt, max_new_tokens) -> str
- tokenizer 层：openvino_tokenizers 转出的 openvino_tokenizer.xml/detokenizer.xml
  （GenAI 兼容的 tokenizer IR，编解码用 ov::InferRequest 驱动）
- 模型层：本仓库分块执行器（chunk IR + 状态张量跨 chunk 推进）——绕开单图 FFN 退化
- 不需要 GenAI 的 StatefulLLMPipeline（它按 transformer KV cache 约定驱动，RNN 不匹配）

用法:
  python scripts/genai_style_pipeline.py <gguf> --ir-dir out/chunks_13.3b \
      --tokenizer out/genai_test3/openvino_tokenizer.xml \
      --detokenizer out/genai_test3/openvino_detokenizer.xml \
      --prompt "The Eiffel Tower is located in the city of" --n 8 --device CPU
"""
import argparse, glob, os, sys, time
import numpy as np
import openvino as ov
import openvino_tokenizers  # 注册 TrieTokenizer 等 tokenizer 自定义 op 扩展（compile tokenizer IR 必需）

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv7_ov import _dequant_np
from rwkv7_ov_layerwise import build_chunk


class RWKVTokenizer:
    """openvino_tokenizers IR 驱动的 encode/decode（GenAI 兼容 tokenizer 层）。"""

    def __init__(self, tok_xml, detok_xml, device="CPU"):
        core = ov.Core()
        self._tok = core.compile_model(tok_xml, device)
        self._detok = core.compile_model(detok_xml, device)

    def encode(self, text: str):
        out = self._tok([text])["input_ids"]
        return np.asarray(out).reshape(-1).tolist()

    def decode(self, ids) -> str:
        arr = np.asarray(ids, np.int64).reshape(1, -1)
        return str(self._detok(arr)["string_output"][0])


class RWKVPipeline:
    """GenAI 风格接口：generate(prompt, max_new_tokens) -> str（分块执行器驱动）。"""

    def __init__(self, gguf_path, ir_dir, tok_xml, detok_xml, device="CPU", threads=8, chunk=8):
        self.device = device
        self.chunk = chunk
        self._ir_dir = ir_dir
        t0 = time.time()

        # GGUF 超参 + emb_ln（复用 rwkv7_ov 的 ln0 预归一）
        import gguf
        r = gguf.GGUFReader(gguf_path)
        T = {t.name: t for t in r.tensors}
        emb_raw = _dequant_np(T["token_embd.weight"])
        V, C = emb_raw.shape
        L = 1 + max(int(n.split(".")[1]) for n in T if n.startswith("blk."))
        H, N = C // 64, 64
        ln0_w = _dequant_np(T["token_embd_norm.weight"]).reshape(1, -1)
        ln0_b = _dequant_np(T["token_embd_norm.bias"]).reshape(1, -1)
        self.emb_ln = np.ascontiguousarray(
            ((emb_raw - emb_raw.mean(-1, keepdims=True)) / np.sqrt(emb_raw.var(-1, keepdims=True) + 1e-5)
             * ln0_w + ln0_b).astype(np.float16))
        del emb_raw
        self.r, self.T = r, T
        self.V, self.C, self.L, self.H, self.N = V, C, L, H, N

        # 状态张量 [L,C]/[L,H,N,N]/[L,C]
        self.states = [np.zeros((L, C), np.float16),
                       np.zeros((L, H, N, N), np.float16),
                       np.zeros((L, C), np.float16)]
        self.chunks = [(i, min(i + chunk, L)) for i in range(0, L, chunk)]
        self._comps = {}

        # 编译缓存
        self.core = ov.Core()
        try:
            self.core.set_property("CPU", {ov.properties.inference_num_threads: threads,
                                           ov.properties.cache_dir: "out/ov_cache"})
        except Exception:
            pass
        self.tokenizer = RWKVTokenizer(tok_xml, detok_xml, device)
        print(f"[pipeline] ready V={V} C={C} L={L} chunks={len(self.chunks)} in {time.time()-t0:.1f}s", flush=True)

    def _comp(self, ci):
        """chunk ci 的 compiled model（缓存常驻，不重编译——单图退化的绕开方案）。"""
        if ci not in self._comps:
            lo, hi = self.chunks[ci]
            fs = glob.glob(f"{self._ir_dir or ''}/chunk{lo}_*.xml")
            m = self.core.read_model(fs[0])
            self._comps[ci] = self.core.compile_model(m, self.device)
        return self._comps[ci]

    # ---- 推理核心（复用分块执行器语义）----
    def _run_chunk(self, comp, ci, tok, x_in, vf_in):
        lo, hi = self.chunks[ci]
        feed = {}
        if lo == 0:
            feed["idx"] = np.array([tok], np.int64)
            feed["emb_table"] = self.emb_ln
        else:
            feed["x_in"] = np.ascontiguousarray(np.asarray(x_in, np.float16).reshape(1, self.C))
            feed["v_first"] = np.ascontiguousarray(np.asarray(vf_in, np.float16).reshape(1, self.C))
        feed["s_att"] = self.states[0][lo:hi]
        feed["s_kv"] = self.states[1][lo:hi]
        feed["s_ffn"] = self.states[2][lo:hi]
        req = comp.create_infer_request()
        req.infer(feed)
        off = 1 if hi == self.L else 0
        self.states[0][lo:hi] = np.array(req.get_output_tensor(off).data)
        self.states[1][lo:hi] = np.array(req.get_output_tensor(off + 1).data)
        self.states[2][lo:hi] = np.array(req.get_output_tensor(off + 2).data)
        x_out = np.array(req.get_output_tensor(off + 3).data)
        vf_out = np.array(req.get_output_tensor(off + 4).data)
        lg = np.array(req.get_output_tensor(0).data) if off == 1 else None
        return x_out, vf_out, lg

    def _full_pass(self, tok_id):
        """一个 token 过全部 chunk（链式 x_in/vf_in 传递，states 持久累积）。
        返回最后一层（含 head）的 logits。"""
        x_in, vf_in = None, None
        last = None
        for ci in range(len(self.chunks)):
            comp = self._comp(ci)
            tokid = tok_id if ci == 0 else None
            x_out, vf_out, lg = self._run_chunk(comp, ci, tokid, x_in, vf_in)
            x_in, vf_in = x_out, vf_out  # 链式：上一 chunk 输出作为下一 chunk 输入
            if lg is not None:
                last = lg
        return last

    # ---- GenAI 风格接口 ----
    def generate(self, prompt: str, max_new_tokens: int = 8, verbose=True) -> str:
        t0 = time.time()
        ids = self.tokenizer.encode(prompt)
        if verbose:
            print(f"[pipeline] prompt {len(ids)} tok: {prompt!r}", flush=True)

        # prompt sweep：逐 token 过全 chunk（states 持久累积）
        last = None
        for i, tid in enumerate(ids):
            last = self._full_pass(tid)
            if verbose:
                print(f"[pipeline] sweep tok {i+1}/{len(ids)} done", flush=True)
        if verbose:
            print(f"[pipeline] prompt sweep done in {time.time()-t0:.1f}s", flush=True)

        # 生成：逐 token 过全 chunk
        gen = []
        for g in range(max_new_tokens):
            t = int(np.argmax(last[0]))
            gen.append(t)
            last = self._full_pass(t)
            if verbose:
                print(f"[pipeline] gen {g+1}/{max_new_tokens}: {self.tokenizer.decode(gen)!r}", flush=True)
        text = self.tokenizer.decode(gen)
        print(f"[pipeline] === generated {max_new_tokens} tokens in {time.time()-t0:.1f}s ===", flush=True)
        print(f"[pipeline] gen text: {text!r}", flush=True)
        print(f"[pipeline] gen ids: {gen}", flush=True)
        return text


def main():
    A = argparse.ArgumentParser()
    A.add_argument("gguf")
    A.add_argument("--ir-dir", default="out/chunks_13.3b")
    A.add_argument("--tokenizer", default="out/genai_test3/openvino_tokenizer.xml")
    A.add_argument("--detokenizer", default="out/genai_test3/openvino_detokenizer.xml")
    A.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    A.add_argument("--n", type=int, default=8)
    A.add_argument("--device", default="CPU")
    A.add_argument("--threads", type=int, default=8)
    args = A.parse_args()
    pipe = RWKVPipeline(args.gguf, args.ir_dir, args.tokenizer, args.detokenizer,
                        device=args.device, threads=args.threads)
    pipe.generate(args.prompt, max_new_tokens=args.n)


if __name__ == "__main__":
    main()
