from __future__ import annotations
import os, re, csv, json, uuid, tempfile, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from Bio import Entrez
from langchain_openai import ChatOpenAI
import openai
from agent.token_meter import METER   # OpenAI 使用量メータ
from queue import Queue, Empty
import logging, textwrap

DEBUG_PROGRESS = os.getenv("DEBUG_PROGRESS", "0") == "1"
logging.basicConfig(
    level=logging.DEBUG if DEBUG_PROGRESS else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ───── 0. 初期化 ────────────────────────────────────
Entrez.email = "atmiyashita@gmail.com"
Entrez.api_key = os.getenv("NCBI_API_KEY") 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
openai_api_client = openai.OpenAI(api_key=OPENAI_API_KEY)
llm_small = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4.1-mini")

# ───── 1. 定数／ユーティリティ ──────────────────────
DETAIL_HEADERS = [
    "sent_idx", "phase", "sentence",
    "keyword", "query",
    "pmid", "title",
    "score", "reason",
    "accepted",
    "explanation"
]

def push(progress, obj):
    """progress があれば JSON 文字列化して渡す"""
    if progress:
        progress(json.dumps(obj, ensure_ascii=False))

def split_sentences(t: str) -> List[str]:
    ABBR = ["e.g.","i.e.","etc.","spp.","sp.","gen.","cf.","subsp.","str.",
            "P.","E.","S.","B.","C.","D.","M.","N.","R.","St.","T.","V.","vs."]
    for a in ABBR:
        t = t.replace(a, a.replace(".", "<DOT>"))
    sents = re.split(r"(?<=[。！？!?\.])\s+", t)
    return [s.replace("<DOT>", ".").strip() for s in sents if s.strip()]

def endnote_tag(p):
    first = p["author"].split(";")[0].split(",")[0] if p["author"] else "Anon"
    return f"{{{first}, {p.get('year', 'XXXX')} #{p['pmid']}}}"

def to_ris(p, n):
    ris = ["TY  - JOUR", f"TI  - {p['title']}"] + [
        f"AU  - {au.strip()}" for au in p["author"].split(";") if au.strip()
    ]
    if p.get("journal"):  ris.append(f"JF  - {p['journal']}")
    if p.get("year"):     ris.append(f"PY  - {p['year']}")
    if p.get("volume"):   ris.append(f"VL  - {p['volume']}")
    if p.get("issue"):    ris.append(f"IS  - {p['issue']}")
    if p.get("pages"):    ris.append(f"SP  - {p['pages']}")
    if p.get("doi"):      ris.append(f"DO  - {p['doi']}")
    ris += [f"UR  - {p['url']}",
            f"AB  - {p['abstract']}",
            f"ID  - {p['pmid']}",
            "ER  - "]
    return "\n".join(ris) + "\n"

# ───── 2. PubMed / PMC ─────────────────────────────
def get_pmcid_from_pmid(pmid) -> str | None:
    try:
        rec = Entrez.read(Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid))
        for db in rec[0].get("LinkSetDb", []):
            if db.get("DbTo") == "pmc":
                for link in db.get("Link", []):
                    return link.get("Id")
    except Exception:
        pass
    return None

def get_pmc_fulltext(pmcid) -> str | None:
    try:
        xml = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml").read()
        root = ET.fromstring(xml)
        return "\n".join(
            "".join(p.itertext()).strip()
            for p in root.findall(".//body//p")
            if "".join(p.itertext()).strip())
    except Exception:
        return None

# ───── 3. GPT 呼び出しヘルパー ─────────────────────
def add_usage(model, rsp):
    METER.add(model, rsp.usage.prompt_tokens, rsp.usage.completion_tokens)

def extract_keywords(text, n=5):
    rsp = openai_api_client.chat.completions.create(
        model="gpt-4.1", temperature=0,
        messages=[{"role":"system",
                   "content":"Extract up to 5 biomedical English keywords (comma separated)."},
                  {"role":"user","content":text}]
    )
    add_usage("gpt-4.1", rsp)
    kws = rsp.choices[0].message.content.replace("\n", ",").split(",")
    return [w.strip() for w in kws if w.strip()][:n]

def build_queries(kw: List[str]) -> List[str]:
    q = []
    pairs = [(kw[i], kw[j]) for i in range(len(kw)) for j in range(i+1, len(kw))]
    for a, b in pairs[:10]:
        q.append(f"{a}[Title/Abstract] AND {b}[Title/Abstract]")
    triples = [(kw[i], kw[j], kw[k])
               for i in range(len(kw))
               for j in range(i+1, len(kw))
               for k in range(j+1, len(kw))]
    for a, b, c in triples[:10]:
        q.append(f"{a}[Title/Abstract] AND {b}[Title/Abstract] AND {c}[Title/Abstract]")
    return q

def ai_is_citable_scored(p, sent):
    joint = (f"要旨: {p['abstract']}\n\n本文抜粋: {p['fulltext'][:2000]}"
             if p.get("fulltext") else f"要旨: {p['abstract']}")
    prompt = ("あなたは論文査読AI。文献が主張を支持する度合いを"
              "5段階で評価し理由を述べて下さい。出力: スコア:(数字)/理由:(説明)")

    logging.debug("[GPT] scoring PMID %s\nPrompt (first 200):\n%s",
                  p["pmid"], textwrap.shorten(prompt, 200))

    rsp = openai_api_client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role":"user",
                   "content":f"{prompt}\n＜主張文＞:{sent}\n＜文献情報＞:\n{joint}"}]
    )
    add_usage("gpt-4.1-mini", rsp)   # 料金メータ

    logging.debug("[GPT] done  PMID %s  → %s (tokens P%d/C%d)",
                  p["pmid"],
                  rsp.choices[0].message.content.strip().split("\n",1)[0],
                  rsp.usage.prompt_tokens, rsp.usage.completion_tokens)

    return rsp.choices[0].message.content.strip()

def ai_parallel(papers, sent, progress=None, workers=6):
    out, total, done = [], len(papers), 0
    push(progress, {"step":"ai","pct":0,"msg":f"AI 0/{total}"})

    def bump():
        nonlocal done
        done += 1
        push(progress,{"step":"ai","pct":int(done/total*100),"msg":f"AI {done}/{total}"})


    papers = [p for p in papers if p]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(ai_is_citable_scored, p, sent): p for p in papers}
        for f in as_completed(fut):
            p = fut[f]
            try:
                p["citable_decision"] = f.result()
                m=re.search(r"\d",p["citable_decision"])
                p["score"]=int(m.group()) if m else 0
                out.append(p)
            except Exception as e:
                import logging  
                logging.warning("ai_scoring failed for PMID %s : %s", p.get("pmid"), e)
            bump()
    return out

# ───── 4. PubMed 検索 ─────────────────────────────
def get_paper_details(pmids, progress=None):
    papers=[]
    for pmid in pmids:
        try:
            med = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text").read()
        except Exception as e:
            push(progress, {"step":"error",
                            "msg": f"Fetch details failed for PMID {pmid}: {e}"})
            continue
        if not med.strip():
            continue

        lines = med.split("\n")
        title  = next((ln[6:].strip() for ln in lines if ln.startswith("TI  - ")), "")
        author = abstract = year = journal = volume = issue = pages = doi = ""
        for ln in lines:
            if ln.startswith("FAU - "): author  += ln[6:].strip() + "; "
            if ln.startswith("AB  - "): abstract+= ln[6:].strip() + " "
            if ln.startswith("DP  - "): m=re.search(r"\d{4}",ln); year=m.group(0) if m else year
            if ln.startswith("JT  - "): journal += ln[6:].strip() + " "
            if ln.startswith("VI  - "): volume  = ln[6:].strip()
            if ln.startswith("IP  - "): issue   = ln[6:].strip()
            if ln.startswith("PG  - "): pages   = ln[6:].strip()
            if ln.startswith("AID  - ") and "[doi]" in ln.lower():
                doi = ln.split("AID  - ")[1].split(" ")[0].strip()
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        pmcid = get_pmcid_from_pmid(pmid)
        full  = get_pmc_fulltext(pmcid) if pmcid else None
        papers.append({"title":title,"author":author.strip(),"abstract":abstract.strip(),
                       "fulltext":full,"url":url,"pmid":pmid,"pmcid":pmcid,
                       "year":year,"journal":journal.strip(),
                       "volume":volume,"issue":issue,"pages":pages,"doi":doi})
    return papers

def find_citable_references(sent, query, hits, progress=None):
    pmids = Entrez.read(
        Entrez.esearch(db="pubmed", term=query, retmax=hits)
    ).get("IdList", [])
    push(progress, {"step":"pubmed","msg":f"PubMed {len(pmids)} hits"})
    papers = get_paper_details(pmids)
    return ai_parallel(papers, sent, progress)

# ───── 5. GPT で上位数本に絞る ─────────────────────
def select_refs_with_llm(refs, sent, max_refs, progress=None):
    """
    refs: score>=4 の候補
    戻り値: (selected_max_refs, explanation_text)
    """
    if len(refs) <= max_refs:
        return refs, "fewer than max_refs"

    # ── LLM に渡すテーブルを作成 ──────────────────
    pool = sorted(refs, key=lambda x: -x["score"])[:15]      # 上位 15 本だけ渡す
    table = "\n".join(f"{i+1}. PMID:{r['pmid']} Score:{r['score']} {r['title'][:60]}"
                      for i, r in enumerate(pool))

    prompt = (
        "You are an expert editor. Choose the best "
        f"{max_refs} papers for the statement and explain briefly.\n"
        "Respond ONLY with a numbered or PMID list.\n"
        f"Statement: {sent}\n\n{table}"
    )
    
    push(progress, {"step": "gpt", "msg": f"[selector] {len(pool)} papers"})
    #if progress:
    #    progress({"step": "gpt", "msg": f"[selector] {len(pool)} papers"})

    rsp = openai_api_client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    add_usage("o4-mini", rsp)
    explanation = rsp.choices[0].message.content.strip()
    
    push(progress,{"step":"selector_raw", "msg": explanation[:300]} )
    #if progress:
    #    progress({"step":"selector_raw", "msg": explanation[:300]})  # 先頭300文字

    # ── ① 行頭の番号 1. 2) … を拾う ─────────────────
    nums = [int(n)-1 for n in re.findall(r'^\s*(\d+)[\.\)]?', explanation, re.M)]
    selected = [pool[i] for i in nums if 0 <= i < len(pool)]

    # ── ② PMID: xxxxxxxx を拾う ─────────────────────
    pmids = re.findall(r'PMID[:\s]*(\d{6,8})', explanation, re.I)
    pool_map = {r["pmid"]: r for r in pool}
    for pm in pmids:
        if pm in pool_map and pool_map[pm] not in selected:
            selected.append(pool_map[pm])

    # ── ③ 裸の 8 桁数値を拾う ──────────────────────
    if len(selected) < max_refs:
        nums_raw = re.findall(r'\b(\d{6,8})\b', explanation)
        for pm in nums_raw:
            if pm in pool_map and pool_map[pm] not in selected:
                selected.append(pool_map[pm])
            if len(selected) == max_refs:
                break

    # ── ④ まだ不足していたらスコア順で補完 ────────────
    if len(selected) < max_refs:
        explanation += "\n(fallback: highest-score papers inserted)"
        for r in pool:
            if r not in selected:
                selected.append(r)
            if len(selected) == max_refs:
                break

    return selected[:max_refs], explanation

# ───── 6. SubordinateAgent ─────────────────────────
class SubordinateAgent:
    def __init__(self, hits, threshold, max_refs, row_cb=None):
        self.hits = hits
        self.th = threshold
        self.max = max_refs
        self.row_cb = row_cb

    def analyze_sentence(self, sent, idx, progress=None):
        push(progress,{"step":"keyword","msg":f"[{idx}] extracting kw"} )
        #if progress:
        #    progress({"step":"keyword","msg":f"[{idx}] extracting kw"})

        kw = extract_keywords(sent)
        queries = build_queries(kw)

        detail_rows = []
        pool, seen = [], set()
        score5_cnt = 0
        low_cnt    = 0

        for qi, q in enumerate(queries, 1):
            push(progress, {"step":"query","msg":f"[{idx}] q{qi}/{len(queries)}"})
            #if progress:
            #    progress({"step":"query","msg":f"[{idx}] q{qi}/{len(queries)}"})
            cands = find_citable_references(sent, q, self.hits, progress)
            for p in cands:
                if not p or p["pmid"] in seen:
                    continue
                seen.add(p["pmid"])

                detail_rows.append({
                    "sent_idx":idx,"phase":"rank","sentence":sent,
                    "keyword":", ".join(kw),"query":q,
                    "pmid":p["pmid"],"title":p["title"],
                    "score":p["score"],"reason":p["citable_decision"],
                    "accepted":False,"explanation":""
                })

                if p["score"] >= 4:
                    pool.append(p)
                else:
                    low_cnt += 1
                if p["score"] == 5:
                    score5_cnt += 1

            if score5_cnt >= 3:     
                break

        push(progress,{"step":"select","msg":f"[{idx}] selecting top {self.max}"} )
        #if progress:
        #    progress({"step":"select","msg":f"[{idx}] selecting top {self.max}"})
        
        notice_msg =""
        if pool:
            selected, why = select_refs_with_llm(pool, sent, self.max, progress)
            if not selected:
                notice_msg = (f"No paper was finally selected (although {len(pool)} "
                      f"papers scored ≥4). Try increasing 'Hits'.")
                why = notice_msg 
                push(progress, {"step": "notice", "msg": notice_msg})
                #if progress: progress({"step": "notice", "msg": notice_msg})
        else:
            notice_msg = (f"No reference scored ≥4 (found {low_cnt} papers scored 1–3). "
                  "Try increasing 'Hits' (e.g. 50 or 100) and run again.")
            push(progress,{"step":"notice","msg":notice_msg} )
            #if progress:
            #    progress({"step":"notice","msg":notice_msg})
            selected, why = [], notice_msg

        # accepted フラグを付与
        accepted_set = {p["pmid"] for p in selected}
        for row in detail_rows:
            row["accepted"] = row["pmid"] in accepted_set

        detail_rows.append({
            "sent_idx":idx,"phase":"select","sentence":sent,
            "keyword":", ".join(kw),"query":"",
            "pmid":"","title":"","score":"","reason":"",
            "accepted":"","explanation":why
        })
        # Notice 行がある場合は detail_rows にも残す
        if notice_msg:
            detail_rows.append({
                "sent_idx": idx, "phase": "notice", "sentence": sent,
                "keyword": ", ".join(kw), "query": "",
                "pmid": "", "title": "", "score": "", "reason": "",
                "accepted": "", "explanation": notice_msg
            })

        if self.row_cb:
            for r in detail_rows:
                self.row_cb(r)
        logging.debug("[select] sent_idx=%s  selected=%s", idx,
                    [p.get("pmid") for p in selected])
        return sorted(selected, key=lambda x: -x.get("score", 0))

# ───── 7. EditAgent ────────────────────────────────
class EditAgent:
    def insert_and_export_endnote(self, sents, all_refs,
                                  max_refs, prefix: str | None = None):
        if prefix:
            txt, risf = f"{prefix}.txt", f"{prefix}.ris"

        lines, ris_blocks, seen, num = [], [], set(), 1
        for s, refs in zip(sents, all_refs):
            logging.debug("EditAgent refs=%s", [r.get("pmid") for r in refs or []])  # ←追加
            if refs is None:
                refs = []
            refs = [p for p in refs if p]
            refs = refs[:max_refs]
            tag = "; ".join(endnote_tag(p).strip("{}") for p in refs)
            if tag and "." in s:
                pos = s.rfind(".")
                s = f"{s[:pos]} {{{tag}}}{s[pos:]}"
            elif tag:
                s = f"{s} {{{tag}}}"
            lines.append(s)

            for p in refs:
                if p["pmid"] in seen:
                    continue
                seen.add(p["pmid"])
                ris_blocks.append(to_ris(p, num))
                num += 1

        cited = "\n".join(lines)
        if prefix:
            open(txt, "w", encoding="utf-8").write(cited + "\n")
            open(risf, "w", encoding="utf-8").write("\n".join(ris_blocks))
        return cited, "\n".join(ris_blocks)

# ───── 8. レポート書き出し ──────────────────────────
def write_report(rows: List[Dict]) -> str:
    tmp  = tempfile.gettempdir()
    base = str(uuid.uuid4())
    csvf = os.path.join(tmp, base + ".csv")
    jsf  = os.path.join(tmp, base + ".json")
    with open(csvf, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        else:
            f.write("sent_idx,pmid,score,reason,accepted\n")
    open(jsf, "w", encoding="utf-8").write(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    return base

def write_detail_report(rows: List[Dict]) -> str:
    tmp  = tempfile.gettempdir()
    base = str(uuid.uuid4())
    csvf = os.path.join(tmp, base + "_detail.csv")
    jsf  = os.path.join(tmp, base + "_detail.json")
    # extrasaction='ignore' で列が増えても落ちないようにする
    with open(csvf, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_HEADERS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    open(jsf, "w", encoding="utf-8").write(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    return base

# ───── 9. パイプライン本体 ───────────────────────────
def run_pipeline(text: str,
                 hits: int = 10,
                 threshold: int = 4,
                 max_refs: int = 3,
                 generate_files: bool = False,
                 out_prefix: str | None = None,
                 progress=None):

    push(progress, {"step":"start","msg":"pipeline start"})
    references =[]
    #if progress:
    #    progress({"step":"start","msg":"pipeline start"})

    sents = split_sentences(text)
    refs_all: List[List[Dict]] = [None]*len(sents)
    report_rows:  List[Dict]   = []

    # ---------- ① detail 行を thread-safe に収集する ------------
    q_detail: Queue = Queue()

    def collect_row(row:dict): q_detail.put(row)
    # ------------------------------------------------------------

    def task(i, sent):
        ag = SubordinateAgent(hits, threshold, max_refs, row_cb=collect_row)
        try:
            refs = ag.analyze_sentence(sent, i, progress)
        except Exception as e:
            import logging
            logging.exception("analyze_sentence failed (sent_idx=%s): %s", i, e)
            refs = []
        refs_all[i] = refs

        for r in refs or []:
            report_rows.append({
                "sent_idx": i,
                "pmid": r["pmid"],
                "score": r["score"],
                "reason": r["citable_decision"],
                "accepted": True
            })

    # 並列で各文を処理
    with ThreadPoolExecutor(max_workers=2) as pool:
        [pool.submit(task, i, s) for i, s in enumerate(sents)]
        pool.shutdown(wait=True)

    # ---------- ② Queue から detail_rows をまとめる ------------
    detail_rows=[]
    while True:
        try:
            detail_rows.append(q_detail.get_nowait())
        except Empty:
            break
    # ------------------------------------------------------------

    cited, ris = EditAgent().insert_and_export_endnote(
        sents, refs_all, max_refs, out_prefix if generate_files else None
    )

    base_rep    = write_report(report_rows)
    base_detail = write_detail_report(detail_rows)

    push(progress, {"step":"cost","usd":round(METER.usd(),4)})
    #if progress:
    #    progress({"step":"cost","usd":round(METER.usd(),4)})

    unique = {}
    for lst in refs_all:
        if not lst: continue
        for p in lst:
            # p は {'pmid', 'title', 'author', 'journal', ...} 全メタを含む
            unique[p['pmid']] = p
        references = list(unique.values())
        
    return cited, ris, base_rep, base_detail, references



# ───── 10. CLI テスト ─────────────────────────────
if __name__ == "__main__":
    import sys, pathlib
    fname = input("txt file (empty=stdin): ").strip()
    full  = pathlib.Path(fname).read_text(encoding="utf-8") if fname else sys.stdin.read()
    cited, ris, _, _ = run_pipeline(full)
    print(cited)
    print("\n--- RIS ---\n", ris)