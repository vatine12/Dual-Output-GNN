"""
Phase-0 improved faithfulness metric for the Dual-Output-GNN project.

Measures REAL entity fabrication, not surface-form mismatch. Removes the artifact
sources that made the old "strict" metric report ~12% hallucination when the true
rate is ~0.5%:

  1. Diacritics / non-latin letters   Kovač->Kovac, Løkke->Lokke  (translit + NFKD)
  2. Date / number verbalization      1982-07-23 -> "July 23rd 1982"
  3. Entity shortening                "Barkov, Jr." -> "Barkov"
  4. Leading articles                 "the Akita Museum" -> "Akita Museum"
  5. Acronym / spacing                "C.D. FAS" -> "CD FAS", "3Arena" -> "Arena"
  6. Predicate verbalization          predicate "isbnNumber" grounds the word "ISBN"

A prediction mention (proper-noun span or numeric literal) is GROUNDED if it traces
back to an input-triple entity, literal, or predicate; otherwise it is a hallucination.
This file is BOTH the offline re-scorer and the reference implementation for the patch
that goes into fixed_ablation_common.py.

Usage:  python3 rescore_metrics.py per_sample_A.csv [per_sample_B.csv ...]
"""
import sys, csv, json, re, unicodedata, statistics as st

# non-decomposing letters with standard latin transliterations
_TRANSLIT = {"ø":"o","Ø":"o","æ":"ae","Æ":"ae","œ":"oe","Œ":"oe","ð":"d","Ð":"d","þ":"th","Þ":"th",
             "ł":"l","Ł":"l","ß":"ss","đ":"d","Đ":"d","ħ":"h","ı":"i","İ":"i","ŋ":"ng"}

def _strip_accents(s):
    s = "".join(_TRANSLIT.get(c, c) for c in str(s))
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

_DET = re.compile(r"^(the|a|an)\s+")

def _norm(s):
    s = _strip_accents(str(s)).lower().strip().strip('"').strip("'")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _DET.sub("", s)

def _word_match(text, surface):
    if not surface:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(surface) + r"(?![a-z0-9])", text) is not None

_MONTHS = ["january","february","march","april","may","june","july","august",
           "september","october","november","december"]

def _date_surface_tokens(literal):
    toks = set()
    raw = _strip_accents(str(literal)).strip().strip('"').strip("'")
    m = re.match(r"^(\d{3,4})-(\d{1,2})-(\d{1,2})$", raw)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        toks.add(str(y))
        if 1 <= mo <= 12:
            toks.add(_MONTHS[mo-1]); toks.add(_MONTHS[mo-1][:3])
        toks.add(str(d)); toks.add(str(d).zfill(2))
        for suf in ("st","nd","rd","th"):
            toks.add(f"{d}{suf}")
        return toks
    if re.match(r"^\d{3,4}$", raw):
        toks.add(raw)
    return toks

def _digits(s):
    return re.sub(r"[^0-9]", "", str(s))

def _iter_vals(triples):
    for t in triples:
        if isinstance(t, dict):
            yield t.get("subject", ""); yield t.get("object", "")
        else:
            yield t[0]; yield t[2]

def _split_camel(s):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))

def _predicate_tokens(triples):
    toks = set()
    for t in triples:
        pred = t.get("predicate", "") if isinstance(t, dict) else (t[1] if len(t) > 1 else "")
        toks.update(_norm(_split_camel(pred)).split())
    return toks

def _build_grounding(triples):
    forms, tokens, numcores = [], set(), set()
    for v in _iter_vals(triples):
        n = _norm(v)
        if n:
            forms.append(n); tokens.update(n.split())
        tokens.update(_date_surface_tokens(v))
        dg = _digits(v)
        if dg:
            numcores.add(dg)
    tokens.update(_predicate_tokens(triples))
    return forms, tokens, numcores

_STOP = frozenset("the a an and or but if then this that these those it its he she they them his "
                  "her their in on at of to by with for from as is are was were be been being there "
                  "here also however".split())

def _extract_mentions(text):
    men = set()
    for m in re.findall(r"[A-Z][A-Za-z]*(?:[ -][A-Z][A-Za-z]*)*", _strip_accents(text)):
        n = _norm(m)
        if len(n) < 3:
            continue
        if " " not in n and n in _STOP:
            continue
        men.add(n)
    for m in re.findall(r"[A-Za-z0-9]+(?:[./\-][A-Za-z0-9]+)*", text):
        if any(c.isdigit() for c in m):
            men.add(_norm(m))
    return {m for m in men if m}

def score(prediction, triples):
    pred_norm = _norm(prediction)
    forms, tokens, numcores = _build_grounding(triples)
    forms_despaced = [f.replace(" ", "") for f in forms]

    found = 0
    for f in forms:
        if _word_match(pred_norm, f) or all(_word_match(pred_norm, w) for w in f.split()):
            found += 1
    recall = found / len(forms) if forms else 1.0

    mentions = _extract_mentions(prediction)

    def grounded(m):
        for f in forms:
            if m == f or _word_match(f, m):
                return True
        if all(tok in tokens for tok in m.split()):
            return True
        dm = _digits(m)
        if dm and any(dm in c or c in dm for c in numcores):
            return True
        md = m.replace(" ", "")
        if len(md) >= 3 and any(md in x or x in md for x in forms_despaced):
            return True
        return False

    supported = {m for m in mentions if grounded(m)}
    hallucinated = mentions - supported
    precision = len(supported) / len(mentions) if mentions else 1.0
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "hallucination_rate": 1.0 - precision,
        "hallucinated_entities": sorted(hallucinated),
    }

def run(path):
    rows = list(csv.DictReader(open(path)))
    H = []; P = []; R = []; flagged = 0; resid = []
    for r in rows:
        s = score(r["prediction"], json.loads(r["input_triples"]))
        P.append(s["entity_precision"]); R.append(s["entity_recall"]); H.append(s["hallucination_rate"])
        if s["hallucinated_entities"]:
            flagged += 1
            if len(resid) < 15:
                resid.append((s["hallucinated_entities"], r["prediction"][:120]))
    try:
        old = st.mean([float(r["hallucination_rate"]) for r in rows])
        oldstr = f"OLD={old:.4f}  "
    except Exception:
        oldstr = ""
    n = len(rows)
    print(f"{path.split('/')[-1]}")
    print(f"  {oldstr}NEW halluc={st.mean(H):.4f}  ent_prec={st.mean(P):.4f}  ent_recall={st.mean(R):.4f}  flagged={100*flagged/n:.1f}%")
    return resid

if __name__ == "__main__":
    resid = None
    for p in sys.argv[1:]:
        resid = run(p)
    if resid:
        print("\n-- residual still flagged (should be mostly genuine) --")
        for he, pr in resid[:12]:
            print(" ", he, "||", pr)
