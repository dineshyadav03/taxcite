"""Build data/processed/cases.jsonl for Precedent RAG.

Every case name, citation, and holding summary below was sourced from
itatonline.org's digest of landmark Supreme Court tax judgments
(https://itatonline.org/digest/articles/landmark-supreme-court-judgments-relevant-to-day-to-day-tax-practice-under-the-income-tax-act-2025-and-income-tax-act-1961/),
then independently cross-checked case-by-case against a secondary source
(indiankanoon.org and/or web search corroboration) before being written
here - a fabricated or wrong case citation would be a worse failure than
anything else in this project, so nothing here is generated or recalled
from model memory. One citation needed correction after the cross-check:
the digest listed Vodafone International Holdings v. UOI without a clear
year ("341 ITR 1"); indiankanoon.org's own judgment record plus
independent web corroboration confirmed the decision date as 20 January
2012, not the digest's apparent "2003" - corrected here rather than
dropped, since the cross-check resolved cleanly. Holding summaries below
are written in this project's own words from the verified facts, not
copied from the source page.

This script only generates section-link CANDIDATES (embed each case's
holding text - the plan's one exception to "zero new embedding calls",
since cases are new content with no existing vector - then query the
existing income_tax_sections collection). It does NOT auto-accept links:
candidates are printed for manual reading against the real section text
(retrieve.fetch_section) before any link is written to cases.jsonl,
same verify-then-trust discipline as the correspondence map, but done by
manual review rather than an LLM confidence label - a case-to-section
link is exactly the kind of claim this project's premise says must be
checked, not asserted.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chromadb

from embed import EMBED_BATCH_SIZE, EMBED_CALL_SPACING_SECONDS, embed_texts
from retrieve import get_collection

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "processed" / "cases.jsonl"
CANDIDATES_PATH = ROOT / "data" / "processed" / "cases_link_candidates.json"
CASE_GRAPH_PATH = ROOT / "data" / "processed" / "case_graph.json"
CHROMA_DIR = ROOT / "chroma_db"

SOURCE_URL = (
    "https://itatonline.org/digest/articles/landmark-supreme-court-judgments-relevant-to-"
    "day-to-day-tax-practice-under-the-income-tax-act-2025-and-income-tax-act-1961/"
)

# case_name, citation, holding (this project's own wording), search_text (used only to
# generate link candidates - not persisted)
CASES = [
    {
        "case_name": "CIT v. Raja Benoy Kumar Sahs Roy",
        "citation": "(1957) 32 ITR 466 (SC)",
        "holding": "Agricultural income requires actual agricultural operations involving human skill and labour on the land; income from spontaneous forest growth with no cultivation is not agricultural income.",
        "search_text": "agricultural income definition requires cultivation human skill and labour on land, forest produce",
    },
    {
        "case_name": "Singhai Rakesh Kumar v. UOI",
        "citation": "(2001) 115 Taxman 101 (SC)",
        "holding": "Upheld the constitutional validity of excluding agricultural land situated within specified municipal limits from the definition of a capital asset.",
        "search_text": "agricultural land within municipal limits excluded from definition of capital asset",
    },
    {
        "case_name": "Gowli Buddanna v. CIT",
        "citation": "(1966) 60 ITR 293 (SC)",
        "holding": "A Hindu Undivided Family can validly consist of a single male coparcener together with female members; the family's status as a taxable HUF does not require at least two male members.",
        "search_text": "Hindu undivided family single male coparcener status assessment",
    },
    {
        "case_name": "CIT v. Indira Balkrishna",
        "citation": "(1960) 39 ITR 546 (SC)",
        "holding": "Mere joint ownership of property and joint receipt of income does not make co-owners an Association of Persons; an AOP requires the members to join in a common purpose or action to produce income.",
        "search_text": "association of persons definition joint ownership common purpose to produce income",
    },
    {
        "case_name": "K.D. Kamath and Co. v. CIT",
        "citation": "(1971) 82 ITR 680 (SC)",
        "holding": "A partnership remains valid for tax purposes so long as mutual agency and profit-sharing exist between partners, even if operational control is concentrated in one partner.",
        "search_text": "valid partnership firm registration mutual agency profit sharing control",
    },
    {
        "case_name": "Vodafone International Holdings v. UOI",
        "citation": "(2012) 341 ITR 1 (SC)",
        "holding": "Legitimate tax planning within the framework of law is permissible; a transaction structured to reduce tax liability is not a sham or colourable device merely because it is tax-efficient, absent evidence the structure lacks commercial substance.",
        "search_text": "tax planning legitimate versus colourable device transfer of shares offshore transaction capital gains",
    },
    {
        "case_name": "CIT v. Durga Prasad More",
        "citation": "(1971) 82 ITR 540 (SC)",
        "holding": "Tax authorities may look beyond the apparent form of a transaction and examine surrounding circumstances using the test of human probabilities to determine whether a claimed source of funds is genuine.",
        "search_text": "test of human probabilities surrounding circumstances genuineness of transaction unexplained investment",
    },
    {
        "case_name": "Sumati Dayal v. CIT",
        "citation": "(1995) 214 ITR 801 (SC)",
        "holding": "Documentary evidence that is inconsistent with normal human conduct may be disregarded by tax authorities, who are entitled to test the reality of a transaction against surrounding circumstances rather than accept it at face value.",
        "search_text": "apparent not real documentary evidence surrounding circumstances winnings undisclosed income",
    },
    {
        "case_name": "T.R.F Ltd v. CIT",
        "citation": "(2010) 323 ITR 397 (SC)",
        "holding": "After the 1989 amendment to Section 36(1)(vii), an assessee claiming a bad-debt deduction need only write the debt off as irrecoverable in its accounts; it is not necessary to separately prove the debt has actually become irrecoverable.",
        "search_text": "bad debt deduction write off accounts irrecoverable section 36",
    },
    {
        "case_name": "Empire Jute Co. Ltd v. CIT",
        "citation": "(1980) 124 ITR 1 (SC)",
        "holding": "Expenditure that merely facilitates a business's trading operations more efficiently, without touching the capital structure, is revenue expenditure even if it produces an advantage of some enduring benefit.",
        "search_text": "capital versus revenue expenditure enduring advantage trading operations",
    },
    {
        "case_name": "Bharat Earth Movers v. CIT",
        "citation": "(2000) 245 ITR 428 (SC)",
        "holding": "A provision for leave encashment based on employees' accrued entitlement is an ascertained (not contingent) liability and is deductible in the year it accrues, even though payment or quantification occurs later.",
        "search_text": "provision for leave encashment ascertained liability contingent liability deduction accrual",
    },
    {
        "case_name": "CIT v. Alps Theatre",
        "citation": "(1967) 65 ITR 377 (SC)",
        "holding": "Depreciation is allowable only on the superstructure of a building, not on the land beneath it, since land does not depreciate.",
        "search_text": "depreciation on building land not depreciable superstructure",
    },
    {
        "case_name": "S. A. Builders Ltd. v. CIT",
        "citation": "(2007) 288 ITR 1 (SC)",
        "holding": "Interest on funds borrowed and advanced interest-free to a sister concern is deductible if advanced on grounds of commercial expediency; the Revenue cannot substitute its own judgment of business prudence for the assessee's.",
        "search_text": "interest deduction borrowed funds advance to sister concern commercial expediency section 36",
    },
    {
        "case_name": "CIT v. Excel Industries Ltd",
        "citation": "(2013) 358 ITR 295 (SC)",
        "holding": "Where the Revenue has accepted a particular view in the assessee's favour across several earlier assessment years without challenge, it cannot depart from that consistent position in a later year without compelling reason.",
        "search_text": "consistency principle revenue accepted earlier assessment years cannot depart",
    },
    {
        "case_name": "Apollo Tyres Ltd v. CIT",
        "citation": "(2002) 255 ITR 273 (SC)",
        "holding": "An Assessing Officer computing book profit under the minimum-alternate-tax provision may only verify that accounts are certified as maintained in accordance with the Companies Act; the officer cannot re-examine or recompute the certified net profit itself.",
        "search_text": "book profit minimum alternate tax auditor certified accounts assessing officer cannot recompute",
    },
    {
        "case_name": "Hindustan Steel Ltd. v. State of Orissa",
        "citation": "(1972) 83 ITR 26 (SC)",
        "holding": "A penalty for breach of a statutory obligation should not be imposed merely because it is lawful to do so; penalty requires a deliberate or conscious disregard of the obligation, not a mere technical or venial breach.",
        "search_text": "penalty mens rea conscious disregard technical breach discretion",
    },
    {
        "case_name": "CIT v. Reliance Petroproducts Pvt Ltd.",
        "citation": "(2010) 322 ITR 158 (SC)",
        "holding": "Making an incorrect legal claim in a return of income, where all material facts were disclosed, does not by itself amount to furnishing inaccurate particulars of income for penalty purposes.",
        "search_text": "penalty concealment inaccurate particulars incorrect claim return of income section 271",
    },
    {
        "case_name": "CIT v. Vegetable Products Ltd.",
        "citation": "(1973) 88 ITR 192 (SC)",
        "holding": "Where a taxing provision is reasonably capable of two different interpretations, the interpretation favourable to the assessee must be adopted.",
        "search_text": "ambiguous taxing provision two reasonable interpretations favourable to assessee",
    },
]

CANDIDATES_PER_CASE = 5

# Manually verified links: for every case, each candidate section's actual
# text (retrieve.fetch_section) was read and compared against the case's
# real holding before being kept here - not auto-accepted from vector
# distance. Several links are NOT the nearest-distance candidate: vector
# search surfaced topically-adjacent-but-substantively-wrong sections as
# the top hit more than once (e.g. Vodafone's nearest hit was a capital-
# gains exemption clause; the actual GAAR "commercial substance" test
# provision the case's principle maps to ranked lower). Two cases (Excel
# Industries, Vegetable Products) turn on a general interpretive
# principle rather than any single operative provision, and none of
# their candidates were a genuine match on manual reading - left
# unlinked rather than forcing a weak match.
LINKS = {
    "CIT v. Raja Benoy Kumar Sahs Roy": [("itact1961", "2")],
    "Singhai Rakesh Kumar v. UOI": [("itact1961", "2")],
    "Gowli Buddanna v. CIT": [("itact1961", "171"), ("itact2025", "315")],
    "CIT v. Indira Balkrishna": [("itact2025", "24")],
    "K.D. Kamath and Co. v. CIT": [("itact1961", "184"), ("itact2025", "325")],
    "Vodafone International Holdings v. UOI": [("itact1961", "97"), ("itact2025", "180")],
    "CIT v. Durga Prasad More": [("itact1961", "69B"), ("itact2025", "103")],
    "Sumati Dayal v. CIT": [("itact1961", "69A"), ("itact2025", "104")],
    "T.R.F Ltd v. CIT": [("itact1961", "36"), ("itact2025", "31")],
    "Empire Jute Co. Ltd v. CIT": [("itact1961", "37"), ("itact2025", "34")],
    "Bharat Earth Movers v. CIT": [("itact1961", "43B"), ("itact2025", "37")],
    "CIT v. Alps Theatre": [("itact1961", "32"), ("itact2025", "33")],
    "S. A. Builders Ltd. v. CIT": [("itact1961", "36"), ("itact2025", "32")],
    "CIT v. Excel Industries Ltd": [],
    "Apollo Tyres Ltd v. CIT": [("itact1961", "115J"), ("itact2025", "206")],
    "Hindustan Steel Ltd. v. State of Orissa": [("itact1961", "273B"), ("itact2025", "470")],
    "CIT v. Reliance Petroproducts Pvt Ltd.": [("itact1961", "271"), ("itact2025", "439")],
    "CIT v. Vegetable Products Ltd.": [],
}


def write_final():
    """Write cases.jsonl from CASES + the manually verified LINKS above."""
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for case in CASES:
            links = LINKS[case["case_name"]]
            row = {
                "case_name": case["case_name"],
                "citation": case["citation"],
                "holding": case["holding"],
                "source_url": SOURCE_URL,
                "linked_sections": [{"act_id": act_id, "section": section} for act_id, section in links],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    linked_count = sum(1 for c in CASES if LINKS[c["case_name"]])
    print(f"Wrote {len(CASES)} cases ({linked_count} with a linked section) -> {OUT_PATH}")


def generate_candidates():
    collection = get_collection()
    candidates = []
    for i, case in enumerate(CASES):
        if i > 0:
            time.sleep(EMBED_CALL_SPACING_SECONDS)
        embedding = embed_texts([case["search_text"]], input_type="query")[0]
        results = collection.query(query_embeddings=[embedding], n_results=CANDIDATES_PER_CASE)
        hits = [
            {
                "act_id": results["metadatas"][0][j]["act_id"],
                "section": results["metadatas"][0][j]["section"],
                "distance": results["distances"][0][j],
            }
            for j in range(len(results["ids"][0]))
        ]
        candidates.append({"case_name": case["case_name"], "candidates": hits})
        print(f"  [{i + 1}/{len(CASES)}] {case['case_name']}: {[(h['act_id'], h['section']) for h in hits]}")

    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=1), encoding="utf-8")
    print(f"\nWrote link candidates -> {CANDIDATES_PATH} (manual review required before cases.jsonl is written)")


def index_cases():
    """Embed each case (case name + holding) into a new income_tax_cases
    ChromaDB collection, batched and paced the same way embed.py indexes
    the two Acts - 18 items easily fits in two batches, but unpaced calls
    risk the same 429 storm the original corpus build already hit once."""
    if not OUT_PATH.exists():
        raise SystemExit("cases.jsonl not found - run with --write first")
    cases = [json.loads(line) for line in OUT_PATH.read_text(encoding="utf-8").splitlines() if line]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]
    if "income_tax_cases" in existing:
        client.delete_collection("income_tax_cases")
    collection = client.create_collection("income_tax_cases")

    documents = [f"{c['case_name']} ({c['citation']}): {c['holding']}" for c in cases]
    ids = [f"case::{i}" for i in range(len(cases))]
    metadatas = [
        {
            "case_name": c["case_name"],
            "citation": c["citation"],
            "holding": c["holding"],
            "source_url": c["source_url"],
        }
        for c in cases
    ]

    all_embeddings = []
    for batch_num, start in enumerate(range(0, len(documents), EMBED_BATCH_SIZE)):
        if batch_num > 0:
            time.sleep(EMBED_CALL_SPACING_SECONDS)
        batch = documents[start : start + EMBED_BATCH_SIZE]
        all_embeddings.extend(embed_texts(batch, input_type="document"))

    collection.add(documents=documents, embeddings=all_embeddings, metadatas=metadatas, ids=ids)
    print(f"Indexed {collection.count()} cases into income_tax_cases collection")

    case_graph = defaultdict(list)
    for c in cases:
        for link in c["linked_sections"]:
            key = f"{link['act_id']}::{link['section']}"
            case_graph[key].append(c["case_name"])
    CASE_GRAPH_PATH.write_text(json.dumps(case_graph, indent=1), encoding="utf-8")
    print(f"Wrote {len(case_graph)} section->case edges -> {CASE_GRAPH_PATH}")


if __name__ == "__main__":
    if "--write" in sys.argv:
        write_final()
    elif "--index" in sys.argv:
        index_cases()
    else:
        generate_candidates()
