import copy
import hashlib
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}


def w(tag):
    return f"{{{W_NS}}}{tag}"


EXPECTED_SOURCE_HASH = (
    "daa0c3bd434e9367880fc484eff3508fa15df8e2acef66acb8af39c8f8facf46"
)


PARAGRAPHS = {
    0: "Ben-Gurion University of the Negev",
    1: "Master's Thesis Research Proposal",
    2: "יוריסטיקות קבילות מבוססות מסדי תבניות לתכנון טמפורלי הסתברותי תחת מועדי יעד",
    3: "Admissible Pattern Database Heuristics for Probabilistic Temporal Planning under Deadlines",
    5: "Abstract",
    6: (
        "This research proposes an admissible pattern database (PDB) heuristic for "
        "probabilistic temporal planning with durative and concurrent actions under a "
        "deadline. The primary objective is to maximize the expected reward accumulated "
        "before the deadline; binary goal achievement is a special case in which the "
        "reward is 1 if the goal is reached in time and 0 otherwise. In this "
        "reward-maximization setting, admissibility means that the heuristic provides an "
        "upper bound on the optimal value."
    ),
    7: (
        "The central idea is to first construct an optimistic PDB without an explicit "
        "representation of time, solve the abstract model exactly, validate the resulting "
        "policy graph using a Simple Temporal Network (STN), and refine the abstraction "
        "only where the policy exploits temporal behavior that cannot occur in the original "
        "model. This process yields an anytime sequence of upper bounds: the initial bound "
        "is inexpensive and loose, and each sound refinement should tighten it without "
        "sacrificing admissibility."
    ),
    8: "Scientific Background",
    9: (
        "Markov decision processes (MDPs) are a fundamental model for decision making "
        "under uncertainty, but their standard formulation assumes instantaneous, "
        "sequential actions. In probabilistic temporal planning, actions may have duration, "
        "execute concurrently, and interact through start, invariant, and end conditions. "
        "Prior models and algorithms extend MDPs to durative and concurrent actions, but "
        "encoding scheduling information in the state can greatly enlarge the state space "
        "[1,2]."
    ),
    10: (
        "TP-MCTS addresses online planning in the CoMDP+ model by combining Monte Carlo "
        "Tree Search with an STN to decide both which action to dispatch and when to "
        "dispatch it. It also introduces PTRPG, which estimates the probability of success "
        "under temporal constraints [1]. These components form the starting point of the "
        "thesis rather than its proposed contribution. The new contribution sought here is "
        "an admissible abstraction-based upper bound that can be precomputed and tightened "
        "during search."
    ),
    11: (
        "Pattern databases solve a small abstraction of the state space exactly and store "
        "the resulting abstract-state values in a table. In deterministic planning, PDBs "
        "are a major source of admissible heuristics; their quality depends on pattern "
        "selection and on preserving a sound relationship between the concrete and abstract "
        "models [3,4]. In MDPs, merging states creates an additional difficulty: concrete "
        "states mapped to the same abstract state may induce different transition "
        "distributions. An optimistic abstraction must therefore represent every relevant "
        "concrete transition rather than replace them by an average that may reduce the "
        "value [5]."
    ),
    12: (
        "The research gap is a compact PDB for probabilistic temporal planning that does "
        "not initially encode every action start time or running action in the state, yet "
        "still yields a formal upper bound and supports targeted temporal refinement. STN "
        "validation can expose infeasible abstract behavior [7], while the reachable "
        "abstract-state graph can be solved using LAO* or other dynamic-programming methods "
        "[6]."
    ),
    14: "Research Objective and Questions",
    15: (
        "The primary objective is to develop, analyze, and evaluate an adaptive admissible "
        "PDB heuristic for CoMDP+, in which temporal information is represented only when "
        "needed to eliminate impossible optimistic behavior. The study will address four "
        "questions:"
    ),
    16: (
        "Which conditions on the abstraction of facts, rewards, and transition "
        "distributions are sufficient for the PDB value to upper-bound the optimal expected "
        "reward before the deadline?"
    ),
    17: (
        "How can a small, useful explanation be extracted from an STN failure and translated "
        "into a state split or temporal feature that removes the spurious behavior without "
        "excluding any legal concrete behavior?"
    ),
    18: (
        "How can the PDB be re-solved efficiently after a local refinement, and how should "
        "offline refinement be combined with online refinement focused on policies "
        "reachable from the current state?"
    ),
    19: (
        "What trade-off arises among computation time, PDB size, bound tightness, "
        "action-selection quality, expected reward, success probability, and temporal "
        "feasibility?"
    ),
    20: "Formal Setting and Proof Objectives",
    21: (
        "Let s be a concrete state, D the remaining time, and R≤D the reward accumulated by "
        "the deadline. The optimal value is the maximum expected reward over all legal "
        "policies. For a pattern Φ and abstraction mapping αΦ, the heuristic HΦ is "
        "admissible if, for every state,"
    ),
    22: "HΦ(αΦ(s))  ≥  V*D(s) = supπ Eπ[R≤D | s]",
    23: (
        "The initial analysis will consider actions with positive deterministic durations "
        "and probabilistic outcomes, with bounded non-negative rewards. Reaching a goal "
        "before the deadline will serve as the base case for testing the proofs. "
        "Intermediate rewards will be included as long as their sum is bounded. If "
        "repeatable positive-reward cycles are possible, a timeless abstraction may have an "
        "unbounded value; in that case, a reward cap or coarse time budget must be included "
        "in the abstract state."
    ),
    24: (
        "The proof objectives are: (i) an admissibility theorem for the initial abstraction "
        "under explicit assumptions; (ii) an admissibility-preservation theorem for each "
        "refinement operator; (iii) monotonic upper bounds, such that every sound refinement "
        "does not increase HΦ; and (iv) termination or convergence conditions when the set "
        "of refinement features is finite. Average empirical performance will not be treated "
        "as a substitute for a formal admissibility proof."
    ),
    25: "Proposed Novelty",
    26: (
        "Unlike an initial SMDP formulation that explicitly records the remaining time and "
        "every running action, the proposed direction begins with a timeless abstraction "
        "and introduces temporal information only in response to concrete evidence of a "
        "failure. The research novelty is the combination of an optimistic probabilistic "
        "PDB, STN-based policy-graph validation, and local refinement guided by temporal "
        "conflicts. The claim is not that the abstract model is equivalent to the original "
        "model, but that it is a controlled relaxation whose value is a useful upper bound."
    ),
    28: "Research Methodology",
    29: "1. Pattern Construction and the Abstract Model",
    30: (
        "The pattern Φ will be initialized with goal facts or reward-bearing facts and "
        "expanded gradually in response to missing conditions and detected conflicts. The "
        "abstract state will contain only the projection onto Φ; absolute time, exact start "
        "times, and the STN will initially be omitted. Delete effects, interference, and "
        "unrepresented conditions will be relaxed only in an optimistic direction. "
        "Already-running actions will be handled by an optimistic completion operator: the "
        "end outcome is represented immediately, harmful effects are ignored, and every "
        "omitted dependency is treated as permitted. The precise definition of this operator "
        "will be a proof obligation rather than an implicit assumption."
    ),
    31: "2. Transition and Reward Abstraction",
    32: (
        "When several concrete states in one abstract cell induce different transition "
        "distributions for the same action, a transition-set abstraction B=(P1,...,Pk) will "
        "be examined. In solving the PDB, maximization will select both an action and an "
        "allowed distribution from B. This additional choice deliberately enlarges the "
        "abstract policy's power and can therefore yield an upper bound, provided that every "
        "concrete transition and its reward are represented and no reward is reduced. More "
        "compact alternatives, such as a convex envelope or partitioning the abstract cell, "
        "will be studied if retaining all distributions is too costly."
    ),
    33: "3. Exact Solution and Policy-Graph Extraction",
    34: (
        "The abstract model will be solved exactly over the portion reachable from the "
        "initial state. LAO* is a natural candidate when the abstraction forms a stochastic "
        "shortest-path problem with proper policies; value iteration or Modified Policy "
        "Iteration will be used when another finite formulation is more appropriate. The "
        "output is a probabilistic policy graph rather than a single sequence. Checking one "
        "trace is therefore useful only for anytime prioritization, whereas a full "
        "feasibility claim requires examining all relevant reachable branches."
    ),
    35: "4. Temporal Validation and Conflict-Guided Refinement",
    36: (
        "For each selected policy trace, an STN will be constructed with action start and "
        "end time points, duration, ordering, concurrency, and deadline constraints. "
        "Inconsistency will yield a negative cycle or a small infeasible constraint core. A "
        "compact feature will then be derived from the conflict, such as whether an action "
        "is still running, an ordering relation between two actions, a slack class, or a "
        "relative-time bound. The relevant abstract state will be split according to this "
        "feature, or an equivalent transition constraint will be added. Every refinement "
        "must map each concrete state to at least one refined state and prove that it "
        "eliminates only behavior with no concrete witness."
    ),
    37: "5. Re-solving and Integration with TP-MCTS",
    38: (
        "After refinement, values and the policy will be updated incrementally. Computation "
        "may begin offline and continue online from the current root, prioritizing "
        "high-value or high-probability branches. Early termination leaves an admissible but "
        "loose bound; further refinement should reduce it. HΦ will be integrated into node "
        "evaluation and action selection in TP-MCTS without restricting the set of legal "
        "actions."
    ),
    39: "Optional Extension: Policy Ranking",
    40: (
        "If state refinement proves too expensive, policies may also be enumerated in value "
        "order and each candidate checked with an STN. Dai and Goldsmith showed that the "
        "kth-best policy differs in one state's action from some preceding, better-ranked "
        "policy; in particular, the second-best policy differs in this way from an optimum "
        "[8]. This result can guide a heap of candidates, but STN filtering alone does not "
        "establish admissibility: sufficient enumeration, full-policy validation, and "
        "explicit treatment of branches and cycles are still required. Policy ranking will "
        "therefore be treated as an extension or ablation until its formal role is "
        "established."
    ),
    42: "Evaluation Plan",
    43: (
        "The evaluation will combine formal verification and experiments. For small "
        "instances, V* will be computed exactly and the inequality HΦ ≥ V* will be checked "
        "directly in every reachable state before and after each refinement. Purpose-built "
        "counterexamples will also be used, especially the m-doors family, in which an "
        "abstraction may ignore repeated trips and key-dependent costs and thereby create a "
        "polynomial gap. These tests are intended to expose uncontrolled optimism, hidden "
        "dependence on start effects, and invalid assumptions about transitions between "
        "abstract states."
    ),
    44: (
        "Experiments will use domains such as NASA Rover, Machine Shop, Hosting, and Stuck "
        "Car across a range of deadlines and probability settings. Comparisons will include "
        "TP-MCTS/PTRPG, PDB variants without refinement and with refinement, and full time "
        "representation on small instances. Measures will include bound tightness, PDB "
        "construction time and memory, the number of refinements and STN failures, online "
        "decision time, expected reward, success probability, completion before the "
        "deadline, and temporal feasibility. Results will report variance, confidence "
        "intervals, and failure rates rather than averages alone."
    ),
    45: "Expected Contributions",
    46: (
        "The expected contributions are: (1) a formal definition of an optimistic PDB for "
        "reward-maximizing probabilistic temporal planning; (2) explicit admissibility "
        "conditions for transition and reward abstraction; (3) an STN-based selective "
        "temporal-refinement algorithm with an anytime upper bound; (4) integration of the "
        "heuristic into TP-MCTS; and (5) a systematic evaluation of the trade-off among "
        "admissibility, bound tightness, and computational cost. Even if a particular "
        "refinement mechanism is replaced, the research framework and the criterion HΦ ≥ V* "
        "will remain stable."
    ),
    47: "References",
}


ALIGNMENTS = {
    0: "center",
    1: "center",
    2: "center",
    3: "center",
    5: "left",
    6: "both",
    7: "both",
    8: "left",
    9: "both",
    10: "both",
    11: "both",
    12: "both",
    14: "left",
    15: "both",
    16: "left",
    17: "left",
    18: "left",
    19: "left",
    20: "left",
    21: "both",
    22: "center",
    23: "both",
    24: "both",
    25: "left",
    26: "both",
    28: "left",
    29: "left",
    30: "both",
    31: "left",
    32: "both",
    33: "left",
    34: "both",
    35: "left",
    36: "both",
    37: "left",
    38: "both",
    39: "left",
    40: "both",
    42: "left",
    43: "both",
    44: "both",
    45: "left",
    46: "both",
    47: "left",
    48: "left",
    49: "left",
    50: "left",
    51: "left",
    52: "left",
    53: "left",
    54: "left",
    55: "left",
}


METADATA = [
    [("Student: ", "Eliezer Revach"), ("Supervisor: ", "Prof. Ronen I. Brafman")],
    [("Program: ", "Computer Science"), ("Student ID: ", "[TO COMPLETE]")],
    [("Email: ", "[TO COMPLETE]"), ("Date: ", "August 2026")],
]


CALLOUT = (
    "To be finalized with the supervisor before submission: ",
    "LAO* versus VI/MPI; the refinement representation (state copies, a temporal "
    "predicate, or a transition constraint); the initial reward model; and whether "
    "policy ranking is an optional extension or a central component.",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_ppr(paragraph):
    ppr = paragraph.find(w("pPr"))
    if ppr is None:
        ppr = etree.Element(w("pPr"))
        paragraph.insert(0, ppr)
    return ppr


def set_paragraph_layout(paragraph, alignment, rtl=False):
    ppr = ensure_ppr(paragraph)
    for child in list(ppr):
        if child.tag in {w("jc"), w("bidi")}:
            ppr.remove(child)
    jc = etree.Element(w("jc"))
    jc.set(w("val"), alignment)
    ppr.append(jc)
    bidi = etree.Element(w("bidi"))
    bidi.set(w("val"), "1" if rtl else "0")
    ppr.append(bidi)


def prepare_rpr(rpr, lang="en-US", rtl=False, remove_shading=False):
    if rpr is None:
        rpr = etree.Element(w("rPr"))
    else:
        rpr = copy.deepcopy(rpr)
    for child in list(rpr):
        if child.tag in {w("rtl"), w("cs")} or (
            remove_shading and child.tag in {w("shd"), w("highlight")}
        ):
            rpr.remove(child)
    rtl_node = etree.Element(w("rtl"))
    rtl_node.set(w("val"), "1" if rtl else "0")
    rpr.append(rtl_node)
    cs_node = etree.Element(w("cs"))
    cs_node.set(w("val"), "1" if rtl else "0")
    rpr.append(cs_node)
    lang_node = rpr.find(w("lang"))
    if lang_node is None:
        lang_node = etree.Element(w("lang"))
        rpr.append(lang_node)
    lang_node.set(w("val"), lang)
    lang_node.set(w("bidi"), lang)
    return rpr


def run_template(paragraph, index=0):
    runs = paragraph.findall(w("r"))
    if not runs:
        return None
    run = runs[min(index, len(runs) - 1)]
    return run.find(w("rPr"))


def clear_paragraph_content(paragraph):
    ppr = paragraph.find(w("pPr"))
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)


def append_run(paragraph, text, rpr=None, lang="en-US", rtl=False, remove_shading=False):
    run = etree.SubElement(paragraph, w("r"))
    final_rpr = prepare_rpr(rpr, lang=lang, rtl=rtl, remove_shading=remove_shading)
    if len(final_rpr) or final_rpr.attrib:
        run.append(final_rpr)
    text_node = etree.SubElement(run, w("t"))
    if text.startswith(" ") or text.endswith(" "):
        text_node.set(f"{{{XML_NS}}}space", "preserve")
    text_node.text = text


def replace_paragraph(paragraph, text, alignment, rtl=False):
    template = run_template(paragraph)
    clear_paragraph_content(paragraph)
    set_paragraph_layout(paragraph, alignment, rtl=rtl)
    append_run(
        paragraph,
        text,
        template,
        lang="he-IL" if rtl else "en-US",
        rtl=rtl,
    )


def replace_labeled_cell(cell, label, value, placeholder=False):
    paragraph = cell.find(w("p"))
    if paragraph is None:
        raise ValueError("Expected a paragraph in each table cell")
    label_template = run_template(paragraph, 0)
    value_template = run_template(paragraph, 1)
    clear_paragraph_content(paragraph)
    set_paragraph_layout(paragraph, "left", rtl=False)
    append_run(paragraph, label, label_template, remove_shading=True)
    append_run(
        paragraph,
        value,
        value_template,
        remove_shading=not placeholder,
    )


def replace_callout(cell, label, value):
    paragraph = cell.find(w("p"))
    label_template = run_template(paragraph, 0)
    value_template = run_template(paragraph, 1)
    clear_paragraph_content(paragraph)
    set_paragraph_layout(paragraph, "left", rtl=False)
    append_run(paragraph, label, label_template, remove_shading=True)
    append_run(paragraph, value, value_template, remove_shading=True)


def build(source, output):
    if source.resolve() == output.resolve():
        raise ValueError("Output path must differ from the retained source")
    if sha256(source) != EXPECTED_SOURCE_HASH:
        raise ValueError("The retained source no longer matches the distilled template")

    with zipfile.ZipFile(source, "r") as src:
        document_xml = src.read("word/document.xml")
        root = etree.fromstring(document_xml)
        numbering_root = etree.fromstring(src.read("word/numbering.xml"))
        body = root.find("w:body", NS)
        body_paragraphs = body.findall("w:p", NS)
        body_tables = body.findall("w:tbl", NS)
        if len(body_paragraphs) != 56 or len(body_tables) != 2:
            raise ValueError(
                f"Unexpected structure: {len(body_paragraphs)} paragraphs, "
                f"{len(body_tables)} tables"
            )

        for index, text in PARAGRAPHS.items():
            replace_paragraph(
                body_paragraphs[index],
                text,
                ALIGNMENTS[index],
                rtl=index == 2,
            )

        for index in range(48, 56):
            set_paragraph_layout(body_paragraphs[index], "left", rtl=False)
            for rpr in body_paragraphs[index].findall(".//w:rPr", NS):
                prepared = prepare_rpr(rpr, lang="en-US", rtl=False)
                rpr.getparent().replace(rpr, prepared)

        metadata_rows = body_tables[0].findall("w:tr", NS)
        for row_index, row in enumerate(metadata_rows):
            cells = row.findall("w:tc", NS)
            for cell_index, cell in enumerate(cells):
                label, value = METADATA[row_index][cell_index]
                replace_labeled_cell(
                    cell,
                    label,
                    value,
                    placeholder=value == "[TO COMPLETE]",
                )

        callout_cell = body_tables[1].find("w:tr/w:tc", NS)
        replace_callout(callout_cell, *CALLOUT)

        bullet_abstract = numbering_root.find(
            ".//w:abstractNum[@w:abstractNumId='91']", NS
        )
        if bullet_abstract is None:
            raise ValueError("Expected source bullet definition 91")
        bullet_level = bullet_abstract.find("w:lvl", NS)
        bullet_level.find("w:lvlJc", NS).set(w("val"), "left")
        bullet_ind = bullet_level.find("w:pPr/w:ind", NS)
        bullet_ind.attrib.pop(w("right"), None)
        bullet_ind.set(w("left"), "520")
        bullet_ind.set(w("hanging"), "260")

        final_xml = etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )
        final_numbering_xml = etree.tostring(
            numbering_root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w") as dst:
            for info in src.infolist():
                if info.filename == "word/document.xml":
                    data = final_xml
                elif info.filename == "word/numbering.xml":
                    data = final_numbering_xml
                else:
                    data = src.read(info.filename)
                dst.writestr(info, data)

    with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(output, "r") as dst:
        if src.namelist() != dst.namelist():
            raise ValueError("Package part inventory changed")
        for name in src.namelist():
            if name not in {"word/document.xml", "word/numbering.xml"} and src.read(name) != dst.read(name):
                raise ValueError(f"Preserve-only package part changed: {name}")

    with zipfile.ZipFile(output, "r") as dst:
        final_text = etree.fromstring(dst.read("word/document.xml"))
        paragraphs = final_text.find("w:body", NS).findall("w:p", NS)
        all_text = "\n".join("".join(p.itertext()) for p in paragraphs)
        allowed_hebrew = PARAGRAPHS[2]
        remainder = all_text.replace(allowed_hebrew, "")
        if re.search(r"[\u0590-\u05FF]", remainder):
            raise ValueError("Unexpected Hebrew text remains outside the required Hebrew title")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: build_english_proposal.py SOURCE.docx OUTPUT.docx")
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    build(source, output)
    print(output)


if __name__ == "__main__":
    main()
