"""Rule-based gating network for LocalMind's tiered expert system.

This is a rule-based gating function inspired by Mixture-of-Experts routing. It
is NOT a learned/trained gate — it uses the existing heuristic classifier plus
a hard rule for image inputs. Real MoE gates are learned end-to-end; this
approximates the routing BEHAVIOR (sparse, input-dependent expert selection)
using interpretable rules instead.

Concretely, the gate has two responsibilities:

* ``decompose`` — split an incoming query into at most four sub-tasks, using
  Llama 3.2 as the decomposition model (with a cheap heuristic short-circuit for
  obviously-simple queries and graceful fallback when the model's output cannot
  be parsed).
* ``gate_score`` — assign a single sub-task to exactly one expert. Image-bearing
  sub-tasks are *hard-routed* to the vision expert (LLaVA); everything else is
  scored by the unchanged heuristic classifier and mapped to one of the three
  text experts.

Only the selected expert per sub-task is executed — the sparse-activation
property that the MoE pattern is known for — but the selection here is a
transparent rule, not a learned function.
"""

from __future__ import annotations

import json
import logging
import re

from classifier import score_complexity, score_privacy
from models import llama32_client

logger = logging.getLogger("localmind.gate")

# Upper bound on how many sub-tasks a single query may be split into.
MAX_SUBTASKS = 4

# JSON schema constraining Llama 3.2's decomposition output (Ollama structured
# outputs). Forcing a top-level array of {subtask, depends_on_image} objects —
# together with temperature 0 — makes decomposition deterministic and reliably
# parseable, which a free-text prompt to a 3B model is not.
_DECOMPOSITION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "subtask": {"type": "string"},
            "depends_on_image": {"type": "boolean"},
        },
        "required": ["subtask", "depends_on_image"],
    },
}

# A query is treated as trivially simple — and decomposition is skipped — when
# it is short AND contains none of these multi-step cue words.
SIMPLE_WORD_LIMIT = 15
MULTI_STEP_CUES = ("and", "also", "compare", "then")

# Recursive decomposition: a sub-task that still reads as a compound request is
# re-decomposed one level deeper so each leaf is a single ask. Recursion is
# bounded two ways — a depth cap and a leaf cap — so it always terminates and
# never explodes the number of experts run. The conjunction cues below (note:
# deliberately *not* "compare" — an analytical "compare X to Y" is a single ask
# that must stay whole, see ANALYSIS_VERBS) are what mark a sub-task as worth a
# second decomposition pass.
MAX_DEPTH = 2
MAX_LEAVES = 6
RECURSION_CUES = ("and", "also", "then", "as well as")

# Complexity cut points mapping a non-image sub-task to a text expert.
LOW_COMPLEXITY = 0.3
HIGH_COMPLEXITY = 0.6
# Above this privacy score, a sub-task is handled by at least the general-purpose
# expert rather than the fastest/weakest one.
PRIVACY_THRESHOLD = 0.6

# Comparison/analysis verbs signal a reasoning-heavy ask. The length-weighted
# complexity heuristic under-scores such asks once decomposition isolates a short
# clause (e.g. "Compare X to Y" is only a few words), so when any of these verbs
# is present the gate adds a fixed boost to the complexity score before
# thresholding. Cheap, fully explainable, and reported in the routing reasoning.
ANALYSIS_VERBS = ("compare", "analyze", "analyse", "evaluate", "contrast", "tradeoff", "trade-off")
ANALYSIS_BOOST = 0.3


def _is_simple(query: str) -> bool:
    """Return True if a query is short and shows no multi-step cue words.

    Used to skip the (relatively expensive) Llama 3.2 decomposition call for
    obviously-trivial queries so they stay fast.
    """
    words = query.split()
    if len(words) >= SIMPLE_WORD_LIMIT:
        return False
    lowered = query.lower()
    return not any(
        re.search(rf"\b{re.escape(cue)}\b", lowered) for cue in MULTI_STEP_CUES
    )


def _has_recursion_cue(text: str) -> bool:
    """Return True if a sub-task still reads as a compound (multi-ask) request.

    Used to decide whether a sub-task is worth a second decomposition pass.
    Matches whole-word conjunction cues only (see ``RECURSION_CUES``); an
    analytical ask like "compare X to Y" intentionally has no cue here so it is
    left whole rather than atomised into sub-definitions.
    """
    lowered = text.lower()
    return any(
        re.search(rf"\b{re.escape(cue)}\b", lowered) for cue in RECURSION_CUES
    )


def _has_analysis_verb(text: str) -> bool:
    """Return True if a sub-task contains a comparison/analysis verb.

    Such a sub-task is a single reasoning-heavy ask (e.g. "compare X to Y and
    evaluate the tradeoffs") and must stay whole: the conjunction inside it joins
    two facets of one analysis, not two independent requests. Recursive
    decomposition keys off ``"and"`` (see ``RECURSION_CUES``), which would
    otherwise shatter such an ask into trivial fragments — observed live — so the
    recursion guard uses this to leave analytical sub-tasks intact. Mirrors the
    boost detection in ``gate_score`` (substring match on ``ANALYSIS_VERBS``).
    """
    lowered = text.lower()
    return any(verb in lowered for verb in ANALYSIS_VERBS)


def _decomposition_prompt(query: str, has_image: bool) -> str:
    """Build the few-shot decomposition instruction for Llama 3.2.

    Pairs with ``_DECOMPOSITION_SCHEMA`` (which guarantees a valid JSON array);
    this prompt's job is the *semantics* — capture every distinct ask without
    dropping any, and keep an analytical ask whole rather than atomising it.
    """
    image_note = (
        " The user has also attached an image; set \"depends_on_image\": true "
        "for any sub-task that requires looking at the image."
        if has_image
        else ""
    )
    return (
        "Break the user's request into every distinct, independent ask it "
        "contains, as a JSON array. Rules: include EVERY separate question the "
        'user asks (a request joined by "and"/"also" usually contains two or '
        "more) and never drop one; keep a single analytical ask (one "
        "comparison, explanation, or definition) as ONE element with its full "
        "original wording, never split into sub-definitions or solution steps. "
        f"At most {MAX_SUBTASKS} elements. Each element has keys \"subtask\" (a "
        'string) and "depends_on_image" (a boolean).'
        f"{image_note}\n\n"
        'Example request: "Summarize this article and translate the summary '
        'into French"\n'
        'Example output: [{"subtask": "Summarize this article", '
        '"depends_on_image": false}, {"subtask": "Translate the summary into '
        'French", "depends_on_image": false}]\n\n'
        f"User request: {query}"
    )


def _extract_subtasks(raw: str) -> list[dict] | None:
    """Parse a Llama 3.2 response into a validated sub-task list.

    Tolerates surrounding prose or code fences by extracting the substring
    between the first ``[`` and the last ``]`` before parsing. Returns a list of
    ``{"subtask": str, "depends_on_image": bool}`` dicts, or ``None`` if the
    text cannot be parsed into that shape.
    """
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None

    subtasks: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict) or "subtask" not in item:
            return None
        text = str(item["subtask"]).strip()
        if not text:
            return None
        subtasks.append(
            {
                "subtask": text,
                "depends_on_image": bool(item.get("depends_on_image", False)),
            }
        )
    return subtasks


def decompose(query: str, has_image: bool, _depth: int = 0) -> list[dict]:
    """Decompose a query into a list of sub-tasks via the Llama 3.2 gate.

    Returns a list of ``{"subtask": str, "depends_on_image": bool, "depth": int}``
    dicts, capped at ``MAX_SUBTASKS`` per level and ``MAX_LEAVES`` overall.
    Behaviour:

    * Trivially-simple queries (see ``_is_simple``) skip the model call entirely
      and return a single sub-task equal to the original query.
    * Otherwise Llama 3.2 is asked to produce strict JSON; its output is parsed
      with one retry. If it still cannot be parsed, the function degrades
      gracefully to a single sub-task equal to the original query (a warning is
      logged; it never raises).
    * When ``has_image`` is True, at least one returned sub-task is guaranteed to
      carry ``depends_on_image=True`` — if the model produced none, an explicit
      image sub-task is appended.
    * **Analytical asks are kept whole.** A query carrying a comparison/analysis
      verb (see ``_has_analysis_verb``) is a single reasoning ask and skips the
      model split entirely — the 3B gate is unreliable here and tends to shred
      one analytical sentence into syntactic fragments (e.g. "Compare X" / "to
      Y" / "and evaluate the tradeoffs"), which the prompt cannot reliably
      prevent. Keeping it whole lets ``gate_score`` route the intact ask (with
      its analysis boost) to the reasoning expert. The trade-off: a query that
      genuinely mixes an analytical ask with unrelated ones stays whole and goes
      to the reasoning expert, which can handle the multi-part request.
    * **Recursive decomposition.** Any sub-task that still reads as a compound
      request (see ``_has_recursion_cue``) is decomposed one level deeper, so
      each leaf is a single ask. ``_depth`` tracks recursion and is recorded on
      every leaf; recursion is bounded by ``MAX_DEPTH`` (how deep) and
      ``MAX_LEAVES`` (total leaves), so it always terminates. ``_depth`` is an
      internal parameter — callers pass only ``query`` and ``has_image``.
    """
    if _is_simple(query) or _has_analysis_verb(query):
        return [{"subtask": query, "depends_on_image": has_image, "depth": _depth}]

    prompt = _decomposition_prompt(query, has_image)
    subtasks: list[dict] | None = None
    for attempt in range(2):
        # Temperature 0 + the JSON-array schema make the gate deterministic and
        # its output reliably parseable.
        result = llama32_client.generate(
            prompt,
            options={"temperature": 0},
            response_format=_DECOMPOSITION_SCHEMA,
        )
        if "error" in result:
            logger.warning(
                "Gate decomposition call failed (attempt %d): %s",
                attempt + 1,
                result["error"],
            )
            continue
        subtasks = _extract_subtasks(result.get("response", ""))
        if subtasks is not None:
            break
        logger.warning(
            "Gate decomposition output was not valid JSON (attempt %d).",
            attempt + 1,
        )

    if subtasks is None:
        logger.warning(
            "Gate decomposition failed after retry; falling back to a single "
            "sub-task equal to the original query."
        )
        return [{"subtask": query, "depends_on_image": has_image, "depth": _depth}]

    subtasks = subtasks[:MAX_SUBTASKS]

    if not has_image:
        # No image was supplied, so nothing can legitimately depend on one. The
        # decomposition model occasionally flags depends_on_image anyway; clear
        # it so those sub-tasks are scored normally instead of hard-routed to
        # the vision expert.
        for st in subtasks:
            st["depends_on_image"] = False

    if has_image and not any(st["depends_on_image"] for st in subtasks):
        # Guarantee the image is actually examined. Keep within MAX_SUBTASKS by
        # making room for the explicit image sub-task if the list is already full.
        if len(subtasks) >= MAX_SUBTASKS:
            subtasks = subtasks[: MAX_SUBTASKS - 1]
        subtasks.append(
            {
                "subtask": "describe and analyze the provided image",
                "depends_on_image": True,
            }
        )

    # Recursive pass: re-decompose any sub-task that still looks compound, one
    # level deeper. Bounded by MAX_DEPTH (we only recurse while the *next* depth
    # stays under the cap) and MAX_LEAVES (the running total of leaves). Three
    # kinds of sub-task are never recursed: image sub-tasks (an atomic
    # hard-route), a sub-task identical to the parent query (would not make
    # progress), and an analytical ask carrying a comparison/analysis verb (its
    # internal "and" joins facets of one analysis, not separate requests — see
    # _has_analysis_verb).
    leaves: list[dict] = []
    for st in subtasks:
        st["depth"] = _depth
        can_recurse = (
            _depth + 1 < MAX_DEPTH
            and not st["depends_on_image"]
            and _has_recursion_cue(st["subtask"])
            and not _has_analysis_verb(st["subtask"])
            and st["subtask"].strip().lower() != query.strip().lower()
            and len(leaves) < MAX_LEAVES
        )
        if can_recurse:
            children = decompose(st["subtask"], False, _depth + 1)
            # Only accept the split if it actually produced more than one leaf;
            # otherwise keep the sub-task whole.
            if len(children) > 1:
                leaves.extend(children)
                continue
        leaves.append(st)

    return leaves[:MAX_LEAVES]


def gate_score(subtask: str, depends_on_image: bool) -> dict:
    """Assign a single sub-task to exactly one expert.

    Image-bearing sub-tasks are hard-routed to the vision expert (``llava``)
    regardless of any complexity/privacy score — this is a hard rule the
    heuristic classifier cannot override. All other sub-tasks are scored by the
    unchanged ``classifier`` heuristics; if the sub-task contains a
    comparison/analysis verb (see ``ANALYSIS_VERBS``) a fixed ``ANALYSIS_BOOST``
    is added to the complexity *before* thresholding, then it is mapped to a
    text expert:

    * ``complexity < 0.3``                          → ``llama3.2``
    * ``0.3 <= complexity < 0.6``                   → ``mistral``
    * ``complexity >= 0.6``                         → ``deepseek-r1:7b``
    * ``privacy > 0.6`` and ``complexity < 0.6``    → ``mistral`` (privacy
      override: sensitive content deserves better handling than the weakest
      model, even when complexity is low)

    The returned ``complexity`` is the *effective* (post-boost) score used for
    the decision; when a boost was applied the ``reasoning`` shows the base
    score, the matched verbs, and the boost so the routing stays explainable.
    Returns ``{"expert", "complexity", "privacy", "reasoning", "hard_routed"}``;
    ``hard_routed`` is True only for the image case.
    """
    base_complexity = round(score_complexity(subtask), 3)
    privacy = round(score_privacy(subtask), 3)

    matched_verbs = [v for v in ANALYSIS_VERBS if v in subtask.lower()]
    boost = ANALYSIS_BOOST if matched_verbs else 0.0
    complexity = round(min(1.0, base_complexity + boost), 3)
    boost_note = (
        f" (base {base_complexity:.2f} + {ANALYSIS_BOOST:.2f} analysis-verb boost "
        f"for {matched_verbs})"
        if boost
        else ""
    )

    if depends_on_image:
        return {
            "expert": "llava",
            "complexity": complexity,
            "privacy": privacy,
            "reasoning": (
                "Sub-task depends on an image, so it is hard-routed to the LLaVA "
                "vision expert regardless of its complexity/privacy score."
            ),
            "hard_routed": True,
        }

    if complexity >= HIGH_COMPLEXITY:
        expert = "deepseek-r1:7b"
        reasoning = (
            f"High complexity ({complexity:.2f} >= {HIGH_COMPLEXITY:.2f}){boost_note}: "
            f"routed to the DeepSeek R1 reasoning expert for multi-step logic."
        )
    elif privacy > PRIVACY_THRESHOLD:
        expert = "mistral"
        reasoning = (
            f"Privacy-sensitive ({privacy:.2f} > {PRIVACY_THRESHOLD:.2f}) at "
            f"complexity {complexity:.2f}{boost_note}: routed to the Mistral "
            f"general-purpose expert rather than the fastest/weakest model."
        )
    elif complexity < LOW_COMPLEXITY:
        expert = "llama3.2"
        reasoning = (
            f"Low complexity ({complexity:.2f} < {LOW_COMPLEXITY:.2f}){boost_note}: "
            f"routed to the fast Llama 3.2 expert."
        )
    else:
        expert = "mistral"
        reasoning = (
            f"Moderate complexity ({LOW_COMPLEXITY:.2f} <= {complexity:.2f} < "
            f"{HIGH_COMPLEXITY:.2f}){boost_note}: routed to the Mistral "
            f"general-purpose expert."
        )

    return {
        "expert": expert,
        "complexity": complexity,
        "privacy": privacy,
        "reasoning": reasoning,
        "hard_routed": False,
    }
