"""Python oracle for von-rs unit tests that need no model weights.

Writes tests/fixtures/python_oracle.json with the exact outputs of the Python
behaviours von-rs must reproduce: float repr, round(), str.strip(), str repr,
_format_state, pydantic model_dump shapes, calibration map handling, the
independent-options attention masks and position ids, split_digits, the
Choice/Score confidence metric, and structured (object/array) instructions.
Floats are carried as IEEE-754 bit patterns (hex) so nothing is lost in JSON.

Usage (from the repo root):
    VON_PY_SRC=bug-fix-fork-von/src uv run python von-rs/tools/export_pyfixtures.py

VON_PY_SRC selects the Python sources to read (default: ../../src, the main
checkout). Point it at a tree that is in sync with upstream master.
"""

import json
import math
import os
import random
import struct
import sys
import warnings

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.environ.get("VON_PY_SRC") or os.path.join(HERE, "..", "..", "src"))

from von.backends.option_marker_backend import (  # noqa: E402
    OptionMarkerBackend,
    _format_state,
    _margin_confidence,
    _validate_calibration_map,
    _validate_noul_prior,
)
from von.models.option_marker import (  # noqa: E402
    build_independent_option_masks,
    build_option_invariant_position_ids,
    split_digits,
)
from von.types import (  # noqa: E402
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

OUT = os.path.join(HERE, "..", "tests", "fixtures", "python_oracle.json")


def bits(x: float) -> str:
    return struct.pack(">d", x).hex()


def float_cases() -> list[float]:
    rng = random.Random(1234)
    fixed = [
        0.0, -0.0, 1.0, -1.0, 0.1, 0.5, 1.5, 2.5, 0.37, 1e-4, 9.999e-5, 1e-5, 1.5e-5, 123456.789,
        1e15, 9999999999999998.0, 1e16, 1.2345e16, 1e22, 1e100, 5e-324, 1.7976931348623157e308,
        0.1 + 0.2, 1 / 3, 2 / 3, math.pi, -math.e, 100.0, 1e2, 0.125, 0.375, 0.0625, 2.675, 1.005,
        0.00001234, 0.0001234, 12345678901234567.0, 3.0e-7, 0.30000000000000004,
    ]
    rand = [rng.uniform(-1, 1) * 10 ** rng.randint(-12, 20) for _ in range(150)]
    return fixed + rand


def round_cases() -> list[tuple[float, int]]:
    rng = random.Random(99)
    ties = [0.5, 1.5, 2.5, -0.5, 0.125, 0.375, 0.00005, 0.00015, 0.00025, 0.0045, 0.0055, 2.675, 1.005, 0.285, 0.5555]
    out = [(x, n) for x in ties for n in (0, 1, 2, 3, 4)]
    # Probabilities as they come out of fp32 softmax, widened to f64.
    for _ in range(300):
        p = struct.unpack("f", struct.pack("f", rng.random()))[0]
        out.append((p, rng.choice((2, 3, 4))))
    # Exact 4-dp and 3-dp boundaries in f32.
    for k in range(0, 10001, 37):
        x = struct.unpack("f", struct.pack("f", k / 10000 + 0.00005))[0]
        out.append((x, 4))
    return out


STRIP_CASES = [
    "  hello  ", "\t\n x \r\n", "\x1c\x1d x \x1e\x1f", "\x85x\x85", "\u00a0x\u00a0", "\u2003x\u3000",
    "\u200bx\u200b", "\ufeffx", "", "   ", "no-strip", "\x0bv\x0c",
]

REPR_CASES = [
    "plain", "it's", 'say "hi"', "both ' and \"", "back\\slash", "tab\there", "new\nline", "cr\rret",
    "bell\x07", "del\x7f", "é ü ß", "📦 emoji", "\u00a0nbsp", "\u200bzwsp", "\u2028ls", "\ufeffbom",
    "\u00adshy", "\ue000pua", "\x85nel", "", "'", '"', "\\", "mixed 'q' \"q\" \\ \n",
]

STATE_CASES = [
    '"just a string"',
    '{"a": 1, "b": "two", "c": 3.0, "d": true, "e": false, "f": null}',
    '{"nested": {"k": "v", "n": [1, 2.5, "x", null, true]}, "empty_list": [], "empty_obj": {}}',
    '{"quote": "it\'s", "list": ["it\'s", "say \\"hi\\""], "unicode": "caf\\u00e9 \\ud83d\\udce6"}',
    '{"floats": [1e-05, 1e16, 1e+22, 0.1, 100.0, -0.0, 123456789012345.6]}',
    '{"big": 12345678901234567890, "neg": -42, "zero": 0}',
    '[1, "two", {"three": 3}]',
    '42', '3.5', 'true', 'null', '""',
    '{"multi\\nline key": "multi\\nline value"}',
    '{"order_z": 1, "order_a": 2, "order_m": 3}',
]


def dumps(model) -> str:
    return json.dumps(model.model_dump(), ensure_ascii=False)


def model_dumps() -> list[dict]:
    items = {
        "noul_min": Noul(instructions="Is it down?"),
        "noul_criteria": Noul(instructions="Is it down?", criteria={"true": "Down", "false": "Up"}),
        "choice": Choice(instructions="Pick", criteria={"b": "Bee", "a": None, "c": "Sea"}),
        "score": Score(instructions="Rate", criteria=["Low", {"what": "High", "examples": ["x", "y"]}, {"what": "Mid"}]),
        "noul_answer": NoulAnswer(noul=0.8123),
        "choice_answer": ChoiceAnswer(choice="b", probabilities={"b": 0.9, "a": 0.05, "c": 0.05}, confidence=0.85),
        "score_answer": ScoreAnswer(score=2.0, confidence=0.5, legend={"0": "Low", "1": "High"}, probabilities={"0": 0.25, "1": 0.75}),
        "response": SystemOneResponse(
            model="von-1.1.0",
            answers={"z": NoulAnswer(noul=1.0), "a": ChoiceAnswer(choice="x", probabilities={"x": 1.0}, confidence=1.0)},
            usage=Usage(input_tokens=12, output_tokens=2),
        ),
    }
    return [{"name": k, "json": dumps(v)} for k, v in items.items()]


CALIB_CASES = [
    None, 5, [], {}, {"lo": 0.1}, {"bias": 1}, {"bias": "1.5", "entropy": True},
    {"bias": None}, {"bias": [1]}, {"bias": 0.2, "lo": 3, "hi": 2}, {"bias": 0.2, "lo": 0, "hi": 60},
    {"bias": 0.2056, "entropy": -3.255, "log_tokens": 20.2391, "n_options": -5.2263, "lo": 0.3, "hi": 12.0},
    {"entropy": 1.0, "extra": 7}, {"n_options": " 2.5 "}, {"bias": "inf"}, {"bias": "nan"}, {"bias": "1_0"},
]


def calib_cases() -> list[dict]:
    out = []
    for raw in CALIB_CASES:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            got = _validate_calibration_map(raw)
        out.append({
            "raw": raw,
            "valid": None if got is None else {k: bits(v) for k, v in got.items()},
            "warned": any(issubclass(x.category, UserWarning) for x in w),
        })
    return out


class _FakeTok:
    def __init__(self, n: int):
        self.n = n

    def encode(self, text, add_special_tokens=False):
        return [0] * self.n


def temperature_cases() -> list[dict]:
    import torch

    rng = random.Random(7)
    real_map = _validate_calibration_map(
        {"bias": 0.2056, "entropy": -3.255, "log_tokens": 20.2391, "n_options": -5.2263, "lo": 0.3, "hi": 12.0})
    out = []
    backend = OptionMarkerBackend.__new__(OptionMarkerBackend)
    for i in range(60):
        k = rng.choice((1, 2, 3, 4, 8, 10))
        logits = [struct.unpack("f", struct.pack("f", rng.uniform(-10, 10)))[0] for _ in range(k)]
        tokens = rng.choice((0, 1, 5, 40, 300, 2000, 8000))
        override = 1.7 if i % 15 == 0 else None
        calib = None if i % 10 == 9 else real_map
        backend._calib_map = calib
        backend._default_temp = 2.2
        t = backend._effective_temperature(torch.tensor(logits, dtype=torch.float32), "s", k, _FakeTok(tokens), override)
        out.append({
            "logits": [bits(x) for x in logits], "n_options": k, "state_tokens": tokens,
            "override": override, "calibrated": calib is not None, "temperature": bits(float(t)),
        })
    return out


NOUL_PRIOR_CASES = [
    None, 1, [], "a", {}, {"a": 1}, {"b": 1}, {"a": 0.5, "b": 0.1}, {"a": "0.5", "b": " 2 "},
    {"a": True, "b": False}, {"a": None, "b": 1}, {"a": "x", "b": 1}, {"a": "1_0", "b": "inf"},
    {"a": "nan", "b": 0}, {"a": 1, "b": 2, "extra": 3}, {"a": [1], "b": 1}, {"a": "1__0", "b": 1},
]


def noul_prior_cases() -> list[dict]:
    out = []
    for raw in NOUL_PRIOR_CASES:
        got = _validate_noul_prior(raw)
        out.append({"raw": raw, "valid": None if got is None else {k: bits(v) for k, v in got.items()}})
    return out


def noul_correction_cases() -> list[dict]:
    """The zero-shot correction exactly as evaluate_noul computes it: a Python
    float times a 0-d float32 tensor, plus a Python float."""
    import torch

    rng = random.Random(11)
    out = []
    for _ in range(40):
        a, b = rng.uniform(-2, 2), rng.uniform(-3, 3)
        l0, l1 = (struct.unpack("f", struct.pack("f", rng.uniform(-12, 12)))[0] for _ in range(2))
        null = torch.tensor([l0, l1], dtype=torch.float32)
        bias = null[0] - null[1]
        out.append({
            "a": bits(a), "b": bits(b), "null": [bits(l0), bits(l1)],
            "fitted": bits(float(a * bias + b)), "default": bits(float(0.7 * bias)),
        })
    return out


def preset_dumps() -> dict:
    from von.presets import email_preset, moderation_preset, security_preset, triage_preset

    def dump(preset):
        return {qid: q.model_dump() for qid, q in preset.items()}

    return {
        "triage": dump(triage_preset()),
        "email": dump(email_preset()),
        "email_empty_custom": dump(email_preset({})),
        "email_custom": dump(email_preset({"ops": "Operations", "hr": "People team"})),
        "moderation": dump(moderation_preset()),
        "security": dump(security_preset()),
    }


def mask_cases() -> list[dict]:
    """Independent-options masks and position ids on small synthetic sequences.

    Each case is one unpadded sequence of `seq_len` tokens ending in [SEP], with
    [MASK] tokens at `mask_positions`. Masks are flattened row-major (row = query)
    as strings of 0/1.
    """
    import torch

    rng = random.Random(7)
    shapes = [(9, [3, 5], 2), (6, [1], None), (4, [], 1), (3, [1], 1)]
    for _ in range(12):
        seq_len = rng.randint(4, 24)
        k = rng.randint(1, min(5, seq_len - 2))
        positions = sorted(rng.sample(range(1, seq_len - 1), k))
        shapes.append((seq_len, positions, rng.choice([None, 1, 2, 3, 5])))

    def flat(mask) -> str:
        return "".join("1" if v else "0" for v in mask.flatten().tolist())

    out = []
    for seq_len, positions, window in shapes:
        input_ids = torch.zeros((1, seq_len), dtype=torch.long)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        pos_ids = build_option_invariant_position_ids(input_ids, attention_mask, [positions])
        masks = build_independent_option_masks(
            input_ids, attention_mask, [positions], pos_ids, window
        )
        out.append({
            "seq_len": seq_len,
            "mask_positions": positions,
            "sliding_window": window,
            "position_ids": pos_ids[0].tolist(),
            "full": flat(masks["full_attention"][0, 0]),
            "sliding": flat(masks["sliding_attention"][0, 0]),
        })
    return out


def confidence_cases() -> list[list[float]]:
    """Probability vectors as they come out of fp32 softmax, widened to f64."""
    rng = random.Random(16)
    f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]  # noqa: E731
    fixed = [[], [1.0], [0.5, 0.5], [1.0, 0.0], [0.7, 0.3], [0.286, 0.363, 0.351],
             [1 / 3, 1 / 3, 1 / 3], [0.25] * 4, [0.2, 0.2, 0.6], [0.0, 0.0, 1.0]]
    out = [[f32(p) for p in probs] for probs in fixed]
    for _ in range(300):
        n = rng.randint(2, 8)
        logits = [rng.gauss(0, rng.choice((0.1, 1.0, 4.0))) for _ in range(n)]
        m = max(logits)
        exps = [f32(math.exp(x - m)) for x in logits]
        total = sum(exps)
        out.append([f32(e / total) for e in exps])
    return out


def digit_cases() -> list[str]:
    """Texts for split_digits: ASCII runs, separators, and Unicode decimal digits
    (category Nd, including astral ones) next to digit-like characters that are
    not Nd (superscripts, Roman numerals, CJK numerals), which must stay as-is.
    Only characters from Unicode 15.0 or earlier, so any Python 3.12+ agrees."""
    arabic_indic = "".join(chr(0x0660 + d) for d in (1, 2, 3))
    devanagari = chr(0x0967) + chr(0x0968)
    fullwidth = chr(0xFF11) + chr(0xFF10)
    math_bold = chr(0x1D7CF) + chr(0x1D7D0)  # MATHEMATICAL BOLD DIGIT ONE, TWO
    adlam = chr(0x1E951) + chr(0x1E952)
    kawi = chr(0x11F51) + chr(0x11F52)  # Unicode 15.0
    superscript_two = chr(0x00B2)
    roman_eight = chr(0x2167)
    cjk_three = chr(0x4E09)
    return [
        "", "no digits here", "7", "2026", "a1b22c333", "3.14159", "-42", "1,000,000",
        "x2026y", "12 34", "007", "v1.2.3", "In 2026 the fee rose from 500 to 692.",
        "[CLS] Q? 42 kg [SEP] [MASK] 40 kg [MASK] over 40",
        "Is 9 > 10? [SEP] [MASK] Yes, condition holds true. [MASK] No, condition is false.",
        arabic_indic, "1" + arabic_indic + "4", devanagari, fullwidth, math_bold, adlam, kawi,
        "x" + superscript_two + "y", "10" + superscript_two, roman_eight + "8", cjk_three + "3",
        "tab\t12\nnew 34",
    ]


def instruction_cases() -> list[dict]:
    """Questions whose `instructions` is not a plain string, and the model_dump
    pydantic makes of them (None when it rejects the question)."""
    snow, clef = chr(0x2603), chr(0x1D11E)
    values = [
        "plain text", "", {}, [], {"task": "Does the customer want money back?"},
        ["step one", "step two"],
        {"z": 1, "a": {"y": [3, {"k2": None, "k1": True}], "b": False}, "m": -0.0},
        [{"z": 1, "a": 2}, ["x", {"d": 1, "c": 2}]],
        {"B": 1, "a": 2, "_": 3, "10": 4, "9": 5, "é": 6, snow: 7, clef: 8, "": 9},
        {"floats": [0.1, 1e-05, 1e16, 1.5e300, 3.0, 12, -7]},
        {"text": "café " + snow + " " + clef + ' "quoted" \\ tab\t nl\n ctl' + chr(1) + chr(0x7F)},
        [None, True, False, 0, [], {}],
        5, 2.5, True, None,
    ]
    out = []
    for qtype, model, extra in (("noul", Noul, {}), ("choice", Choice, {"criteria": {"a": None}}),
                                ("score", Score, {"criteria": ["low", "high"]})):
        for v in values:
            q = {"type": qtype, "instructions": v, **extra}
            try:
                dump = dumps(model(**q))
            except ValueError:
                dump = None
            out.append({"json": json.dumps(q, ensure_ascii=False), "out": dump})
    return out


# `independent_options` values as the backend reads them: bool(cdata.get(...)).
FLAG_CASES = ['true', 'false', 'null', '0', '1', '0.0', '2.5', '""', '"no"', '[]', '[0]', '{}', '{"a": 1}']


def main() -> None:
    fixture = {
        "float_repr": [{"x": bits(x), "repr": repr(x)} for x in float_cases()],
        "round": [{"x": bits(x), "n": n, "out": bits(round(x, n))} for x, n in round_cases()],
        "strip": [{"s": s, "out": s.strip()} for s in STRIP_CASES],
        "str_repr": [{"s": s, "out": repr(s)} for s in REPR_CASES],
        "format_state": [{"json": j, "out": _format_state(json.loads(j))} for j in STATE_CASES],
        "models": model_dumps(),
        "calibration_maps": calib_cases(),
        "temperatures": temperature_cases(),
        "noul_priors": noul_prior_cases(),
        "noul_corrections": noul_correction_cases(),
        "presets": preset_dumps(),
        "independent_masks": mask_cases(),
        "margin_confidence": [
            {"probs": [bits(p) for p in probs], "out": bits(_margin_confidence(probs))}
            for probs in confidence_cases()
        ],
        "split_digits": [{"s": s, "out": split_digits(s)} for s in digit_cases()],
        "structured_instructions": instruction_cases(),
        "independent_flags": [{"json": j, "out": bool(json.loads(j))} for j in FLAG_CASES],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT}: " + ", ".join(f"{k}={len(v)}" for k, v in fixture.items()))


if __name__ == "__main__":
    main()
