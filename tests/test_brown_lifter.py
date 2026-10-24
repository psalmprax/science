"""Brown-lifter paper data: F/I constancy + Eq41 spot check (Ianconesu 2011)."""
import json
import math

PATH = "data/brown_lifter_tables.jsonl"


def _tables():
    with open(PATH) as fh:
        return {json.loads(line)["table"]: json.loads(line) for line in fh
                if line.strip()}


def test_table1_force_current_ratio_constant():
    t1 = _tables()[1]["rows"]
    ratios = [r["lift_g"] / r["I_mA"] for r in t1]
    mean = sum(ratios) / len(ratios)
    assert abs(mean - 18.4) < 0.5, (mean, ratios)
    assert (max(ratios) - min(ratios)) / mean < 0.05, ratios


def test_eq41_predicts_table2_point():
    eps0 = 8.8541878128e-12
    a, b, perim, V0 = 7.5e-5, 0.028, 0.6, 20500.0
    F = (4.792 * math.pi * eps0 * perim * V0 / b
         * (V0 / (1.011 * math.log(b / a) + 1.3035)
            - 3e6 * (a + 0.03 * math.sqrt(a))))
    assert abs(F / 9.80665 * 1000 - 10.0) / 10.0 < 0.15, F


def test_measured_table_noisier_than_calculated():
    t = _tables()
    calc = [r["lift_g"] / r["I_mA"] for r in t[1]["rows"]]
    meas = [r["lift_g"] / r["I_mA"] for r in t[2]["rows"]
            if r["lift_g"] is not None]
    import statistics
    assert statistics.pstdev(meas) > statistics.pstdev(calc)
