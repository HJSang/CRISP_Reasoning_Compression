"""Pure CPU tests for preserved reservations and outcome-independent selection."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_PATH = Path(__file__).resolve().parents[2] / "tools" / "opsd" / "prepare_math_cycle.py"
_SPEC = importlib.util.spec_from_file_location("prepare_math_cycle", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
partition_rows = _MODULE.partition_rows


class MathCycleSplitTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{"problem": str(i), "metadata": {"problem_id": str(i), "answer": str(i)}} for i in range(100)]
        self.options = dict(reserved=set(map(str, range(20))), final_test={"0", "1"}, prior_training={"20", "21"},
                            seed=137, validation_fraction=0.1, panel_size=8, screen_size=12)

    def ids(self, rows):
        return {row["metadata"]["problem_id"] for row in rows}

    def test_reservations_override_ratio_and_seal_final_test(self):
        pools = partition_rows(self.rows, **self.options)
        self.assertEqual(len(pools["validation"]), 20)
        self.assertTrue(self.options["reserved"].isdisjoint(self.ids(pools["train"])))
        self.assertTrue(self.options["final_test"] <= self.ids(pools["validation-sealed"]))
        self.assertEqual(self.ids(pools["validation"]), self.ids(pools["validation-tuning"]) | self.ids(pools["validation-sealed"]))
        self.assertTrue(self.ids(pools["validation-tuning"]).isdisjoint(self.ids(pools["validation-sealed"])))
        self.assertTrue(self.options["prior_training"].isdisjoint(self.ids(pools["screen"])))
        self.assertTrue(self.options["prior_training"] <= self.ids(pools["train"]))

    def test_selection_ignores_input_order_and_answers(self):
        first = partition_rows(self.rows, **self.options)
        changed = [{"problem": row["problem"], "metadata": dict(row["metadata"], answer="changed")} for row in reversed(self.rows)]
        second = partition_rows(changed, **self.options)
        for name in first:
            self.assertEqual([row["metadata"]["problem_id"] for row in first[name]], [row["metadata"]["problem_id"] for row in second[name]])

    def test_new_validation_excludes_prior_training(self):
        options = dict(self.options, reserved={"0"}, validation_fraction=0.3)
        pools = partition_rows(self.rows, **options)
        self.assertEqual(len(pools["validation"]), 30)
        self.assertTrue(options["prior_training"].isdisjoint(self.ids(pools["validation"])))

    def test_reserved_training_overlap_never_reclaimed_or_tuned(self):
        options = dict(self.options, prior_training={"0", "2", "20"})
        pools = partition_rows(self.rows, **options)
        self.assertTrue({"0", "2"} <= self.ids(pools["validation-sealed"]))
        self.assertTrue(options["prior_training"].isdisjoint(self.ids(pools["validation-tuning"])))

    def test_invalid_inputs_fail_closed(self):
        for rows, options in [(self.rows + [self.rows[0]], self.options),
                              (self.rows, dict(self.options, panel_size=100)),
                              (self.rows, dict(self.options, screen_size=100)),
                              (self.rows, dict(self.options, validation_fraction=1))]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                partition_rows(rows, **options)

    def test_reference_corrections_reject_conflicts_and_problem_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.jsonl", Path(directory) / "second.jsonl"
            row = {"problem": "Original problem", "metadata": {"source_row": 0, "solution": "First solution"}}
            first.write_text(json.dumps(row) + "\n")
            row["metadata"]["solution"] = "Conflicting solution"
            second.write_text(json.dumps(row) + "\n")
            specs = [{"path": str(path), "source_rows": [0]} for path in [first, second]]
            with self.assertRaisesRegex(ValueError, "Conflicting reference"):
                _MODULE._corrected_references(specs, [{"problem": "Original problem"}])
            with self.assertRaisesRegex(ValueError, "changed the source problem"):
                _MODULE._corrected_references(specs[:1], [{"problem": "Changed problem"}])

    @unittest.skipUnless(importlib.util.find_spec("datasketch"), "Optional data-preparation runtime")
    def test_duplicate_components_close_transitive_and_benchmark_edges(self):
        words = ["word" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(40)]
        rows = [{"problem": " ".join(words[:size])} for size in [30, 35, 40]]
        groups = _MODULE._components(rows, [])
        self.assertEqual(len(set(groups.values())), 1)
        rows = [{"problem": "Compute the sum of 3 and 7."}, {"problem": "Compute the sum of 8 and 9."}]
        groups = _MODULE._components(rows, [{"id": "heldout", "problem": rows[1]["problem"]}])
        self.assertEqual(set(groups.values()), {"benchmark:heldout"})


if __name__ == "__main__":
    unittest.main()
