import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_seed = load_script("build_seed_corpus", "build_seed_corpus.py")
evaluate = load_script("evaluate_predictions", "evaluate_predictions.py")


class DataPipelineTests(unittest.TestCase):
    def test_normalize_uses_nfc_and_collapses_whitespace(self):
        self.assertEqual(build_seed.normalize("  A\u0301   sentence  "), "Á sentence")

    def test_target_script_gate_accepts_olchiki_and_rejects_latin(self):
        self.assertTrue(build_seed.OL_CHIKI.search("ᱥᱟᱱᱛᱟᱲᱤ"))
        self.assertIsNone(build_seed.OL_CHIKI.search("Santali"))

    def test_fallback_metrics_are_perfect_for_identical_pairs(self):
        refs = ["ᱥᱟᱱᱛᱟᱲᱤ", "ᱡᱚᱦᱟᱨ"]
        self.assertEqual(evaluate.exact_match(refs, refs), 1.0)
        self.assertEqual(evaluate.char_f1(refs[0], refs[0]), 1.0)
        self.assertEqual(evaluate.bleu_fallback(refs, refs), 100.0)

    def test_prepare_input_csv_is_traceable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            path.write_text("English,Santali\nHello,ᱡᱚᱦᱟᱨ\n", encoding="utf-8")
            self.assertEqual(path.read_text(encoding="utf-8").count("ᱡᱚᱦᱟᱨ"), 1)


if __name__ == "__main__":
    unittest.main()
