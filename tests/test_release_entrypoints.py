"""CPU checks for release entrypoints; no model downloads or CUDA required."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Scalar:
    def float(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return 0.125


class RetentionMetadataTests(unittest.TestCase):
    def test_volucompose_reports_measured_retention_on_all_four_backbones(self):
        adapters = [
            ("Qwen2_5_VL/Qwen2_5_VL_hf.py", "Qwen2_5_VL", "llm", False),
            ("Qwen3_VL/Qwen3_VL_hf.py", "Qwen3VL", "llm", True),
            ("MedGemma/MedGemma.py", "MedGemma", "llm", True),
            ("HuluMed_Qwen2/HuluMed_Qwen2.py", "HuluMed_Qwen2", "model", False),
        ]
        # Execute each real adapter class without importing/loading its heavyweight
        # backend; only generation is substituted, to isolate metadata recording.
        for relative, name, attr, nested in adapters:
            with self.subTest(backbone=name):
                path = ROOT / "MedUniEval/models" / relative
                tree = ast.parse(path.read_text())
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
                namespace = {"tqdm": lambda values, **kwargs: values}
                exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
                model = namespace[name].__new__(namespace[name])
                model.methed = "volucompose"
                model.generate_output = lambda message: "A"
                backend = SimpleNamespace(compression_rate=Scalar(), selected_len=2)
                setattr(model, attr, SimpleNamespace(model=backend) if nested else backend)
                outputs, rates = model.generate_outputs([{"prompt": "question"}])
                self.assertEqual(outputs, ["A"])
                self.assertEqual(rates[0]["compose_rate"], 0.125)
                model.methed = "none"
                self.assertEqual(model.generate_outputs([{}])[1], [{"compose_rate": 1.0}])


class PackageImportTests(unittest.TestCase):
    def test_imports_do_not_require_every_model_backend(self):
        code = "import importlib,sys\n"
        for module in ["medcompose", "medcompose.volucompose", "medcompose.visionzip",
                       "medcompose.holitom", "medcompose.unicomp"]:
            code += f"importlib.import_module({module!r})\n"
        code += "assert 'torch' not in sys.modules\nassert 'transformers' not in sys.modules\n"
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


class LauncherTests(unittest.TestCase):
    def record(self, script, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            capture = tmp / "commands.jsonl"
            recorder = tmp / "python"
            recorder.write_text(
                f"#!{sys.executable}\nimport os,sys,json\n"
                "with open(os.environ['COMMAND_CAPTURE'],'a') as f:\n"
                " f.write(json.dumps({'args':sys.argv[1:],'gpu':os.getenv('CUDA_VISIBLE_DEVICES')})+'\\n')\n"
            )
            recorder.chmod(0o755)
            env = {**os.environ, "PATH": str(tmp) + os.pathsep + os.environ["PATH"],
                   "PYTHON": str(recorder), "COMMAND_CAPTURE": str(capture),
                   "MODEL_PATH": "/models/example", "DATASETS_PATH": "/datasets/example",
                   "OUTPUT_PATH": str(tmp / "outputs"), **overrides}
            run = subprocess.run(["bash", str(ROOT / "MedUniEval" / script)],
                                 cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            return [json.loads(line) for line in capture.read_text().splitlines()]

    def test_single_launcher_honors_overrides_and_pruning_method(self):
        calls = self.record("eval.sh", MODEL_NAME="Qwen3-VL", EVAL_DATASETS="M3D",
                            COMPRESSION="volucompose", CUDA_VISIBLE_DEVICES="2")
        self.assertEqual(len(calls), 1)
        args = calls[0]["args"]
        for key, value in [("--model_name", "Qwen3-VL"), ("--model_path", "/models/example"),
                           ("--datasets_path", "/datasets/example"),
                           ("--compression", "volucompose"), ("--eval_datasets", "M3D")]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertEqual(calls[0]["gpu"], "2")

    def test_chunked_launcher_has_sequential_shards_and_no_required_judge_key(self):
        calls = self.record("eval_M3D_chunked.sh", CHUNKS="3", CUDA_VISIBLE_DEVICES="0,2",
                            COMPRESSION="none", USE_LLM_JUDGE="False")
        self.assertEqual(len(calls), 3)
        for i, call in enumerate(calls):
            args = call["args"]
            self.assertEqual(args[args.index("--num_chunks") + 1], "3")
            self.assertEqual(args[args.index("--chunk_idx") + 1], str(i))
            self.assertEqual(args[args.index("--compression") + 1], "none")
            self.assertEqual(args[args.index("--use_llm_judge") + 1], "False")


if __name__ == "__main__":
    unittest.main()
