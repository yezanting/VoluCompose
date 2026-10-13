"""Verify a wheel contains the actual backend source, without build recursion."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("build"), "requires the build package")
class WheelContractTests(unittest.TestCase):
    def test_wheel_includes_exact_backend_source_and_licenses(self):
        with tempfile.TemporaryDirectory() as output:
            run = subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "--no-isolation",
                 "--outdir", output, str(ROOT / "medcompose")],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            wheel = next(Path(output).glob("*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                actual = {name for name in names if name.endswith(".py")}
                expected = {str(p.relative_to(ROOT)) for p in (ROOT / "medcompose").rglob("*.py")
                            if "build" not in p.parts}
                self.assertEqual(actual, expected)
                self.assertTrue(any(name.endswith("licenses/LICENSE") for name in names))
                self.assertTrue(any(name.endswith("licenses/LICENSES/Apache-2.0.txt") for name in names))


if __name__ == "__main__":
    unittest.main()
