from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks" / "scirep_waveform_susceptibility_step9_colab.ipynb"
    output = root / "notebooks" / "step9_executed.ipynb"
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(notebook, timeout=3600, kernel_name="python3")
    client.execute(cwd=str(root.parent))
    nbformat.write(notebook, output)


if __name__ == "__main__":
    main()
