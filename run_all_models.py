"""Tüm regresyon notebook'larını çalıştırır ve her yöntemin test R² değerini listeler.

Kullanım:  .venv/bin/python run_all_models.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent

NOTEBOOKS = {
    "Multiple Linear Regression": "multiple_linear_regression.ipynb",
    "Polynomial Regression": "polynomial_regression.ipynb",
    "Support Vector Regression": "support_vector_regression.ipynb",
    "Decision Tree Regression": "decision_tree_regression.ipynb",
    "Random Forest Regression": "random_forest_regression.ipynb",
}


def run_notebook(path: Path) -> float:
    """Notebook'u çalıştırır, son hücrenin (r2_score) çıktısını döndürür."""
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    last_code = [c for c in nb.cells if c.cell_type == "code"][-1]
    for out in last_code.outputs:
        if "data" in out and "text/plain" in out["data"]:
            return float(out["data"]["text/plain"])
    raise RuntimeError("R² çıktısı bulunamadı")


def main() -> None:
    results = []
    for name, file in NOTEBOOKS.items():
        print(f"Çalıştırılıyor: {name} ...", flush=True)
        try:
            results.append((name, run_notebook(HERE / file)))
        except Exception as e:  # bir model hata verirse diğerleri yine çalışsın
            print(f"  HATA: {e}")
            results.append((name, float("nan")))

    results.sort(key=lambda r: r[1], reverse=True)
    print(f"\n{'Yöntem':<30}{'R²':>10}")
    print("-" * 40)
    for name, r2 in results:
        print(f"{name:<30}{r2:>10.4f}")
    print(f"\nEn iyi model: {results[0][0]} (R² = {results[0][1]:.4f})")


if __name__ == "__main__":
    main()
