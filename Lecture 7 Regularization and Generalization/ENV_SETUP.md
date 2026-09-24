# Lecture 7 environment setup

Use Python 3.11 and a dedicated environment. Run these commands from the
`Lecture 7 Regularization and Generalization` directory.

## macOS / Linux

```bash
python3.11 -m venv regularization_env
source regularization_env/bin/activate
```

## Windows PowerShell

```powershell
py -3.11 -m venv regularization_env
.\regularization_env\Scripts\Activate.ps1
```

## Install and register the kernel

With the environment activated, on any platform:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name lecture7-regularization --display-name "Python (Lecture 7 Regularization)"
```

Open the notebook in your existing Jupyter interface or notebook editor and
select **Python (Lecture 7 Regularization)**. If you need a browser interface,
install and launch Jupyter Notebook in this environment:

```bash
python -m pip install notebook
jupyter notebook
```

## Tested runtime

The lecture-specific requirements record the installed versions used for the
complete CPU execution on macOS with Python **3.11.13**:

| Package | Version |
| --- | --- |
| TensorFlow | 2.16.2 |
| Keras | 3.15.1 |
| NumPy | 1.26.4 |
| pandas | 3.0.1 |
| Matplotlib | 3.10.8 |
| scikit-learn | 1.8.0 |
| IPython | 9.17.1 |
| ipykernel | 7.3.0 |
| Jinja2 | 3.1.6 |

The notebook downloads Fashion-MNIST through Keras on first use. It needs no
private files or separate figure assets. Other operating systems and accelerator
backends may produce different numerical results; they were not part of this
CPU verification. Jupyter Notebook is an optional interface, not part of the recorded
execution environment. Colab uses the notebook's existing setup cell and was
not part of this local verification.
