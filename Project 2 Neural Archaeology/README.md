# Neural Archaeology: Decoding and Rewiring Language Model Representations

**ECE4424/CS4824: Machine Learning, Fall 2025**

## Quick Start

You can run this project either on **Google Colab** (recommended for beginners) or **locally** (recommended if you have a GPU). Choose the setup guide below that matches your preference.

---

## Option 1: Google Colab Setup (Recommended for Beginners)

### Step 1: Upload Project Files to Colab

1. **Download the project package** from Canvas/course website
2. **Extract the ZIP file** to get the project folder
3. **Upload to Google Drive** (optional but recommended for persistence)
4. **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
5. **Upload the notebook**: `File > Upload notebook` → Select `Neural_Archaeology_Student.ipynb`

### Step 2: Upload Data Files

The project includes a `data/` folder with required datasets. The notebook uses
`joy.json` for the joy category; it contains the same scenarios as the original
`happiness.json`, which is retained for compatibility. If an older project package
is missing `joy.json`, copy `data/emotions/happiness.json` to
`data/emotions/joy.json` in your working data folder and rerun the emotion-loading
cell.

Upload the data folder to Colab:

```python
# Run this cell to upload the data folder
from google.colab import files
import zipfile
import os

# Option A: Upload data as ZIP (easier)
print("Please upload the 'data.zip' file from the project package")
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/')
        print(f"✓ Extracted {filename}")

# Verify
if os.path.exists('/content/data/safety/circuit_breakers_train.json'):
    print("✓ Data uploaded successfully!")
else:
    print("✗ Data upload failed. Please check the file structure.")
```

**Alternative**: If you saved the project to Google Drive, mount it:
```python
from google.colab import drive
drive.mount('/content/drive')
# Then navigate to your project folder
```

### Step 3: Enable GPU & Get HuggingFace Token

**Enable GPU for faster processing:**
1. Click `Runtime > Change runtime type`
2. Set `Hardware accelerator` to **GPU** (T4 recommended)
3. Click `Save` (runtime will restart)

**Get HuggingFace Token** (required for model download):
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Read** access
3. Copy the token (starts with `hf_...`)
4. Run this cell in the notebook:

```python
from huggingface_hub import login

# Paste your token here
login(token="YOUR_HF_TOKEN_HERE")
```

**Note**: The notebook will automatically download the SmolLM-1.7B-Instruct model (~3.4 GB) when you run the setup cells. This may take 5-10 minutes on first run.

---

## Option 2: Local Setup (Recommended if You Have a GPU)

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv neural_env
source neural_env/bin/activate  # On Windows: neural_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download Model

**Option A: Using the provided script (easiest)**
```bash
python download_models.py
```
This will prompt you for your HuggingFace token and download SmolLM-1.7B-Instruct to `./models/`

**Option B: Let the notebook auto-download**
The notebook will automatically download the model when first run. It will be cached in `~/.cache/huggingface/hub/`

### Step 3: Run the Notebook

```bash
# Start Jupyter
jupyter notebook Neural_Archaeology_Student.ipynb

# Or use JupyterLab
jupyter lab Neural_Archaeology_Student.ipynb
```

---

### Step 3: Verify Complete Setup

Run this verification cell before starting the assignment:
```python
import os
import torch

def verify_setup():
    """Comprehensive setup verification"""
    checks = []

    # 1) GPU check
    checks.append(f"GPU: {'OK' if torch.cuda.is_available() else 'MISSING'}")
    if torch.cuda.is_available():
        checks.append(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 2) Data files
    required_files = {
        'data/safety/circuit_breakers_train.json': 'Safety training data',
        'data/safety/circuit_breakers_val.json': 'Safety validation data',
        'data/emotions/joy.json': 'Joy scenarios',
        'data/emotions/sadness.json': 'Sadness scenarios',
        'data/emotions/anger.json': 'Anger scenarios',
        'data/emotions/fear.json': 'Fear scenarios'
    }
    for path, desc in required_files.items():
        checks.append(f"{desc}: {'FOUND' if os.path.exists(path) else 'MISSING'} ({path})")

    # 3) Directories
    dirs = ['models', 'results/visualizations', 'results/evaluations', 
            'results/saved_states', 'results/student_analysis']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        checks.append(f"Dir ready: {d}")

    # 4) Package versions
    import transformers, sklearn
    checks.append(f"Transformers: {transformers.__version__}")
    checks.append(f"Scikit-learn: {sklearn.__version__}")
    checks.append(f"PyTorch: {torch.__version__}")

    print("SETUP VERIFICATION\n" + "="*50)
    for c in checks:
        print("-", c)

verify_setup()
```

### Step 4: Expected File Structure

After setup, your Colab environment should have:
```
/content/
├── learn-ml-by-building/           # Cloned repo (if using Option A)
│   └── Project 2 Neural Archaeology/
│       └── data/                   # Original data location
├── data/                           # Working data directory
│   ├── safety/
│   │   ├── circuit_breakers_train.json  (~2.5MB)
│   │   └── circuit_breakers_val.json    (~500KB)
│   └── emotions/
│       ├── joy.json      (~100KB)
│       ├── sadness.json  (~100KB)
│       ├── anger.json    (~100KB)
│       └── fear.json     (~100KB)
├── models/                         # Your saved models
├── results/
│   ├── visualizations/            # Generated plots
│   ├── evaluations/               # Metrics/scores
│   ├── saved_states/              # Pickled objects
│   └── student_analysis/          # YOUR ANSWERS
└── neural_archaeology.ipynb        # This notebook
```

### Step 5: Memory & Runtime Management

__Monitor Resources__
```bash
!nvidia-smi  # GPU memory
```
```python
import psutil
print(f"RAM: {psutil.virtual_memory().percent}% used")
```
```bash
!df -h /content  # Disk space
```

__If you run out of memory__
```python
import gc, torch
gc.collect(); torch.cuda.empty_cache()
```
Last resort: Runtime > Restart runtime (then rerun setup cells)

### Common Setup Issues

- **"No such file or directory: data/safety/..."** — Data not prepared. Re-run Step 1.
- **"CUDA out of memory"** — Reduce batch_size to 4 or 8.
- **"Model download interrupted"** — Runtime > Restart runtime, then retry.
- **"Can't import transformers"** — `!pip install transformers accelerate`
- **Files disappear after restart** — Normal for Colab; re-run setup cells.

### Quick Test After Setup

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, os

# Test model loading (use local path if downloaded)
model_path = "./models/SmolLM-1.7B-Instruct"  # or "HuggingFaceTB/SmolLM-1.7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path)
print(f"✓ Model loaded: {model.config.num_hidden_layers} layers")

with open('data/safety/circuit_breakers_train.json', 'r') as f:
    data = json.load(f)
print(f"✓ Safety data: {len(data)} examples")

with open('data/emotions/joy.json', 'r') as f:
    joy = json.load(f)
print(f"✓ Emotion data: {len(joy)} joy examples")

print("\n🎉 Setup complete! Proceed to Section 0")
```

---

## Submission Guidelines

### What to Submit

Submit **ONE PDF file** containing your completed notebook with all outputs visible.

**Required Components:**
1. **All code cells executed** - Show your implementations and outputs
2. **All visualizations displayed** - Plots must be visible in the PDF
3. **All analysis sections completed** - Answer all "Required Analysis" questions
4. **Code implementations** - Complete all `# YOUR CODE HERE` sections

### How to Generate the PDF

**From Jupyter Notebook:**
```bash
# Method 1: Using nbconvert
jupyter nbconvert --to pdf Neural_Archaeology_Student.ipynb

# Method 2: Print from browser
# File > Print Preview > Save as PDF
```

**From Google Colab:**
```
File > Print > Save as PDF
```

**Important**: Make sure all cells are executed and outputs are visible before generating the PDF!

### Grading Rubric

| Component | Points | Description |
|-----------|--------|-------------|
| **Question 1: Pooling Strategies** | 15 | Implementation + analysis of pooling methods |
| **Question 2: PCA Implementation** | 20 | From-scratch PCA with correctness verification |
| **Question 3: K-Means Clustering** | 20 | Clustering implementation + emotion analysis |
| **Question 4: Safety Detection** | 20 | Logistic regression classifier + evaluation |
| **Question 5: Layer Analysis** | 15 | Layer-wise performance comparison |
| **Question 6: Adversarial Testing** | 10 | Robustness evaluation + insights |
| **Code Quality & Documentation** | 10 | Clean code, comments, proper execution |
| **Visualizations** | 10 | Clear, well-labeled plots |
| **Written Analysis** | 10 | Thoughtful responses to analysis questions |
| **BONUS: Creative Extensions** | +10 | Novel experiments or insights |
| **Total** | **130** | (120 base + 10 bonus) |

### Academic Integrity

- You may discuss concepts with classmates, but **all code and analysis must be your own**
- Do not share code or solutions
- Cite any external resources you use (beyond course materials)
- Using AI assistants (ChatGPT, Copilot, etc.) is allowed for **debugging only**, not for generating solutions

---

## References

- **Representation Engineering** - [Zou et al., 2023](https://arxiv.org/abs/2310.01405)
- **Circuit Breaker** - [Zou et al., 2024](https://www.circuit-breaker.ai/)
- **Backtracking for Safety** - [Zhang et al., 2024](https://arxiv.org/abs/2409.14586), [Sel et al., 2025](https://arxiv.org/abs/2503.08919)
