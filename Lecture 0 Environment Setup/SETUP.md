# Quick Environment Setup Guide

## Prerequisites
- Computer with admin access
- 5GB free disk space  
- Internet connection

## Step 1: Install Python 3.14 (or the latest version)

### Windows
1. Download Python 3.14 from: https://www.python.org/downloads/release/python-396/
   - Choose "Windows installer (64-bit)"
2. Run installer
   - **CHECK** "Add Python 3.14 to PATH" 
   - Click "Install Now"
3. Verify in Command Prompt:
   ```cmd
   python --version
   ```
   Should show: `Python 3.14`

### macOS
1. Download Python 3.14 from: https://www.python.org/downloads/release/python-396/
   - Choose "macOS 64-bit universal2 installer"
2. Run the .pkg installer
3. Verify in Terminal:
   ```bash
   python3 --version
   ```
   Should show: `Python 3.14`

### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip

# Verify
python3.9 --version
```

## Step 2: Create Virtual Environment

Open Terminal (Mac/Linux) or Command Prompt (Windows) and navigate to where you would like to put materials for this class. 

Run: 
```bash
git clone https://github.com/jinming99/learn-ml-by-building.git
```

 Then navigate to the project folder:

```bash
cd path/to/learn-ml-by-building
```

### Windows
```cmd
py -3.14 -m venv ml_lectures_env
ml_lectures_env\Scripts\activate
```
In case you get an error like "running scripts is not authorized on this system", run: 
```bash
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```


### macOS/Linux
```bash
python3.14 -m venv ml_lectures_env    # or: python3 -m venv ml_lectures_env
source ml_lectures_env/bin/activate
```

You should see `(ml_lectures_env)` in your terminal prompt.

## Step 3: Install Required Packages

With virtual environment activated:

```bash
# Upgrade pip first (use interpreter-scoped pip)
python -m pip install --upgrade pip

# Install all requirements (repo root requirements.txt at /Users/ming/Dropbox/learn-ml-by-building/requirements.txt)
python -m pip install -r requirements.txt
```

  Note: On macOS/Linux you may prefer `python3 -m pip ...`; both are fine when the environment is activated.
  
  This will take 5-15 minutes and download several hundred MB of packages. The requirements include deep learning dependencies `torch==2.2.2` and `torchvision==0.17.2`.

In case you get errors for torch or torchvision versions, please check the error for the versions that are available. Then go to requirements.txt and change the version and run the last step again. 

## Step 4: Launch Jupyter Notebook

Still in the activated virtual environment:

```bash
jupyter notebook
```

This will open your browser. Navigate to `00-Environment-Setup.ipynb` and open it.

### RISE slides and interactive widgets
- RISE is installed and enabled. In the classic Notebook toolbar, look for the "Enter/Exit RISE Slideshow" button.
- ipywidgets is installed and enabled. Test with a quick cell:
  
```python
import ipywidgets as widgets
widgets.IntSlider(description="Test")
```

If the RISE button is missing or widgets don't render, make sure you launched Jupyter from the activated environment so it uses the correct interpreter.

### Optional: Register this environment as a Jupyter kernel
This lets you pick the kernel by name inside Jupyter.

```bash
python -m ipykernel install --user --name ml_lectures_env --display-name "Python (ml_lectures_env)"
```

## Troubleshooting

### "python not recognized" (Windows)
- Reinstall Python and ensure "Add to PATH" is checked
- Or use `py -3.9` instead of `python`

### "jupyter: command not found"
- Make sure virtual environment is activated (you see `(ml_lectures_env)`)
- Reinstall with interpreter-scoped pip: `python -m pip install jupyter notebook`

### "ModuleNotFoundError: No module named 'torch'"
- Ensure you've installed the repo requirements: `python -m pip install -r requirements.txt`
- Or install directly: `python -m pip install torch torchvision`
- Restart the Notebook kernel after installation (Kernel → Restart Kernel) and re-run your cells.

### Permission errors
- Avoid using `sudo` with pip. Ensure your virtual environment is activated and try again.
- Windows: Run Command Prompt as Administrator if needed for system operations.

### Virtual environment not activating
- Windows PowerShell: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Mac/Linux: `chmod +x ml_lectures_env/bin/activate`

## Daily Workflow

Every time you work on the course:

```bash
# 1. Navigate to project
cd path/to/learn-ml-by-building

# 2. Activate environment
source ml_lectures_env/bin/activate     # Mac/Linux
ml_lectures_env\Scripts\activate       # Windows

# 3. Start Jupyter
jupyter notebook

# 4. When done, deactivate
deactivate
```

---

**Once Jupyter opens, continue with the `00-Environment-Setup.ipynb` notebook!**
