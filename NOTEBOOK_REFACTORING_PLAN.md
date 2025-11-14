# Comprehensive Notebook Refactoring Plan
## Plotting Code and Presentation Style Analysis

**Date:** 2025-11-14
**Repository:** learn-ml-by-building
**Scope:** 17 Jupyter notebooks analyzed (~60,000 lines of code)

---

## Executive Summary

### Key Findings
- **182 figure/subplot creations** across all notebooks
- **582 inline font styling parameters** (repeated styling code)
- **15+ notebooks** using identical style configuration
- **Extensive code duplication** in plotting setup and configuration
- **Inconsistent patterns** in a few notebooks (PCA uses different style)
- **Significant opportunity** for consolidation into reusable utilities

### Expected Impact
- **Code Reduction:** ~30% fewer lines by eliminating duplication
- **Consistency:** 100% style uniformity across all notebooks
- **Maintainability:** Single source of truth for plotting functions
- **Readability:** Shorter cells, clearer logical structure

---

## Part 1: Common Patterns Identified

### 1.1 Universal Style Configuration
**Found in 15+ notebooks:**
```python
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
```

**Files:**
- 02-KNN.ipynb (line 103)
- 04-Optimization-GD.ipynb (line 161)
- 05-ProbClass.ipynb (line 100)
- 07-Overfitting-Regularization.ipynb (line 100)
- 08-DecisionTrees-ModernView.ipynb (line 124)
- 09-Ensemble-Methods.ipynb (line 97)
- 10-GP-Kernel-Methods.ipynb
- 11-KMeans.ipynb
- 13-FCNeuralNet.ipynb (line 77)
- 14-Transformers.ipynb (line 182)
- 15-CNN.ipynb (line 65)
- 16_RNN.ipynb (line 81)

**Exception:** 12-PCA.ipynb uses `sns.set_style('whitegrid')` - inconsistent!

### 1.2 Repeated Subplot Creation Patterns
**Pattern found 144 times:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
# or
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
```

### 1.3 Excessive Inline Font Styling
**Pattern found 582 times:**
```python
ax.set_title('Title', fontsize=15, fontweight='bold')
ax.set_xlabel('X Label', fontsize=13, fontweight='bold')
ax.set_ylabel('Y Label', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
```

**Heaviest usage:**
- 10-GP-Kernel-Methods.ipynb (extensive throughout)
- 14-Transformers.ipynb (almost every plot)
- 06-ModelEval.ipynb (12 figures)
- 13-FCNeuralNet.ipynb (16 subplots)

---

## Part 2: Notebook-Specific Findings

### High-Impact Notebooks

#### 06-ModelEval.ipynb ⭐ BEST PRACTICES
**Location:** `Lecture 6 Evaluation Pitfalls and Data Visualization/`

**Strengths:**
- ✅ Already has reusable `plot_model_comparison()` helper function (lines 38-95)
- ✅ Centralized color schemes
- **This notebook shows the RIGHT approach!**

**Issues:**
- 12 separate figure creations with repeated `figsize=(15, 10)`
- Simpson's Paradox section (lines 200-350): 150-line cell needs breaking up
- Datasaurus visualization (lines 130-170): complex grid creation

**Refactoring Priority:** HIGH (use as template for others)

---

#### 10-GP-Kernel-Methods.ipynb ⚠️ MOST COMPLEX
**Location:** `Lecture 10 Kernel Methods/`

**Issues:**
- Most complex plotting in entire repository
- Custom `sample_gp_prior()` function buried in plotting code
- Extensive subplot usage without helper functions
- Very long cells mixing computation and visualization

**Refactoring Priority:** HIGH (highest code reduction potential)

**Example Complexity:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.set_xlabel('Year', fontsize=13, fontweight='bold')
ax1.set_title('Gaussian Process: Smooth + Honest Uncertainty',
              fontsize=15, fontweight='bold')
# ... repeated many times
```

---

#### 13-FCNeuralNet.ipynb
**Location:** `Lecture 13 NN Architecture/`

**Issues:**
- 16 subplot creations with repeated patterns
- 40+ line `visualize_single_neuron()` function
- Some cells exceed 100 lines
- Training loop output uses print statements (should use progress bars)

**Refactoring Priority:** MEDIUM

---

#### 14-Transformers.ipynb
**Location:** `Lecture 14 Transformers/`

**Strengths:**
- ✅ Excellent markdown structure (KEEP THIS!)
- ✅ Good educational progression

**Issues:**
- Heavy inline styling throughout
- Feature extraction + visualization in same cells

**Refactoring Priority:** MEDIUM

---

#### 17_llm_agents.ipynb
**Location:** `Lecture 17 LLM Agents/`

**Issues:**
- Uses older subplot syntax: `plt.subplot(1, 2, 1)` instead of modern `plt.subplots()`
- Should modernize for consistency

**Refactoring Priority:** LOW (but easy fix)

---

### Other Notebooks
All remaining notebooks (02, 03, 04, 05, 07, 08, 09, 11, 12, 15, 16) follow similar patterns:
- Same style configuration duplication
- Repeated subplot patterns
- Inline font styling
- **Refactoring Priority:** MEDIUM (batch refactor with shared utilities)

---

## Part 3: Proposed Solution

### 3.1 Create Shared Plotting Utilities Module

**File:** `ml_plotting_utils.py`

```python
"""
Shared plotting utilities for ML educational notebooks
Provides consistent styling and reduces code duplication
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_plotting_style():
    """Apply consistent plotting style across all notebooks"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    sns.set_palette("husl")

def style_axes(ax, title, xlabel='', ylabel='',
               title_size=15, label_size=13, grid=True):
    """
    Apply consistent styling to axes

    Args:
        ax: Matplotlib axes object
        title: Plot title
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
        title_size: Font size for title
        label_size: Font size for axis labels
        grid: Whether to show grid
    """
    ax.set_title(title, fontsize=title_size, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size, fontweight='bold')
    if grid:
        ax.grid(True, alpha=0.3)

def create_comparison_plot(n_plots=2, figsize_per_plot=(8, 5),
                          layout='horizontal'):
    """
    Create side-by-side or stacked comparison plots

    Args:
        n_plots: Number of subplots
        figsize_per_plot: Size of each individual subplot
        layout: 'horizontal' or 'vertical'

    Returns:
        fig, axes tuple
    """
    if layout == 'horizontal':
        figsize = (figsize_per_plot[0] * n_plots, figsize_per_plot[1])
        return plt.subplots(1, n_plots, figsize=figsize)
    else:  # vertical
        figsize = (figsize_per_plot[0], figsize_per_plot[1] * n_plots)
        return plt.subplots(n_plots, 1, figsize=figsize)

def create_grid_plot(rows, cols, figsize_per_cell=(5, 4)):
    """
    Create grid of subplots with automatic sizing

    Args:
        rows: Number of rows
        cols: Number of columns
        figsize_per_cell: Size of each cell

    Returns:
        fig, flattened axes array
    """
    figsize = (figsize_per_cell[0] * cols, figsize_per_cell[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    return fig, np.array(axes).flatten() if rows * cols > 1 else np.array([axes])

def plot_training_curves(history_dict, metrics=['loss', 'accuracy'],
                        split_plots=True):
    """
    Plot training history for multiple metrics

    Args:
        history_dict: Dictionary with keys like 'train_loss', 'val_loss', etc.
        metrics: List of metric names to plot
        split_plots: If True, separate plot per metric; else combined

    Returns:
        fig, axes tuple
    """
    n_metrics = len(metrics)
    if split_plots:
        fig, axes = create_comparison_plot(n_metrics)
        axes = [axes] if n_metrics == 1 else axes
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax] * n_metrics

    for ax, metric in zip(axes, metrics):
        for split in ['train', 'val', 'test']:
            key = f'{split}_{metric}'
            if key in history_dict:
                ax.plot(history_dict[key], label=split.capitalize(), linewidth=2)

        style_axes(ax, f'{metric.capitalize()} Over Time',
                  'Epoch', metric.capitalize())
        ax.legend()

    plt.tight_layout()
    return fig, axes

def plot_confusion_matrix_styled(cm, classes, ax=None, title='Confusion Matrix'):
    """
    Plot confusion matrix with consistent styling

    Args:
        cm: Confusion matrix array
        classes: List of class names
        ax: Axes to plot on (creates new if None)
        title: Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    style_axes(ax, title, 'Predicted', 'Actual', grid=False)

    return ax
```

### 3.2 Domain-Specific Utilities

**File:** `gp_plotting_utils.py` (for Gaussian Processes)

```python
"""Gaussian Process specific plotting utilities"""
import numpy as np
import matplotlib.pyplot as plt
from ml_plotting_utils import style_axes

def sample_gp_prior(kernel, X, n_samples=5, noise=1e-10):
    """
    Sample functions from GP prior

    Args:
        kernel: Kernel function
        X: Input points
        n_samples: Number of samples to draw
        noise: Numerical stability term

    Returns:
        Array of shape (len(X), n_samples)
    """
    K = kernel(X, X)
    L = np.linalg.cholesky(K + noise * np.eye(len(K)))
    samples = L @ np.random.randn(len(X), n_samples)
    return samples

def plot_gp_samples(X, samples, ax=None, title='GP Prior Samples',
                   xlabel='Input', ylabel='Output'):
    """
    Visualize samples from Gaussian Process

    Args:
        X: Input points
        samples: Samples array (n_points, n_samples)
        ax: Axes to plot on
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(samples.shape[1]):
        ax.plot(X, samples[:, i], alpha=0.6, linewidth=1.5)

    style_axes(ax, title, xlabel, ylabel)
    return ax

def plot_gp_posterior(X_train, y_train, X_test, mean, std, samples=None, ax=None):
    """
    Plot GP posterior with uncertainty

    Args:
        X_train: Training inputs
        y_train: Training outputs
        X_test: Test inputs
        mean: Posterior mean
        std: Posterior standard deviation
        samples: Optional posterior samples to plot
        ax: Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot training data
    ax.scatter(X_train, y_train, c='red', marker='x', s=100,
               label='Training Data', zorder=3)

    # Plot posterior mean
    ax.plot(X_test, mean, 'b-', linewidth=2, label='Posterior Mean')

    # Plot uncertainty
    ax.fill_between(X_test.ravel(),
                    mean - 2*std, mean + 2*std,
                    alpha=0.3, label='95% Confidence')

    # Plot samples if provided
    if samples is not None:
        for i in range(samples.shape[1]):
            ax.plot(X_test, samples[:, i], 'b-', alpha=0.2, linewidth=1)

    style_axes(ax, 'Gaussian Process Posterior', 'Input', 'Output')
    ax.legend()
    return ax
```

### 3.3 Standard Import Pattern

**Add to first code cell of every notebook:**

```python
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML plotting utilities
from ml_plotting_utils import (
    setup_plotting_style,
    style_axes,
    create_comparison_plot,
    create_grid_plot,
    plot_training_curves
)

# Apply consistent style
setup_plotting_style()
```

---

## Part 4: Before/After Comparisons

### Example 1: Basic Subplot Creation

**BEFORE (16 lines):**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1
ax1.plot(epochs, train_loss)
ax1.set_title('Training Loss', fontsize=15, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2
ax2.plot(epochs, train_acc)
ax2.set_title('Training Accuracy', fontsize=15, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
```

**AFTER (6 lines):**
```python
fig, (ax1, ax2) = create_comparison_plot(2, figsize_per_plot=(8, 5))

ax1.plot(epochs, train_loss)
style_axes(ax1, 'Training Loss', 'Epoch', 'Loss')

ax2.plot(epochs, train_acc)
style_axes(ax2, 'Training Accuracy', 'Epoch', 'Accuracy')
```

**Reduction:** 62.5% fewer lines

---

### Example 2: Training History Plot

**BEFORE (20+ lines):**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Validation')
ax1.set_title('Model Loss', fontsize=15, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history['train_accuracy'], label='Train')
ax2.plot(history['val_accuracy'], label='Validation')
ax2.set_title('Model Accuracy', fontsize=15, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
```

**AFTER (1 line):**
```python
plot_training_curves(history, metrics=['loss', 'accuracy'])
```

**Reduction:** 95% fewer lines!

---

### Example 3: Grid of Subplots

**BEFORE:**
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    # ... plotting code ...
    ax.set_title(f'Iteration {i}', fontsize=15, fontweight='bold')
    ax.set_xlabel('X', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
```

**AFTER:**
```python
fig, axes = create_grid_plot(2, 3, figsize_per_cell=(5, 3.5))

for i, ax in enumerate(axes):
    # ... plotting code ...
    style_axes(ax, f'Iteration {i}', 'X', 'Y')
```

---

## Part 5: Presentation Style Improvements

### 5.1 Cell Length Guidelines

**Problems Identified:**
- `06-ModelEval.ipynb`: 150-line cells (Simpson's Paradox)
- `10-GP-Kernel-Methods.ipynb`: 100+ line cells mixing theory and code
- `13-FCNeuralNet.ipynb`: Long training loops with verbose output

**New Standard:**
- **Maximum 50 lines per code cell**
- **Separate concerns:** Data loading → Processing → Visualization
- **Use markdown headers** between logical sections

**Example Structure:**
```markdown
### Model Training

**Load Data:**
[Code cell: data loading]

**Train Model:**
[Code cell: training loop with progress bar]

**Visualize Results:**
[Code cell: plotting only]
```

### 5.2 Import Organization

**Current:** Imports scattered throughout notebooks
**New Standard:** Consolidate at top

```python
# Cell 1: Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: ML Libraries
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cell 3: Custom Utilities
from ml_plotting_utils import setup_plotting_style, style_axes
setup_plotting_style()

# Cell 4: Configuration
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
```

### 5.3 Training Loop Improvements

**Replace verbose print statements with progress bars:**

**BEFORE:**
```python
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # ... training ...
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')
```

**AFTER:**
```python
from tqdm import tqdm

for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    for data, target in pbar:
        # ... training ...
        pbar.set_postfix({'loss': f'{loss:.4f}'})
```

### 5.4 Standardize Style Configuration

**Issue:** `12-PCA.ipynb` uses `sns.set_style('whitegrid')` while others use `'seaborn-v0_8-darkgrid'`

**Action:** Standardize to `seaborn-v0_8-darkgrid` OR document why different

---

## Part 6: Implementation Plan

### Phase 1: Foundation (Week 1)
**Priority: CRITICAL**

1. **Create `ml_plotting_utils.py`**
   - Implement all utility functions
   - Add comprehensive docstrings
   - Create unit tests
   - **Estimated time:** 4 hours

2. **Create `gp_plotting_utils.py`**
   - Extract GP-specific functions
   - Test with 10-GP-Kernel-Methods.ipynb
   - **Estimated time:** 2 hours

3. **Test utilities with one notebook**
   - Use 06-ModelEval.ipynb as pilot
   - Verify all functions work correctly
   - **Estimated time:** 2 hours

**Deliverables:**
- `ml_plotting_utils.py` with tests
- `gp_plotting_utils.py` with tests
- Documentation/README for utilities

---

### Phase 2: High-Priority Notebooks (Week 2)
**Priority: HIGH**

4. **Refactor 06-ModelEval.ipynb**
   - Already has good patterns to build on
   - Break up 150-line Simpson's Paradox cell
   - Use as template for others
   - **Estimated time:** 3 hours

5. **Refactor 10-GP-Kernel-Methods.ipynb**
   - Highest complexity reduction potential
   - Extract GP utilities
   - Break up long cells
   - **Estimated time:** 4 hours

6. **Refactor 13-FCNeuralNet.ipynb**
   - Simplify `visualize_single_neuron()`
   - Add progress bars to training
   - Break cells <50 lines
   - **Estimated time:** 3 hours

**Deliverables:**
- 3 fully refactored notebooks
- Documented patterns for team

---

### Phase 3: Medium-Priority Notebooks (Week 3)
**Priority: MEDIUM**

7. **Batch refactor remaining notebooks:**
   - 02-KNN.ipynb
   - 03-Linear-Regression.ipynb
   - 04-Optimization-GD.ipynb
   - 05-ProbClass.ipynb
   - 07-Overfitting-Regularization.ipynb
   - 08-DecisionTrees-ModernView.ipynb
   - 09-Ensemble-Methods.ipynb
   - 11-KMeans.ipynb
   - 14-Transformers.ipynb
   - 15-CNN.ipynb
   - 16_RNN.ipynb

   **Per notebook:**
   - Replace style config with utility import
   - Replace repeated subplot patterns
   - Replace inline styling with `style_axes()`
   - **Estimated time:** 1.5 hours each = 16.5 hours total

**Deliverables:**
- 11 refactored notebooks
- Consistent style across all

---

### Phase 4: Polish & Documentation (Week 4)
**Priority: LOW**

8. **Fix 12-PCA.ipynb style inconsistency**
   - Standardize to `seaborn-v0_8-darkgrid`
   - **Estimated time:** 0.5 hours

9. **Modernize 17_llm_agents.ipynb**
   - Replace old `plt.subplot()` syntax
   - Apply standard refactoring
   - **Estimated time:** 2 hours

10. **Create documentation:**
    - Usage guide for `ml_plotting_utils.py`
    - Style guide for future notebooks
    - Before/after examples
    - **Estimated time:** 3 hours

11. **Final consistency pass:**
    - Verify all imports at top
    - Check all cells <50 lines
    - Ensure markdown structure
    - **Estimated time:** 4 hours

**Deliverables:**
- Complete refactored notebook suite
- Comprehensive documentation
- Style guide for future development

---

## Part 7: Success Metrics

### Quantitative Metrics
- **Code reduction:** Target 30% fewer lines (baseline: ~60,000 lines)
- **Consistency:** 100% of notebooks use shared utilities
- **Cell length:** 95%+ of cells under 50 lines
- **Import consolidation:** All imports in first 2-3 cells

### Qualitative Metrics
- **Readability:** Easier for students to follow
- **Maintainability:** Single source for plotting changes
- **Extensibility:** Easy to add new plot types
- **Teaching clarity:** Code doesn't obscure concepts

---

## Part 8: Risk Assessment

### Low Risk
- ✅ Creating utility modules (no existing code affected)
- ✅ Refactoring individual notebooks (isolated changes)
- ✅ Breaking up long cells (improves clarity)

### Medium Risk
- ⚠️ Changing imports (could break if utilities have bugs)
  - **Mitigation:** Thorough testing of utilities first
- ⚠️ Standardizing PCA style (visual changes)
  - **Mitigation:** Verify with instructor/team first

### Considerations
- Notebooks should still run top-to-bottom without errors
- Educational content and explanations must remain intact
- Visualizations should look the same (or better)
- Students should benefit from cleaner code

---

## Part 9: Team Coordination

### Roles Needed
1. **Developer:** Implement utilities and refactor notebooks
2. **Reviewer:** Verify educational content preserved
3. **Tester:** Run all notebooks end-to-end
4. **Documenter:** Create usage guides

### Review Checklist (per notebook)
- [ ] Runs without errors from top to bottom
- [ ] All visualizations render correctly
- [ ] Educational explanations intact
- [ ] Code is more concise than before
- [ ] Imports organized at top
- [ ] Cells under 50 lines (except where necessary)
- [ ] Uses shared utilities where applicable

---

## Part 10: Future Enhancements

### Beyond Initial Refactoring
1. **Interactive plots:** Consider plotly for some visualizations
2. **Animation utilities:** Standardize training animation code
3. **Export utilities:** Save plots with consistent naming/format
4. **Theme support:** Easy switch between light/dark themes
5. **Accessibility:** Colorblind-friendly palettes by default

### Continuous Improvement
- Monthly review of new plotting patterns
- Update utilities as new notebooks added
- Collect student feedback on clarity
- Benchmark notebook load/run times

---

## Appendix A: Quick Reference

### Utility Functions Available

```python
# Style configuration
setup_plotting_style()

# Axes styling
style_axes(ax, title, xlabel, ylabel)

# Subplot creation
fig, axes = create_comparison_plot(n=2)  # Side-by-side
fig, axes = create_grid_plot(rows=2, cols=3)  # Grid

# Training visualization
plot_training_curves(history, metrics=['loss', 'accuracy'])

# Confusion matrix
plot_confusion_matrix_styled(cm, classes)
```

### GP-Specific Functions

```python
# Sample from GP
samples = sample_gp_prior(kernel, X, n_samples=5)

# Plot samples
plot_gp_samples(X, samples, ax)

# Plot posterior
plot_gp_posterior(X_train, y_train, X_test, mean, std)
```

---

## Appendix B: File Locations

### Notebooks Analyzed
```
Lecture 2 KNN/02-KNN.ipynb
Lecture 3 Linear Regression/03-Linear-Regression.ipynb
Lecture 4 Optimization/04-Optimization-GD.ipynb
Lecture 5 Probabilistic Classification/05-ProbClass.ipynb
Lecture 6 Evaluation Pitfalls and Data Visualization/06-ModelEval.ipynb
Lecture 7 Regularization and Generalization/07-Overfitting-Regularization.ipynb
Lecture 8 Modern Decision Trees/08-DecisionTrees-ModernView.ipynb
Lecture 9 Ensemble Methods/09-Ensemble-Methods.ipynb
Lecture 10 Kernel Methods/10-GP-Kernel-Methods.ipynb
Lecture 11 K-Means/11-KMeans.ipynb
Lecture 12 PCA/12-PCA.ipynb
Lecture 13 NN Architecture/13-FCNeuralNet.ipynb
Lecture 14 Transformers/14-Transformers.ipynb
Lecture 15 CNN/15-CNN.ipynb
Lecture 16 RNN and Recurrent Language Models/16_RNN.ipynb
Lecture 17 LLM Agents/17_llm_agents.ipynb
```

### Utilities to Create
```
ml_plotting_utils.py (root directory)
gp_plotting_utils.py (root directory)
```

---

## Summary

This comprehensive analysis identified significant opportunities to improve code quality across 17 educational machine learning notebooks. By creating shared plotting utilities and applying consistent presentation standards, we can reduce code duplication by ~30%, improve maintainability, and enhance the learning experience for students.

The phased implementation plan allows for incremental progress with early validation, minimizing risk while delivering measurable improvements.

**Total Estimated Time:** ~45 hours across 4 weeks
**Expected Completion:** End of Week 4
**Next Step:** Review this plan and approve Phase 1 to begin implementation
