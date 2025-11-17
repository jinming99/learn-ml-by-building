"""
Notebook-friendly utilities for quick plotting and formatted output in tutorials.

Features
- setup_plotting_style: Apply consistent matplotlib/seaborn style across notebooks
- style_axes: Quick styling for axes (reduces fontsize/fontweight repetition)
- create_comparison_plot / create_grid_plot: Simplified subplot creation with auto-sizing
- quick_plot: Simple declarative plotting for common cases (line, bar, bar_grouped, heatmap, hist, scatter, fill, boxplot)
- show / show_metrics: Lightweight, formatted console output helpers
- log family: Styled logging for notebooks

These utilities are intentionally lightweight and avoid optional dependencies.
They complement specialized plotting functions in tutorials/utils_ev.py rather than replace them.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Prefer rich display in notebooks if available
try:  # pragma: no cover - optional dependency
    from IPython.display import display as _ipydisplay, HTML as _ipyHTML, update_display as _ipyUpdateDisplay  # type: ignore
except Exception:  # pragma: no cover
    _ipydisplay = None  # type: ignore
    _ipyHTML = None  # type: ignore
    _ipyUpdateDisplay = None  # type: ignore
import html as _html

__all__ = [
    "setup_plotting_style",
    "style_axes",
    "create_comparison_plot",
    "create_grid_plot",
    "quick_plot",
    "show",
    "show_metrics",
    "set_compact_notebook_style",
    "log",
    "log_info",
    "log_warning",
    "log_error",
    "log_success",
]

_PROGRESS_CREATED: Dict[str, bool] = {}

# =============================
# NEW: Plotting style helpers
# =============================

def setup_plotting_style(style='seaborn-v0_8-darkgrid', figsize=(10, 6),
                        fontsize=12, palette='husl'):
    """
    Apply consistent plotting style across all notebooks.

    This addresses the repetition of style configuration found in 15+ notebooks.
    Call once at the beginning of each notebook for consistency.

    Args:
        style: Matplotlib style to use
        figsize: Default figure size
        fontsize: Default font size
        palette: Seaborn color palette

    Example:
        from utils_nb import setup_plotting_style
        setup_plotting_style()
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = fontsize

    try:
        import seaborn as sns
        sns.set_palette(palette)
    except ImportError:
        pass  # Seaborn is optional


def style_axes(ax, title='', xlabel='', ylabel='',
               title_size=15, label_size=13, grid=True):
    """
    Apply consistent styling to axes - reduces repetitive fontsize/fontweight code.

    This addresses the 582 instances of repetitive inline font styling found across notebooks.
    Instead of:
        ax.set_title('Title', fontsize=15, fontweight='bold')
        ax.set_xlabel('X', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    Use:
        style_axes(ax, 'Title', 'X', 'Y')

    Args:
        ax: Matplotlib axes object
        title: Plot title (optional)
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
        title_size: Font size for title
        label_size: Font size for axis labels
        grid: Whether to show grid

    Example:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        style_axes(ax, 'Training Progress', 'Epoch', 'Loss')
    """
    if title:
        ax.set_title(title, fontsize=title_size, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size, fontweight='bold')
    if grid:
        ax.grid(True, alpha=0.3)


def create_comparison_plot(n_plots=2, figsize_per_plot=(8, 5),
                          layout='horizontal', **subplot_kw):
    """
    Create side-by-side or stacked comparison plots with automatic sizing.

    This addresses the 144 instances of repeated subplot creation patterns.

    Args:
        n_plots: Number of subplots
        figsize_per_plot: Size of each individual subplot (width, height)
        layout: 'horizontal' (side-by-side) or 'vertical' (stacked)
        **subplot_kw: Additional arguments passed to plt.subplots()

    Returns:
        fig, axes tuple (axes is array if n_plots > 1, single Axes otherwise)

    Examples:
        # Two side-by-side plots
        fig, (ax1, ax2) = create_comparison_plot(2)

        # Three stacked plots
        fig, axes = create_comparison_plot(3, layout='vertical')

        # Custom sizing per plot
        fig, axes = create_comparison_plot(3, figsize_per_plot=(6, 4))
    """
    if layout == 'horizontal':
        nrows, ncols = 1, n_plots
        figsize = (figsize_per_plot[0] * n_plots, figsize_per_plot[1])
    else:  # vertical
        nrows, ncols = n_plots, 1
        figsize = (figsize_per_plot[0], figsize_per_plot[1] * n_plots)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplot_kw)
    return fig, axes


def create_grid_plot(rows, cols, figsize_per_cell=(5, 4), **subplot_kw):
    """
    Create grid of subplots with automatic sizing and flattened axes array.

    Args:
        rows: Number of rows
        cols: Number of columns
        figsize_per_cell: Size of each cell (width, height)
        **subplot_kw: Additional arguments passed to plt.subplots()

    Returns:
        fig, flattened axes array (always 1D array for easy iteration)

    Examples:
        # 2x3 grid
        fig, axes = create_grid_plot(2, 3)
        for i, ax in enumerate(axes):
            ax.plot(data[i])
            style_axes(ax, f'Plot {i+1}')

        # With shared axes
        fig, axes = create_grid_plot(2, 2, sharex=True, sharey=True)
    """
    figsize = (figsize_per_cell[0] * cols, figsize_per_cell[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, **subplot_kw)

    # Always return flattened array for consistent iteration
    if rows * cols == 1:
        return fig, np.array([axes])
    else:
        return fig, np.array(axes).flatten()


# -----------------------------
# Printing helpers
# -----------------------------

def show(template: str, **kwargs) -> None:
    """
    Simplified formatted printing.

    Examples:
        show("header: Training Results")
        show("metric: Reward = {reward:.2f} ± {std:.2f}", reward=100.5, std=5.2)
        show("table", df=results_df)
        show("section: {title}", title="Robustness Analysis")
        show("list: Algorithms", items=['PPO', 'SAC', 'DQN'])
    """
    # Parse template type
    KNOWN_TYPES = {"header", "section", "metric", "table", "list", "progress", "result", "warning"}
    if ':' in template:
        msg_type, content = template.split(':', 1)
        msg_type = msg_type.strip().lower()
        content = content.strip()
    else:
        tentative = template.strip().lower()
        if tentative in KNOWN_TYPES:
            msg_type = tentative
            content = ""
        else:
            msg_type = 'text'
            content = template

    # Format based on type
    if msg_type == 'header':
        text = content.format(**kwargs) if kwargs else content
        if _ipydisplay is not None and _ipyHTML is not None:
            _ipydisplay(_ipyHTML(f"""
            <div style="margin:0;padding:6px 8px;border-radius:6px;line-height:1.1;
                        background:linear-gradient(180deg,#ffffff, #ffffff);border:1px solid #e5e7eb;display:inline-block;">
              <div style="font-weight:700;font-size:2.20rem;color:#111827;">
                {_html.escape(text)}
              </div>
            </div>
            """))
        else:
            print("\n" + "=" * 50)
            print(text)
            print("=" * 50)

    elif msg_type == 'section':
        text = content.format(**kwargs) if kwargs else content
        if _ipydisplay is not None and _ipyHTML is not None:
            _ipydisplay(_ipyHTML(f"""
            <div style="margin:0;padding:2px 6px;display:inline-block;border-left:4px solid #4F46E5;background:#F8FAFC;border-radius:6px;line-height:1.1;">
              <span style="font-weight:600;color:#1F2937;">{_html.escape(text)}</span>
            </div>
            """))
        else:
            print("\n" + "-" * 40)
            print(text)
            print("-" * 40)

    elif msg_type == 'metric':
        text = content.format(**kwargs)
        if _ipydisplay is not None and _ipyHTML is not None:
            _ipydisplay(_ipyHTML(f"""
            <div style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
                                color:#065F46;background:#ECFDF5;border:1px solid #A7F3D0;display:inline-block;
                                padding:4px 8px;border-radius:6px;margin:2px 0;">{_html.escape(text)}</div>
            """))
        else:
            print(f"  {text}")

    elif msg_type == 'table':
        df = kwargs.get('df')
        # Optional title/content line before the table
        if content:
            cap = content.format(**kwargs) if kwargs else content
            if _ipydisplay is not None and _ipyHTML is not None:
                _ipydisplay(_ipyHTML(f"""
                <div style=\"margin:6px 0 4px 0;color:#374151;font-weight:600;\">{_html.escape(cap)}</div>
                """))
            else:
                print("\n" + cap)
        if isinstance(df, pd.DataFrame):
            try:
                # Prefer rich HTML rendering in notebooks
                if _ipydisplay is not None:
                    _ipydisplay(df)
                else:
                    print("\n" + df.to_string())
            except Exception:
                print("\n" + df.to_string())
        else:
            # Fallback if no DataFrame provided
            print(content.format(**kwargs))

    elif msg_type == 'list':
        items = kwargs.get('items', [])
        if _ipydisplay is not None and _ipyHTML is not None:
            lis = ''.join(f"<li style='margin:2px 0;'>{_html.escape(str(it))}</li>" for it in items)
            title = _html.escape(content) if content else 'Items'
            _ipydisplay(_ipyHTML(f"""
            <div style="margin:8px 0;">
              <div style="font-weight:600;color:#111827;margin-bottom:4px;">{title}</div>
              <ul style="margin:0 0 0 20px;color:#374151;">{lis}</ul>
            </div>
            """))
        else:
            print(f"\n{content}:")
            for item in items:
                print(f"  - {item}")

    elif msg_type == 'progress':
        step = int(kwargs.get('step', 0))
        total = max(int(kwargs.get('total', 100)), 1)
        pct = int(100 * min(max(step / total, 0), 1))
        label = content.format(**kwargs)
        key = str(kwargs.get('key', label))
        html = f"""
        <div style="font-family:ui-sans-serif, system-ui; margin:6px 0;">
          <div style="color:#374151;font-size:0.9rem;margin-bottom:4px;">{_html.escape(label)} ({step}/{total})</div>
          <div style="background:#E5E7EB;border-radius:9999px;overflow:hidden;height:10px;">
            <div style="width:{pct}%;background:#4F46E5;height:10px;"></div>
          </div>
        </div>
        """
        if _ipydisplay is not None and _ipyHTML is not None and _ipyUpdateDisplay is not None:
            try:
                created = _PROGRESS_CREATED.get(key, False)
                if not created:
                    _ipydisplay(_ipyHTML(html), display_id=key)
                    _PROGRESS_CREATED[key] = True
                else:
                    _ipyUpdateDisplay(_ipyHTML(html), display_id=key)
            except Exception:
                _ipydisplay(_ipyHTML(html))  # fallback: append when update not available
        else:
            # Console fallback: overwrite line when possible
            try:
                import sys
                sys.stdout.write(f"\r[{step}/{total}] {label}")
                sys.stdout.flush()
                if step >= total:
                    sys.stdout.write("\n")
            except Exception:
                print(f"[{step}/{total}] {label}")

    elif msg_type == 'result':
        text = content.format(**kwargs)
        if _ipydisplay is not None and _ipyHTML is not None:
            _ipydisplay(_ipyHTML(f"""
            <div style="display:inline-flex;align-items:center;padding:4px 8px;border-radius:6px; background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0;">
              <span style="margin-right:6px;">✔</span><span>{_html.escape(text)}</span>
            </div>
            """))
        else:
            print(f"\u2713 {text}")

    elif msg_type == 'warning':
        text = content.format(**kwargs)
        if _ipydisplay is not None and _ipyHTML is not None:
            _ipydisplay(_ipyHTML(f"""
            <div style="display:inline-flex;align-items:center;padding:4px 8px;border-radius:6px;background:#FFFBEB;color:#92400E;border:1px solid #FDE68A;">
              <span style="margin-right:6px;">⚠️</span><span>{_html.escape(text)}</span>
            </div>
            """))
        else:
            print(f"\u26A0 {text}")

    else:  # Default text
        print(content.format(**kwargs) if kwargs else content)


def show_metrics(metrics: Dict[str, Any], title: str = "Metrics") -> None:
    """Show multiple metrics at once.

    If a value is a nested dict, prints nested keys indented.
    """
    # Rich HTML card in notebooks
    if _ipydisplay is not None and _ipyHTML is not None:
        # Build nested list HTML (compact spacing)
        def _fmt_val(v: Any) -> str:
            try:
                if isinstance(v, float):
                    return f"{v:.3f}"
                return _html.escape(str(v))
            except Exception:
                return _html.escape(str(v))

        def _dict_to_ul(d: Dict[str, Any]) -> str:
            items_html: List[str] = []
            for k, v in d.items():
                k_html = _html.escape(str(k))
                if isinstance(v, dict):
                    items_html.append(
                        f"<li style='margin:2px 0;'><b>{k_html}</b>: "
                        f"<ul style='margin:2px 0 0 14px;padding:0;'>{_dict_to_ul(v)}</ul></li>"
                    )
                else:
                    items_html.append(f"<li style='margin:2px 0;'><b>{k_html}</b>: {_fmt_val(v)}</li>")
            return "".join(items_html)

        content_html = _dict_to_ul(metrics)
        _ipydisplay(_ipyHTML(f"""
        <div style="margin:0;padding:6px 8px;border-radius:6px;border:1px solid #e5e7eb;background:#ffffff;display:block;line-height:1.15;">
          <div style="font-weight:700;color:#111827;margin-bottom:2px;">{_html.escape(title)}</div>
          <ul style="margin:0 0 0 10px;padding:0;color:#374151;list-style-position:inside;">{content_html}</ul>
        </div>
        """))
    else:
        # Console fallback
        show(f"section: {title}")
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k2, v2 in value.items():
                    print(f"    - {k2}: {v2}")
            elif isinstance(value, float):
                show(f"metric: {key} = {value:.3f}")
            else:
                show(f"metric: {key} = {value}")


# -----------------------------
# Logging helpers
# -----------------------------

def log(message: str, level: str = "info") -> None:
    """Lightweight logger that renders nicely in notebooks and falls back to console.

    Levels: info, warning, error, success, debug (debug renders like info).
    """
    level = str(level).lower()
    if _ipydisplay is not None and _ipyHTML is not None:
        color_map = {
            "info": "#1f2937",      # gray-800
            "warning": "#b45309",   # amber-600
            "error": "#b91c1c",     # red-700
            "success": "#065f46",   # emerald-800
            "debug": "#374151",     # gray-700
        }
        bg_map = {
            "info": "#f8fafc",
            "warning": "#fffbeb",
            "error": "#fef2f2",
            "success": "#ecfdf5",
            "debug": "#f3f4f6",
        }
        icon_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "⛔",
            "success": "🍺",
            "debug": "🐞",
        }
        color = color_map.get(level, color_map["info"])
        bg = bg_map.get(level, bg_map["info"])
        icon = icon_map.get(level, icon_map["info"])
        _ipydisplay(_ipyHTML(f"""
        <div style="font-family:ui-sans-serif, system-ui; margin:6px 0; padding:8px 10px; border-radius:6px; background:{bg}; color:{color}; border:1px solid rgba(0,0,0,0.05);">
          <span style="margin-right:6px;">{icon}</span>
          <span>{_html.escape(str(message))}</span>
        </div>
        """))
    else:
        prefixes = {
            "info": "[INFO]",
            "warning": "[WARN]",
            "error": "[ERROR]",
            "success": "[OK]",
            "debug": "[DEBUG]",
        }
        print(f"{prefixes.get(level, '[INFO]')} {message}")


def log_info(message: str) -> None:
    log(message, level="info")


def log_warning(message: str) -> None:
    log(message, level="warning")


def log_error(message: str) -> None:
    log(message, level="error")


def log_success(message: str) -> None:
    log(message, level="success")


# -----------------------------
# Notebook style helper
# -----------------------------

def set_compact_notebook_style(output_gap_px: int = 0, list_gap_px: int = 0) -> None:
    """No-op placeholder to avoid changing global notebook Markdown styling.

    This function previously injected global CSS. It is now intentionally
    empty so that only `show()` / `show_metrics()` inline styles control
    spacing. Safe to call; does nothing.
    """
    return


# -----------------------------
# Plotting helpers
# -----------------------------

def quick_plot(spec: Union[str, Dict[str, Any]], data: Optional[Union[Dict[str, Any], pd.DataFrame]] = None):
    """
    Create plots from simple specifications.

    Examples:
        # Simple line plot
        quick_plot("line: x=time, y=reward, title='Training Progress'",
                   {'time': t, 'reward': r})

        # Multi-panel plot (set 'return_obj': True to get the Figure back)
        quick_plot({
            'layout': '2x3',
            'plots': [
                {'type': 'line', 'x': 'time', 'y': 'demand', 'color': 'red'},
                {'type': 'bar', 'x': 'algo', 'y': 'score'},
                {'type': 'heatmap', 'data': 'actions', 'cmap': 'viridis'},
                {'type': 'hist', 'data': 'slack', 'bins': 50},
                {'type': 'scatter', 'x': 'carbon', 'y': 'reward'},
                {'type': 'boxplot', 'data': 'violations', 'by': 'algorithm'}
            ],
            'title': 'Comprehensive Analysis',
            'tight': True
        }, data)

        # Multi-series line plot from dict-of-series (legend auto)
        quick_plot({
            'type': 'line', 'x': 'timestep',
            'y': {'PPO': ppo_rewards, 'SAC': sac_rewards},
            'title': 'Algorithm Comparison', 'legend': True
        }, {'timestep': np.arange(len(ppo_rewards))})

        # Grouped line plot from DataFrame (one line per group value)
        # combined_df columns: ['timestep','lambda','variant']
        quick_plot("line: x=timestep, y=lambda, group=variant, legend=True", combined_df)
    """
    if data is None:
        data = {}

    # Parse string spec if provided
    if isinstance(spec, str):
        spec = _parse_plot_spec(spec)

    # Handle single vs multi-panel
    if 'layout' in spec:
        return _multi_panel_plot(spec, data)
    else:
        return _single_plot(spec, data)


def _coerce_value(val: str) -> Any:
    v = val.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    # tuple/list-like: (a,b) or [a,b]
    if (v.startswith('(') and v.endswith(')')) or (v.startswith('[') and v.endswith(']')):
        inner = v[1:-1]
        parts = [p.strip() for p in inner.split(',') if p.strip()]
        coerced = [_coerce_value(p) for p in parts]
        return tuple(coerced)
    # numeric
    try:
        if '.' in v:
            return float(v)
        return int(v)
    except Exception:
        pass
    # strip quotes if any
    return v.strip("\"'")


def _parse_plot_spec(spec_str: str) -> Dict[str, Any]:
    """Parse string like 'line: x=time, y=reward, title="Progress"' """
    parts = spec_str.split(':', 1)
    plot_type = parts[0].strip()

    params: Dict[str, Any] = {}
    if len(parts) > 1:
        # split on commas not within quotes (keep simple: assume commas in quoted strings are rare)
        for param in parts[1].split(','):
            if '=' in param:
                key, val = param.split('=', 1)
                key = key.strip()
                val = _coerce_value(val)
                params[key] = val
    params['type'] = plot_type
    return params


def _resolve(key_or_value: Any, dataset: Union[Dict[str, Any], pd.DataFrame]) -> Any:
    """Resolve a value: if it's a string key present in data, return data[key], else return the value itself."""
    if isinstance(key_or_value, str):
        # dict lookup
        if isinstance(dataset, dict) and key_or_value in dataset:
            return dataset[key_or_value]
        # DataFrame column lookup
        if isinstance(dataset, pd.DataFrame) and key_or_value in dataset.columns:
            return dataset[key_or_value]
    return key_or_value


def _single_plot(spec: Dict[str, Any], dataset: Union[Dict[str, Any], pd.DataFrame], ax: Optional[plt.Axes] = None):
    """Create a single plot based on spec."""
    # Track if this function created its own Figure (no Axes supplied)
    _created_figure = False
    if ax is None:
        # Figure size: prefer explicit 'figsize'; fall back to 'size' only if it's a 2-tuple/list
        _fs = spec.get('figsize', spec.get('size', (8, 5)))
        if not (isinstance(_fs, (list, tuple)) and len(_fs) == 2):
            _fs = (8, 5)
        fig, ax = plt.subplots(figsize=_fs)
        _created_figure = True

    plot_type = spec.get('type', 'line')

    # Convenience: allow passing a DataFrame directly via spec['data']
    df: Optional[pd.DataFrame] = None
    if 'data' in spec:
        dval = _resolve(spec['data'], dataset)
        if isinstance(dval, pd.DataFrame):
            df = dval
    # Fallback: use top-level dataset if it's a DataFrame
    if df is None and isinstance(dataset, pd.DataFrame):
        df = dataset

    # Extract data and render
    if plot_type == 'line':
        x = _resolve(spec.get('x'), dataset) if ('x' in spec) else None
        # allow alias: 'data' for y
        y_key = 'y' if ('y' in spec) else ('data' if ('data' in spec) else None)
        y = _resolve(spec.get(y_key), dataset) if y_key is not None else None
        color = spec.get('color', 'blue')
        lw = spec.get('lw', 2)
        label = spec.get('label')
        marker = spec.get('marker', None)
        markersize = spec.get('markersize', 8)
        linestyle = spec.get('linestyle', '-')
        alpha = spec.get('alpha', 1.0)

        # Grouped line plot from DataFrame: one line per group value
        group_col = spec.get('group') or spec.get('by')
        if df is not None and group_col and isinstance(spec.get('x'), str) and isinstance(y, str) \
           and spec.get('x') in df.columns and y in df.columns and group_col in df.columns:
            x_col = spec.get('x')
            for name, g in df.groupby(group_col):
                ax.plot(
                    g[x_col].to_numpy(), g[y].to_numpy(),
                    linewidth=lw, label=str(name),
                    marker=marker, markersize=markersize,
                    linestyle=linestyle, alpha=alpha
                )
            if spec.get('legend', True):
                ax.legend()

        # y can be array-like OR dict-of-series for multiple lines
        elif isinstance(y, dict):
            for lbl, series in y.items():
                ax.plot(
                    x if x is not None else range(len(series)), series,
                    linewidth=lw, label=str(lbl),
                    marker=marker, markersize=markersize,
                    linestyle=linestyle, alpha=alpha
                )
            if spec.get('legend', True):
                ax.legend()
        elif isinstance(y, (np.ndarray, list, pd.Series)):
            ax.plot(
                x if x is not None else range(len(y)), y,
                color=color, linewidth=lw, label=label,
                marker=marker, markersize=markersize,
                linestyle=linestyle, alpha=alpha
            )
            if label and spec.get('legend', False):
                ax.legend()
        else:
            # If df and columns given
            if df is not None and isinstance(y, str) and y in df.columns:
                ax.plot(
                    df[spec['x']] if x is not None else range(len(df)), df[y],
                    color=color, linewidth=lw, label=label,
                    marker=marker, markersize=markersize,
                    linestyle=linestyle, alpha=alpha
                )
                if label and spec.get('legend', False):
                    ax.legend()

    elif plot_type == 'line_errorbar':
        # Mean line with error bars (e.g., std or confidence intervals)
        x = _resolve(spec.get('x'), dataset)
        y = _resolve(spec.get('y'), dataset)
        yerr = _resolve(spec.get('yerr'), dataset)
        color = spec.get('color', 'blue')
        marker = spec.get('marker', 'o')
        linestyle = spec.get('linestyle', '-')
        lw = spec.get('lw', 2)
        capsize = spec.get('capsize', 5)
        label = spec.get('label')
        alpha = spec.get('alpha', 1.0)
        ax.errorbar(
            x, y, yerr=yerr,
            color=color, marker=marker, linestyle=linestyle,
            linewidth=lw, capsize=capsize, label=label, alpha=alpha
        )
        if label and spec.get('legend', False):
            ax.legend()

    elif plot_type == 'bar':
        if df is not None and ('x' in spec and 'y' in spec):
            ax.bar(df[spec['x']], df[spec['y']], color=spec.get('color', 'blue'), alpha=spec.get('alpha', 0.7))
        else:
            x = _resolve(spec.get('x'), dataset)
            y = _resolve(spec.get('y'), dataset)
            ax.bar(x, y, color=spec.get('color', 'blue'), alpha=spec.get('alpha', 0.7))

    elif plot_type == 'bar_grouped':
        # Flexible grouped bar rendering from a DataFrame or a pivot-like dict
        width = spec.get('width', 0.8)
        rot = spec.get('rot', 0)
        agg = spec.get('agg', 'mean')
        if df is not None:
            # If explicit pivot instructions
            idx = spec.get('index')
            cols = spec.get('columns')
            values = spec.get('values') or spec.get('y') or 'value'
            try:
                if idx and cols and values:
                    pvt = pd.pivot_table(df, index=idx, columns=cols, values=values, aggfunc=agg)
                elif {'scenario', 'algorithm', values}.issubset(set(df.columns)):
                    pvt = pd.pivot_table(df, index='scenario', columns='algorithm', values=values, aggfunc=agg)
                else:
                    # Try wide format: non-numeric columns as index
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    non_num = [c for c in df.columns if c not in num_cols]
                    pvt = df.set_index(non_num)[num_cols]
                pvt.plot(kind='bar', ax=ax, width=width, rot=rot)
            except Exception:
                df.plot(kind='bar', ax=ax, width=width, rot=rot)
        else:
            # Support dict: {'labels': [...], 'series': {name: values}}
            d = _resolve(spec.get('data'), dataset)
            if isinstance(d, dict) and 'labels' in d and 'series' in d:
                labels = list(d['labels'])
                series = dict(d['series'])
                x = np.arange(len(labels))
                n = len(series)
                bar_w = width / max(n, 1)
                for i, (name, vals) in enumerate(series.items()):
                    ax.bar(x + i * bar_w, vals, width=bar_w, label=str(name), alpha=0.8)
                ax.set_xticks(x + (n - 1) * bar_w / 2)
                ax.set_xticklabels(labels, rotation=rot)
                ax.legend()

    elif plot_type == 'bar_multi':
        """
        Multiple bar series with explicit positioning for side-by-side bars.
        Spec example:
            {'type': 'bar_multi', 'bars': [
                {'x': x1, 'y': y1, 'offset': -0.175, 'width': 0.35, 'label': 'A'},
                {'x': x2, 'y': y2, 'offset':  0.175, 'width': 0.35, 'label': 'B'}
            ]}
        """
        for bar_spec in spec.get('bars', []):
            x = np.asarray(_resolve(bar_spec.get('x'), dataset))
            y = np.asarray(_resolve(bar_spec.get('y'), dataset))
            offset = bar_spec.get('offset', 0)
            width = bar_spec.get('width', 0.35)
            ax.bar(
                x + offset, y,
                width=width,
                label=bar_spec.get('label'),
                alpha=bar_spec.get('alpha', 0.7),
                color=bar_spec.get('color', 'steelblue')
            )

    elif plot_type == 'bar_stacked':
        """
        Stacked bar chart.
        Spec example:
            {'type': 'bar_stacked', 'x': x_data, 'stacks': [
                {'y': data1, 'label': 'Clean', 'color': 'lightgreen', 'alpha': 0.7},
                {'y': data2, 'label': 'Dirty', 'color': 'lightcoral', 'alpha': 0.7}
            ]}
        """
        x = np.asarray(_resolve(spec.get('x'), dataset))
        bottom = np.zeros(len(x))
        width = spec.get('width', 0.8)
        for stack in spec.get('stacks', []):
            y = np.asarray(_resolve(stack.get('y'), dataset))
            ax.bar(
                x, y,
                width=width,
                bottom=bottom,
                label=stack.get('label'),
                color=stack.get('color', 'steelblue'),
                alpha=stack.get('alpha', 0.7)
            )
            bottom += y

    elif plot_type == 'heatmap':
        arr = _resolve(spec.get('data'), dataset)
        arr = np.asarray(arr)
        im = ax.imshow(
            arr,
            aspect='auto',
            cmap=spec.get('cmap', 'viridis'),
            vmin=spec.get('vmin', None),
            vmax=spec.get('vmax', None)
        )
        plt.colorbar(im, ax=ax, label=spec.get('clabel', ''))
        # Optional tick labels
        xtl = _resolve(spec.get('xticklabels'), dataset)
        ytl = _resolve(spec.get('yticklabels'), dataset)
        if xtl is not None:
            try:
                ax.set_xticks(np.arange(len(xtl)))
                ax.set_xticklabels(list(xtl), rotation=spec.get('xtickrot', 0))
            except Exception:
                pass
        if ytl is not None:
            try:
                ax.set_yticks(np.arange(len(ytl)))
                ax.set_yticklabels(list(ytl))
            except Exception:
                pass

    elif plot_type == 'hist':
        arr = _resolve(spec.get('data'), dataset)
        bins = spec.get('bins', 30)
        alpha = spec.get('alpha', 0.7)
        color = spec.get('color', 'blue')
        if isinstance(arr, dict):
            for lbl, vals in arr.items():
                ax.hist(vals, bins=bins, alpha=alpha, label=str(lbl))
            if spec.get('legend', True):
                ax.legend()
        else:
            ax.hist(arr, bins=bins, alpha=alpha, color=color)

    elif plot_type == 'hist_counts':
        # Draw histogram from precomputed counts and bins
        counts = _resolve(spec.get('counts'), dataset)
        bins = _resolve(spec.get('bins'), dataset)
        alpha = spec.get('alpha', 0.7)
        color = spec.get('color', 'blue')
        align = spec.get('align', 'edge')  # 'edge' or 'mid'
        if isinstance(counts, dict):
            for lbl, cts in counts.items():
                b = bins[lbl] if isinstance(bins, dict) and lbl in bins else bins
                b = np.asarray(b)
                cts = np.asarray(cts)
                widths = np.diff(b)
                left = b[:-1]
                if align == 'mid':
                    left = b[:-1] + widths / 2.0
                ax.bar(left, cts, width=(widths if align == 'edge' else widths*0.9), alpha=alpha, label=str(lbl), align=('edge' if align=='edge' else 'center'))
            if spec.get('legend', True):
                ax.legend()
        else:
            b = np.asarray(bins)
            cts = np.asarray(counts)
            widths = np.diff(b)
            left = b[:-1]
            if align == 'mid':
                left = b[:-1] + widths / 2.0
            ax.bar(left, cts, width=(widths if align == 'edge' else widths*0.9), alpha=alpha, color=color, align=('edge' if align=='edge' else 'center'))

    elif plot_type == 'scatter':
        # Prefer DataFrame columns if a panel-specific DataFrame is provided
        x_spec = spec.get('x')
        y_spec = spec.get('y')
        hue = spec.get('hue')
        if 'data' in spec:
            dval = _resolve(spec['data'], dataset)
            if isinstance(dval, pd.DataFrame):
                df = dval
        # Resolve x
        if isinstance(x_spec, str) and df is not None and x_spec in df.columns:
            x = df[x_spec].to_numpy()
        else:
            x = np.asarray(_resolve(x_spec, dataset))
        # Resolve y
        if isinstance(y_spec, str) and df is not None and y_spec in df.columns:
            y = df[y_spec].to_numpy()
        else:
            y = np.asarray(_resolve(y_spec, dataset))
        alpha = spec.get('alpha', 0.6)
        size = spec.get('size', 50)
        color = spec.get('color', 'blue')
        marker = spec.get('marker', 'o')
        if hue is not None:
            # Resolve hue from DataFrame if possible, otherwise from dataset dict
            if isinstance(hue, str) and df is not None and hue in df.columns:
                h = df[hue].to_numpy()
            else:
                h = _resolve(hue, dataset)
            try:
                import pandas as _pd
                h_series = _pd.Series(h)
                # categorical if small unique set
                uniq = list(h_series.dropna().unique())
                if len(uniq) <= spec.get('hue_max_categories', 20) and (h_series.dtype == 'object' or h_series.dtype.name == 'category' or len(uniq) < len(h_series)/2):
                    cmap_name = spec.get('palette', 'tab10')
                    cmap = plt.cm.get_cmap(cmap_name, len(uniq))
                    for i, cat in enumerate(uniq):
                        mask = (h_series == cat).to_numpy()
                        ax.scatter(x[mask], y[mask], alpha=alpha, s=size, color=cmap(i), label=str(cat), marker=marker)
                    if spec.get('legend', True):
                        ax.legend(title=str(hue))
                else:
                    cmap_name = spec.get('cmap', 'viridis')
                    sc = ax.scatter(x, y, c=h_series.to_numpy(), alpha=alpha, s=size, cmap=cmap_name, marker=marker)
                    if spec.get('colorbar', True):
                        plt.colorbar(sc, ax=ax, label=spec.get('clabel', str(hue)))
            except Exception:
                ax.scatter(x, y, alpha=alpha, s=size, c=color, marker=marker)
        else:
            ax.scatter(x, y, alpha=alpha, s=size, c=color, marker=marker)

    elif plot_type == 'line_band':
        # Mean line with variability band
        x = _resolve(spec.get('x'), dataset)
        y = _resolve(spec.get('y'), dataset)
        y_low = _resolve(spec.get('y_lower'), dataset)
        y_up = _resolve(spec.get('y_upper'), dataset)
        color = spec.get('color', 'steelblue')
        alpha = spec.get('alpha', 0.2)
        lw = spec.get('lw', 2)
        ax.plot(x, y, color=color, linewidth=lw, label=spec.get('label'))
        ax.fill_between(x, y_low, y_up, color=color, alpha=alpha)

    elif plot_type == 'fill':
        x = _resolve(spec.get('x'), dataset)
        y1 = _resolve(spec.get('y1'), dataset)
        if 'y2' in spec:
            y2_spec = spec.get('y2')
            if isinstance(y2_spec, str) and y2_spec.lower() == 'zeros':
                y2 = np.zeros_like(np.asarray(y1))
            else:
                y2 = _resolve(y2_spec, dataset)
        else:
            y2 = np.zeros_like(np.asarray(y1))
        color = spec.get('color', None)
        if color is not None:
            ax.fill_between(x, y1, y2, alpha=spec.get('alpha', 0.3), color=color)
        else:
            ax.fill_between(x, y1, y2, alpha=spec.get('alpha', 0.3))

    elif plot_type == 'boxplot':
        if df is not None:
            col = spec.get('data') if isinstance(spec.get('data'), str) else None
            by = spec.get('by')
            try:
                df.boxplot(column=col, by=by, ax=ax)
                # Remove pandas auto-suptitle
                if ax.figure is not None:
                    try:
                        ax.figure.suptitle("")
                    except Exception:
                        pass
            except Exception:
                # fallback to matplotlib if needed
                values = df[col].dropna().values if (col and col in df) else df.select_dtypes(include=[np.number]).values
                ax.boxplot(values)
        else:
            arr = _resolve(spec.get('data'), dataset)
            if isinstance(arr, dict):
                # dict of label -> array
                vals = list(arr.values())
                ax.boxplot(vals, labels=list(arr.keys()))
            else:
                ax.boxplot(arr)

    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    # --- Reference lines ---
    # Horizontal lines (single value, dict, or list of dict/values)
    if 'hline' in spec:
        hspec = spec['hline']
        hlist = hspec if isinstance(hspec, (list, tuple)) else [hspec]
        for h in hlist:
            if isinstance(h, dict):
                ax.axhline(
                    y=h.get('y', 0),
                    color=h.get('color', 'red'),
                    linestyle=h.get('linestyle', '--'),
                    linewidth=h.get('lw', h.get('linewidth', 1.5)),
                    alpha=h.get('alpha', 1.0),
                    label=h.get('label')
                )
            else:
                ax.axhline(y=h, color='red', linestyle='--')

    # Vertical lines (single value, dict, or list)
    if 'vline' in spec:
        vspec = spec['vline']
        vlist = vspec if isinstance(vspec, (list, tuple)) else [vspec]
        for v in vlist:
            if isinstance(v, dict):
                ax.axvline(
                    x=v.get('x', 0),
                    color=v.get('color', 'gray'),
                    linestyle=v.get('linestyle', ':'),
                    linewidth=v.get('lw', v.get('linewidth', 1.5)),
                    alpha=v.get('alpha', 1.0),
                    label=v.get('label')
                )
            else:
                ax.axvline(x=v, color='gray', linestyle=':')

    # --- Twin y-axis support ---
    if 'twin_y' in spec:
        twin_spec = spec['twin_y'] or {}
        ax2 = ax.twinx()
        t_color = twin_spec.get('color', 'blue')
        t_alpha = twin_spec.get('alpha', 0.5)
        t_label = twin_spec.get('label')
        t_marker = twin_spec.get('marker', None)
        t_linestyle = twin_spec.get('linestyle', '-')
        t_lw = twin_spec.get('lw', 2)
        t_type = twin_spec.get('type', 'line')

        if t_type == 'line':
            tx = _resolve(twin_spec.get('x'), dataset)
            ty = _resolve(twin_spec.get('y'), dataset)
            ax2.plot(tx, ty, color=t_color, alpha=t_alpha, label=t_label, marker=t_marker, linestyle=t_linestyle, linewidth=t_lw)
        elif t_type == 'scatter':
            tx = _resolve(twin_spec.get('x'), dataset)
            ty = _resolve(twin_spec.get('y'), dataset)
            ax2.scatter(tx, ty, color=t_color, alpha=t_alpha, label=t_label, s=twin_spec.get('size', 30), marker=twin_spec.get('marker', 'o'))
        elif t_type == 'bar':
            tx = _resolve(twin_spec.get('x'), dataset)
            ty = _resolve(twin_spec.get('y'), dataset)
            ax2.bar(tx, ty, color=t_color, alpha=t_alpha, label=t_label)

        if 'ylabel' in twin_spec:
            ax2.set_ylabel(twin_spec['ylabel'], color=t_color)
        if twin_spec.get('legend', False) and t_label:
            ax2.legend(loc=twin_spec.get('legend_loc', 'upper right'))

    # --- Annotations ---
    if 'annotations' in spec:
        anns = spec['annotations']
        if isinstance(anns, dict):
            anns = [anns]
        for ann in anns:
            if 'text' in ann:
                ax.annotate(
                    ann['text'],
                    xy=ann.get('xy', (0, 0)),
                    xytext=ann.get('xytext', ann.get('xy', (0, 0))),
                    arrowprops=ann.get('arrowprops'),
                    fontsize=ann.get('fontsize', 10),
                    color=ann.get('color')
                )
            if 'region' in ann:
                x0, x1 = ann['region'][:2]
                ymin = ann.get('ymin', 0)
                ymax = ann.get('ymax', 1)
                ax.axvspan(x0, x1, ymin=ymin, ymax=ymax, alpha=ann.get('alpha', 0.2), color=ann.get('color', 'yellow'), label=ann.get('label'))

    # --- Overlay helpers (markers or lines) ---
    # Accept both 'overlay_markers' (original) and 'overlay' (alias)
    if 'overlay_markers' in spec or 'overlay' in spec:
        overlays = spec.get('overlay_markers', spec.get('overlay'))
        overlays_list = overlays if isinstance(overlays, (list, tuple)) else [overlays]
        for overlay_spec in overlays_list:
            if overlay_spec is None:
                continue
            x_ov = _resolve(overlay_spec.get('x'), dataset)
            y_ov = _resolve(overlay_spec.get('y'), dataset)
            marker = overlay_spec.get('marker', 'x')
            linestyle = overlay_spec.get('linestyle')
            color = overlay_spec.get('color', 'red')
            lw = overlay_spec.get('linewidth', overlay_spec.get('lw', 3))
            alpha = overlay_spec.get('alpha', 1.0)
            label = overlay_spec.get('label')
            otype = overlay_spec.get('type')  # may be 'line' or 'scatter'

            # If user indicates a line, or uses a line-style in 'marker', draw a line
            line_markers = {'-', '--', '-.', ':'}
            treat_as_line = (otype == 'line') or (isinstance(marker, str) and marker in line_markers) or (linestyle is not None)

            if treat_as_line:
                # Determine linestyle: prefer explicit 'linestyle'; else use marker if it is a line style
                ls = linestyle if linestyle is not None else (marker if isinstance(marker, str) and marker in line_markers else '-')
                ax.plot(x_ov, y_ov, linestyle=ls, color=color, linewidth=lw, alpha=alpha, label=label, zorder=5)
            else:
                ax.scatter(
                    x_ov,
                    y_ov,
                    marker=marker,
                    s=overlay_spec.get('size', 100),
                    color=color,
                    linewidths=lw,
                    label=label,
                    alpha=alpha,
                    zorder=5
                )

    # Apply common settings
    if 'title' in spec:
        ax.set_title(spec['title'])
    if 'xlabel' in spec:
        ax.set_xlabel(spec['xlabel'])
    if 'ylabel' in spec:
        ax.set_ylabel(spec['ylabel'])
    if 'xlim' in spec:
        ax.set_xlim(spec['xlim'])
    if 'ylim' in spec:
        ax.set_ylim(spec['ylim'])
    if spec.get('grid', True):
        ax.grid(True, alpha=0.3)
    if spec.get('legend'):
        ax.legend()

    # Save if requested
    if 'save' in spec:
        try:
            ax.figure.savefig(spec['save'], dpi=spec.get('dpi', 150), bbox_inches='tight')
        except Exception:
            pass

    # Display policy: by default, show once and return nothing to avoid duplicate Out[ ] rendering.
    # Only show here if this function created the Figure (standalone plot). For multi-panel usage,
    # _multi_panel_plot() will handle the single show() call for the composed Figure.
    if spec.get('show', True) and _created_figure:
        try:
            plt.show()
        except Exception:
            pass

    return ax.figure if spec.get('return_obj', False) else None


def _multi_panel_plot(spec: Dict[str, Any], data: Union[Dict[str, Any], pd.DataFrame]):
    """Create multi-panel plot from spec."""
    layout = spec['layout']
    # Determine rows, cols, and figure size from layout
    if isinstance(layout, str):
        rows, cols = map(int, layout.lower().split('x'))
        # Prefer explicit 'figsize'; fall back to 'size' if it looks like a 2-seq; else default
        _fs = spec.get('figsize', spec.get('size'))
        if isinstance(_fs, (list, tuple)) and len(_fs) == 2:
            figsize = tuple(_fs)
        else:
            figsize = (5 * cols, 4 * rows)
    elif isinstance(layout, dict):
        rows = int(layout.get('rows', 1))
        cols = int(layout.get('cols', 1))
        # Prefer explicit 'figsize' (spec first, then layout), then 'size' if 2-seq
        if 'figsize' in spec:
            figsize = spec['figsize']
        elif 'figsize' in layout:
            figsize = layout['figsize']
        else:
            _fs = spec.get('size', None)
            if isinstance(_fs, (list, tuple)) and len(_fs) == 2:
                figsize = tuple(_fs)
            else:
                figsize = (5 * cols, 4 * rows)
    else:
        # tuple/list like (rows, cols)
        rows, cols = layout
        _fs = spec.get('figsize', spec.get('size'))
        if isinstance(_fs, (list, tuple)) and len(_fs) == 2:
            figsize = tuple(_fs)
        else:
            figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=figsize,
        sharex=spec.get('sharex', False),
        sharey=spec.get('sharey', False),
    )
    axes_list: List[plt.Axes]
    if isinstance(axes, np.ndarray):
        axes_list = list(axes.flatten())
    else:
        axes_list = [axes]

    # Support both keys: 'plots' (original) and 'panels' (alias used in tutorials)
    plot_list = spec.get('plots') if ('plots' in spec) else spec.get('panels', [])
    for i, plot_spec in enumerate(plot_list):
        if i < len(axes_list):
            _single_plot(plot_spec, data, axes_list[i])
    # Hide any unused axes to avoid empty panels
    for j in range(len(plot_list), len(axes_list)):
        try:
            axes_list[j].axis('off')
        except Exception:
            pass

    if 'title' in spec:
        fig.suptitle(spec['title'], fontsize=14, fontweight='bold')

    if spec.get('tight', True):
        try:
            fig.tight_layout()
        except Exception:
            pass

    # Save if requested (for full dashboard)
    if 'save' in spec:
        try:
            fig.savefig(spec['save'], dpi=spec.get('dpi', 150), bbox_inches='tight')
        except Exception:
            pass

    # Display policy: show once by default and avoid returning the figure so Jupyter doesn't render twice.
    if spec.get('show', True):
        try:
            plt.show()
        except Exception:
            pass

    return fig if spec.get('return_obj', False) else None
