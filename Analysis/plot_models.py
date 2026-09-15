# -*- coding: utf-8 -*-
"""Which models the figures contain, and how they are labelled — one config file for every script.

The list lives in `model_plot_config.json`, next to a results root (currently
`results/四次双模式/model_plot_config.json`), as

    {
      "models": [
        { "key": "egcn", "name": "EvolveGCN-H" },
        // { "key": "tna", "name": "BiTNA" },        <- 注释掉这一行就不画它
      ]
    }

so that dropping a model from every figure is one commented line instead of an edit in each of
`visualize_results.py`, `plot_train_4metrics.py`, `seed_ci_figures.py`, `seed_ci_summary.py` and
`PrROC_all_models.py`. Strict JSON has no comments, so the reader strips `//` line comments and
`/* */` blocks first (and tolerates the trailing comma that commenting out the last entry leaves
behind); `"include": false` on an entry does the same thing without touching the punctuation.

Lookup order for the file (first hit wins):
    1. an explicit `--model-config PATH`
    2. $IMPUT_MODEL_CONFIG
    3. <results-dir>/model_plot_config.json          (pointed straight at the dataset folder)
    4. <results-dir>/../model_plot_config.json       (pointed at .../ensemble or .../confidence)
When none exists the scripts behave exactly as before: every model found on disk is drawn and the
display names come from the built-in table.

Usage
-----
    from plot_models import load_plot_models          # no heavy imports, safe for any script

    cfg = load_plot_models(results_dir)               # None when there is no config file
    if cfg is not None:
        names = cfg.mapping(available=set(discovered), base_names=BUILTIN)
"""
import json
import os
from pathlib import Path

CONFIG_NAME = 'model_plot_config.json'
ENV_VAR = 'IMPUT_MODEL_CONFIG'


def _strip_comments(text):
    """Remove // and /* */ comments. String literals are respected, so a `//` inside a name
    is not mistaken for a comment."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == '\\' and i + 1 < n:
                out.append(ch)
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == '/' and i + 1 < n and text[i + 1] == '/':
            nl = text.find('\n', i)
            if nl < 0:
                break          # comment runs to the end of the file
            i = nl             # keep the newline (the '/' itself is dropped)
            continue
        elif ch == '/' and i + 1 < n and text[i + 1] == '*':
            end = text.find('*/', i + 2)
            i = n if end < 0 else end + 2
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _drop_dangling_commas(text):
    """Remove a comma that is followed only by whitespace before `}` or `]`.

    Commenting out the last entry of a list leaves exactly such a comma behind, and it has to be
    removed *after* the comments are gone: in the original text a comment usually sits between
    the comma and the closing bracket.
    """
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == '\\' and i + 1 < n:
                out.append(ch)
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == ',':
            j = i + 1
            while j < n and text[j] in ' \t\r\n':
                j += 1
            if j < n and text[j] in '}]':
                i += 1
                continue
        out.append(ch)
        i += 1
    return ''.join(out)


def strip_comments(text):
    """Comments out, then the dangling commas they leave behind -> parseable JSON."""
    return _drop_dangling_commas(_strip_comments(text))


def resolve_config_path(results_dir=None, config_path=None):
    """Path of the plot config to use, or None when the scripts should not filter at all."""
    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--model-config not found: {path}")
        return path
    env = os.environ.get(ENV_VAR)
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"${ENV_VAR} points to a missing file: {path}")
        return path
    if results_dir:
        root = Path(results_dir).expanduser()
        for candidate in (root / CONFIG_NAME, root.parent / CONFIG_NAME):
            if candidate.is_file():
                return candidate
    return None


class PlotModels:
    """Config-ordered model entries: list order = drawing order = colour order."""

    def __init__(self, path, entries, skipped=()):
        self.path = Path(path)
        self.entries = entries            # [(key, name), ...]
        self.skipped = list(skipped)      # entries dropped by "include": false

    def __bool__(self):
        return bool(self.entries)

    @property
    def keys(self):
        return [key for key, _ in self.entries]

    def mapping(self, available=None, base_names=None, verbose=True):
        """Ordered {model: display name} for the models this config selects.

        `available` is the set of model directories found on disk: entries that are not there are
        reported and dropped (a config is allowed to describe more than one results root), and
        directories that are not in the config are reported as skipped.
        """
        base_names = base_names or {}
        avail = None if available is None else set(available)
        out = {}
        for key, name in self.entries:
            if avail is not None and key not in avail:
                if verbose:
                    print(f"  [WARN] {self.path.name} lists '{key}' but no such directory under "
                          f"the results root — skipped")
                continue
            out[key] = name or base_names.get(key, key)
        if avail is not None and verbose:
            unknown = sorted(avail - set(out))
            if unknown:
                print(f"  [SKIP] on disk but not in {self.path.name}: {', '.join(unknown)}")
        return out


def load_plot_models(results_dir=None, config_path=None):
    """Read the plot config; None when there is none (meaning: no filtering)."""
    path = resolve_config_path(results_dir, config_path)
    if path is None:
        return None
    payload = json.loads(strip_comments(path.read_text(encoding='utf-8')))
    entries, skipped = [], []
    for raw in payload.get('models', []):
        key = (raw.get('key') or '').strip()
        name = (raw.get('name') or '').strip()
        if not key:
            continue
        if raw.get('include', True) is False:
            skipped.append(key)
            continue
        if key in [k for k, _ in entries]:
            print(f"  [WARN] {path.name} lists '{key}' more than once — first entry wins")
            continue
        entries.append((key, name))
    return PlotModels(path, entries, skipped)


def active_models(base_names, available=None, results_dir=None, config_path=None, verbose=True):
    """The mapping the plotting scripts should use.

    No config file -> `base_names` unchanged (same behaviour as before this module existed).
    """
    cfg = load_plot_models(results_dir, config_path)
    if cfg is None:
        return dict(base_names)
    mapping = cfg.mapping(available=available, base_names=base_names, verbose=verbose)
    if verbose:
        print(f"  Plot config   : {cfg.path}")
        print(f"  Models drawn  : {len(mapping)} ({', '.join(mapping)})")
        if cfg.skipped:
            print(f"  Models off    : {', '.join(cfg.skipped)} (\"include\": false)")
    return mapping
