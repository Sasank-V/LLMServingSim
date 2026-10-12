import json
from pathlib import Path

nb_path = Path('notebooks/v4_FairRoute_Full_Analysis.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 2 (imports) to import display from IPython.display
cell_2 = nb['cells'][2]
cell_2_code = "".join(cell_2['source'])

display_import = """try:
    from IPython.display import display, HTML, Markdown
except ImportError:
    def display(*args, **kwargs):
        for a in args:
            if hasattr(a, 'to_string'):
                print(a.to_string())
            else:
                print(a)
"""

if "from IPython.display import display" not in cell_2_code:
    # insert right after standard imports
    lines = cell_2_code.split("\n")
    # find where to insert (around line 20 after scipy import)
    insert_idx = 0
    for idx, line in enumerate(lines):
        if "from scipy" in line:
            insert_idx = idx + 1
            break
    if insert_idx == 0:
        insert_idx = len(lines)
    lines.insert(insert_idx, "\n" + display_import)
    cell_2['source'] = [l + "\n" for l in "\n".join(lines).split("\n")]
    if cell_2['source'] and cell_2['source'][-1] == "\n":
        cell_2['source'][-1] = ""

# 2. Update Cell 4 to ensure safe discovery and safe display(ct)
cell_4 = nb['cells'][4]
cell_4_code = "".join(cell_4['source'])

# Make ct display safe
safe_ct = """# Cross-tab
if len(run_index) > 0 and 'workload' in run_index.columns and 'rho' in run_index.columns:
    ct = run_index.groupby(['workload', 'rho']).policy.nunique().unstack(fill_value=0)
    display(ct)
else:
    print('No runs found or missing columns.')
"""

if "if len(run_index) > 0" not in cell_4_code and "ct = run_index.groupby" in cell_4_code:
    cell_4_code = cell_4_code.replace(
        "ct = run_index.groupby(['workload', 'rho']).policy.nunique().unstack(fill_value=0)\ndisplay(ct)",
        safe_ct
    )
    cell_4['source'] = [l + "\n" for l in cell_4_code.split("\n")]
    if cell_4['source'] and cell_4['source'][-1] == "\n":
        cell_4['source'][-1] = ""

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully patched display and safety checks in v4 notebook!')
