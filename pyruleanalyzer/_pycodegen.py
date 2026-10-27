"""Building blocks of the native Python code generators.

Shared by ``NativePythonMixin`` (``_native_python.py``): the in-memory
``native_predict``, the standalone ``.py`` export and the per-tree rule
lookup all walk a tree rebuilt from its rules into nested ``if``/``else``
Python source, and differ only in what a leaf returns.
"""

from collections import defaultdict

from ._rules import _py_split_test


# Function to read a class label as the value the generated code returns
def label_value(label):
    """The label as an int when it reads as one, else the label itself.

    Args:
        label: A class label (``rule.class_``, ``default_class``), possibly
            prefixed with 'Class'.

    Returns:
        int | Any: ``label`` stripped of any 'Class' prefix and converted to
        int, or ``label`` unchanged when that conversion fails.
    """
    try:
        return int(str(label).replace('Class', '').strip())
    except Exception:
        return label


# Function to group rules by the tree they come from
def group_by_tree(rules, default_tid=None):
    """Group rules by tree identifier (the rule name's prefix before '_').

    Args:
        rules (List[Rule]): The rules to group, in order.
        default_tid (str, optional): Identifier of the rules whose name has
            no '_'. None groups each such rule under its own name.

    Returns:
        Dict[str, List[Rule]]: The rules of each tree, trees in order of
        first appearance.
    """
    tree_rules_map = defaultdict(list)
    for rule in rules:
        if '_' in rule.name:
            tid = rule.name.split('_')[0]
        else:
            tid = rule.name if default_tid is None else default_tid
        tree_rules_map[tid].append(rule)
    return tree_rules_map


# Function to replace the rule kept at each node with the value it returns
def set_leaf_values(tree, payload):
    """Store at each node (key 'v') the value its leaf returns.

    Args:
        tree (Dict[int, dict]): Nodes from `_tree_from_rules`; their 'rule'
            entry is removed.
        payload (Callable[[Rule | None], Any]): The value of a node from its
            rule (None for internal nodes and uncovered regions).

    Returns:
        Dict[int, dict]: `tree`, modified in place.
    """
    for node in tree.values():
        node['v'] = payload(node.pop('rule', None))
    return tree


# Function to generate the nested if/else walk of one tree
def tree_walk_source(tree, leaf, nan=True, node_id=0, indent=1):
    """Python source walking `tree` from `node_id` down to its leaves.

    Thresholds are written with ``repr(float(t))`` for precision, and the
    sample is indexed directly: a missing feature raises KeyError instead of
    being read as 0.

    Args:
        tree (Dict[int, dict]): Nodes with their leaf values ('v').
        leaf (Callable[[Any], str]): The expression a leaf returns, from its
            value.
        nan (bool): Send NaN where each split sends missing values. False for
            models that refuse NaN (GradientBoostingClassifier).
        node_id (int): The node to start from.
        indent (int): Indentation level of the generated lines.

    Returns:
        str: The source lines.
    """
    node = tree[node_id]
    tab = '    ' * indent
    if node['f'] == -2:
        return f"{tab}return {leaf(node['v'])}\n"
    missing = node.get('m') if nan else None
    code = f"{tab}if {_py_split_test(node['f'], node['t'], missing)}:\n"
    code += tree_walk_source(tree, leaf, nan, node['l'], indent + 1)
    code += f"{tab}else:\n"
    code += tree_walk_source(tree, leaf, nan, node['r'], indent + 1)
    return code


# Function to generate a one-tree function `name(sample)`
def function_source(name, tree, leaf, nan=True):
    """Source of ``def name(sample):`` walking `tree`.

    Args:
        name (str): The function name (a generated identifier, never a rule
            name or class label).
        tree (Dict[int, dict]): Nodes with their leaf values ('v').
        leaf (Callable[[Any], str]): The expression a leaf returns.
        nan (bool): See `tree_walk_source`.

    Returns:
        str: The function source.
    """
    return f"def {name}(sample):\n" + tree_walk_source(tree, leaf, nan)
