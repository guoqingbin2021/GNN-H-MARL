# Namespace directory

The retained `onpolicy/__init__.py` imports `onpolicy.runner`. This file preserves that namespace directory in a Git checkout without bundling the upstream benchmark runners or changing the package initializer. UAMCO-DAG uses its own baseline adapters rather than those benchmark runners.
