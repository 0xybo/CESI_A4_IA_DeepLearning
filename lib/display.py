from typing import Callable, Dict, Any
import ipywidgets as widgets
from IPython.display import display


def print_tabs(tabs: Dict[str, Callable[[], Any]]) -> None:
    widget = widgets.Tab()
    titles, children = [], []
    for name in tabs:
        titles.append(name)
        children.append(widgets.Output())
        with children[-1]:
            display(tabs[name]())

    widget.children, widget.titles = children, titles
    display(widget)
from typing import Callable, Dict, Any
import ipywidgets as widgets
from IPython.display import display


def print_tabs(tabs: Dict[str, Callable[[], Any]]) -> None:
    widget = widgets.Tab()
    titles, children = [], []
    for name in tabs:
        titles.append(name)
        children.append(widgets.Output())
        with children[-1]:
            display(tabs[name]())

    widget.children, widget.titles = children, titles
    display(widget)
