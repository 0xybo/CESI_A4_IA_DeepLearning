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


# def print_tabs(tabs: Dict[str, Callable[[], Any]]) -> None:
#     widget = widgets.Tab()
#     titles, children = [], []
#     for name in tabs:
#         titles.append(name)
#         children.append(widgets.Output())
#         with children[-1]:
#             display(tabs[name]())

#     widget.children, widget.titles = children, titles
#     display(widget)

def print_tabs(tabs: Dict[str, Callable[[], Any]]) -> None:
    """
    Print the content of the tabs as banners. Each tab will be printed as a banner with the name of the tab as the title and the content of the tab as the body.

    Args:
        tabs (Dict[str, Callable[[], Any]]): A dictionary where the keys are the names of the tabs and the values are callables that return the content of the tabs.
    Returns:
        None
    """
    for name in tabs:
        print(f"╭{'─' * 56}╮")
        print(f"│{name.center(56)}│")
        print(f"╰{'─' * 56}╯")
        tabs[name]()

