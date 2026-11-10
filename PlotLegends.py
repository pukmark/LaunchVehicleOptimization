"""Consistent vehicle-level keys for trajectory plots."""
from matplotlib.lines import Line2D


def set_vehicle_legend(container, phases, **options):
    """Show only Main vehicle / Booster, and omit single-phase legends."""
    if hasattr(container, 'get_legend'):
        legend = container.get_legend()
        if legend is not None:
            legend.remove()
    else:
        for legend in list(container.legends):
            legend.remove()
    if not {'Main vehicle', 'Booster'} <= set(phases):
        return
    handles = [Line2D([], [], color='.25', linestyle=style, linewidth=1.5, label=label)
               for label, style in (('Main vehicle', '-'), ('Booster', '--'))]
    container.legend(handles=handles, **options)
