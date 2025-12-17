import matplotlib.pyplot as plt
import numpy as np
import pypsa


def prepare_optimized_capacities_for_plotting(n: pypsa.Network, components: list[str]) -> dict:
    """Prepare optimized capacities of technologies for plotting.

    Creates a nested dictionary with buses as the top-level keys, then pypsa
    components as sub-keys (e.g., 'Generator', 'StorageUnit', 'Line'), then the
    carrier of each component  as sub-sub-keys (e.g., 'solar', for 'Generator',
    'battery' for 'StorageUnit'). The value under each sub-sub-key is sum of the
    optimized capacities of all components of that specific carrier at that bus.

    The nested dictionary looks as follows:
    ----------------------------------------------------------------------------
    {
        'bus1': {
            'Generator': {
                'gas': 10.0,
                'solar': 5.0,
                'wind': 15.0,
            },
            'StorageUnit': {
                'battery': 8.0,
                'pumped-hydro': 12.0
            },
            'Line': {
                'line1': 30.0
            }
        },
        'bus2': {
            'Generator': {
                'gas': 12.0,
                'solar': 8.0,
                'wind': 10.0,
            },
            'StorageUnit': {
                'battery': 10.0,
                'pumped-hydro': 15.0
            },
            'Line': {
                'line1': 30.0
            }
        },
        # ... more buses
    }
    ----------------------------------------------------------------------------
    """
    results = {bus: {component: {} for component in components} for bus in n.buses.index}

    for bus in n.buses.index:
        for component in components:
            if component == "Generator":
                # Iterate through each carrier for generators (e.g., 'gas', 'solar', 'wind')
                for carrier in n.generators.carrier.unique():
                    # Get the generators of the current bus and technology category (sum all units of this type)
                    gen_indices = n.generators[(n.generators.bus == bus) & (n.generators.carrier == carrier)].index
                    if not gen_indices.empty:
                        results[bus][component][carrier] = n.generators.loc[gen_indices, "p_nom_opt"].sum()
                    else:
                        results[bus][component][carrier] = 0  # No generators of this type at this bus
            elif component == "StorageUnit":
                for carrier in n.storage_units.carrier.unique():
                    # Get the storage units of the current bus and technology category
                    storage_indices = n.storage_units[
                        (n.storage_units.bus == bus) & (n.storage_units.carrier == carrier)
                    ].index
                    if not storage_indices.empty:
                        # Sum the p_nom_opt values for the storage units of this bus and technology category
                        results[bus][component][carrier] = n.storage_units.loc[storage_indices, "p_nom_opt"].sum()
                    else:
                        results[bus][component][carrier] = 0  # No storage units of this type at this bus
            elif component == "Line":
                for carrier in n.lines.carrier.unique():
                    # Get the lines of the current bus and technology category
                    line_indices = n.lines[
                        (n.lines.bus0 == bus) | (n.lines.bus1 == bus) & (n.lines.carrier == carrier)
                    ].index
                    if not line_indices.empty:
                        # Sum the s_nom_opt values for the lines of this bus and technology category
                        results[bus][component][carrier] = n.lines.loc[line_indices, "s_nom_opt"].sum()
                    else:
                        results[bus][component][carrier] = 0  # No lines of this type at this bus

    return results


def plot_optimized_capacities(
    n: pypsa.Network,
    components=["Generator", "StorageUnit", "Line"],
    carrier_color_mapping={
        "gas": "#E12346",
        "wind": "#3366CC",
        "solar": "#FFCC66",
        "battery": "#28D6D3",
        "transmission": "#D961C5",
    },
) -> None:
    """Plot the optimized capacities of technologies."""
    results = prepare_optimized_capacities_for_plotting(n, components)

    buses = list(results.keys())
    unique_components = set()
    unique_carriers = set()
    component_tech_carrier_mapping = {}

    # Build component_tech_carrier_mapping with unique components and tech_carriers
    # E.g., {'Generator': ['gas', 'solar', 'wind'], 'StorageUnit': ['battery'], 'Line': ['transmission']}
    for bus, components in results.items():
        unique_components.update(components.keys())
        for component, component_carriers in components.items():
            if component not in component_tech_carrier_mapping:
                component_tech_carrier_mapping[component] = []
            for carrier in component_carriers.keys():
                if carrier not in component_tech_carrier_mapping[component]:
                    component_tech_carrier_mapping[component].append(carrier)
            unique_carriers.update(component_carriers.keys())

    unique_components = sorted(list(unique_components))  # Sort components for consistent plotting order

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_width = 0.35
    ax.set_ylim(0, 1000)
    index = np.arange(len(unique_components))  # Used to calculate x-positions for bars
    legend_handles = {}

    # Iterate through each bus to plot its components
    for i, bus_name in enumerate(buses):
        # Calculate x-position for this bus's bars (left or right of the center tick)
        x_positions = index + (i - (len(buses) - 1) / 2) * bar_width

        # Iterate through each component (e.g., 'Generator', 'StorageUnit', 'Line')
        for j, component in enumerate(unique_components):
            current_x_pos = x_positions[j]
            bottom_val = 0

            # Check if the component exists on this bus and plot each tech's capacity as a segment in the stacked bar
            if component in results[bus_name]:
                for tech_carrier in component_tech_carrier_mapping[component]:
                    # Get value from the nested dict. Use .get() to avoid errors if a sub-cat is missing.
                    tech_carrier_cap = results[bus_name][component].get(tech_carrier, 0)

                    if tech_carrier_cap > 0:
                        bar_segment = ax.bar(
                            current_x_pos,
                            tech_carrier_cap,
                            bar_width,
                            bottom=bottom_val,
                            color=carrier_color_mapping.get(tech_carrier, "gray"),  # Use gray for undefined colors
                            edgecolor="white",
                            linewidth=0.5,
                            label=tech_carrier,
                        )
                        bottom_val += tech_carrier_cap

                        if tech_carrier not in legend_handles:
                            legend_handles[tech_carrier] = bar_segment
                    elif tech_carrier_cap == 0:
                        # If the value is zero, we still want to plot a bar segment for consistency
                        bar_segment = ax.bar(
                            current_x_pos,
                            0,
                            bar_width,
                            bottom=bottom_val,
                            color=carrier_color_mapping.get(tech_carrier, "gray"),
                            edgecolor="white",
                            linewidth=0.5,
                        )
                        bottom_val += 0
                        if tech_carrier not in legend_handles:
                            legend_handles[tech_carrier] = bar_segment

    # Add text labels for 'Bus1' and 'Bus2' below each group of bars to clearly indicate grouping
    for j, main_cat in enumerate(unique_components):
        bus1_x_center = index[j] - bar_width / 2
        bus2_x_center = index[j] + bar_width / 2
        # Use ax.get_ylim()[1] to dynamically place labels below the lowest point of the plot
        ax.text(bus1_x_center, -0.05 * ax.get_ylim()[1], "Bus1", ha="center", va="top", fontsize=9, color="dimgray")
        ax.text(bus2_x_center, -0.05 * ax.get_ylim()[1], "Bus2", ha="center", va="top", fontsize=9, color="dimgray")

        # Add a subtle line to visually group Bus1 and Bus2 labels under each component
        ax.plot(
            [bus1_x_center - bar_width / 2, bus2_x_center + bar_width / 2],
            [-0.03 * ax.get_ylim()[1], -0.03 * ax.get_ylim()[1]],
            color="gray",
            linestyle="-",
            linewidth=0.8,
            clip_on=False,
        )

    # Create a sorted legend
    ordered_legend_labels = sorted(legend_handles.keys())
    ordered_legend_handles = [legend_handles[label] for label in ordered_legend_labels]
    ax.legend(
        ordered_legend_handles, ordered_legend_labels, title="Technology", bbox_to_anchor=(1, 1.01), loc="upper left"
    )

    # Customizing the Plot
    ax.set_ylabel("Installed capacity [MW]")
    ax.set_xticks(index)
    ax.set_xticklabels(unique_components)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.xaxis.grid(False)

    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
