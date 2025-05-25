import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualisation.vis import find_representative_subset, reward_type, group_type, ind_type, TYPE_NOTION, type_columns
from scenario.create_fair_env import reward_indices


def add_underline(full_figure, obj):
    if obj == ["ARH", "SB"]:
        # Add an underline shape just below the title, above the radar plots
        full_figure.add_shape(
            type="rect",
            x0=-0.04, x1=0.62,  # Adjust width fraction here
            y0=1.088, y1=1.084,  # adjusted lower, thin line
            xref='paper', yref='paper',
            line=dict(color="black", width=0),
            fillcolor="black",
            layer="above"  # Make sure it stays above background
        )
    elif obj == ["ARH", "SB", "SBF"]:
        full_figure.add_shape(
            type="rect",
            x0=-0.04, x1=0.77,  # Adjust width fraction here
            y0=1.088, y1=1.084,  # adjusted lower, thin line
            xref='paper', yref='paper',
            line=dict(color="black", width=0),
            fillcolor="black",
            layer="above"  # Make sure it stays above background
        )
    elif obj == ["ARH", "ABFTU"]:
        full_figure.add_shape(
            type="rect",
            x0=-0.04, x1=0.75,  # Adjust width fraction here
            y0=1.088, y1=1.084,  # adjusted lower, thin line
            xref='paper', yref='paper',
            line=dict(color="black", width=0),
            fillcolor="black",
            layer="above"  # Make sure it stays above background
        )

    elif obj == ["ARH", "SBF"]:
        full_figure.add_shape(
            type="rect",
            x0=-0.04, x1=0.66,  # Adjust width fraction here
            y0=1.088, y1=1.084,  # adjusted lower, thin line
            xref='paper', yref='paper',
            line=dict(color="black", width=0),
            fillcolor="black",
            layer="above"  # Make sure it stays above background
        )
    else:
        full_figure.add_shape(
            type="rect",
            x0=-0.04, x1=0.89,  # Adjust width fraction here
            y0=1.088, y1=1.084,  # adjusted lower, thin line
            xref='paper', yref='paper',
            line=dict(color="black", width=0),
            fillcolor="black",
            layer="above"  # Make sure it stays above background
        )

def format_number(x):
    if round(x, 1) == 0:
        return round(x, 2)
    return round(x, 1)

def add_range_labels(full_figure, ranges):
    print(ranges)
    offset = 0.07
    small_offset = 0.03
    quart = 0.15
    middle = 0.5
    top = 1
    low = 0
    positions = {
        "ARI": [(middle + 0.04, top + small_offset), (middle, middle + offset - 0.01)],

        "ARH": [(quart + 0.04, top-quart+(small_offset*2)), (middle - offset, middle + offset)],

        "SBS": [(top-0.02, middle-0.04), (middle + offset - 0.01, middle)],

        "ABFTA": [(top-quart+0.02, top-quart+0.01), (middle + offset, middle + offset)],

        "SB_W": [(low+small_offset, middle+small_offset), (middle - offset + 0.01, middle)],

        "SB_S": [(quart-0.01, quart-0.02), (middle - offset, middle - offset)],

        "SB_L": [(middle, low - 0.04), (middle, middle - offset + 0.01)],

        "SB": [(top-quart-0.04, quart-0.06), (middle + offset, middle - offset)],
    }
    font = dict(
        size=7,
        color="black"
    )
    for o, r in ranges.items():
        #print(f"Range for {o} is {r}")
        hi = positions[o][0]
        lo = positions[o][1]
        low_text = format_number(r[0])
        high_text = format_number(r[1])
        full_figure.add_annotation(x=lo[0], y=lo[1], showarrow=False, text=low_text, font=font)
        full_figure.add_annotation(x=hi[0], y=hi[1], showarrow=False, text=high_text, font=font)


def get_radar_font():
    plotly_font_config = {
        "font": dict(
            family="serif",  # similar to matplotlib's 'serif'
            size=14,  # base size for labels and ticks
            color="black"
        ),
    }
    return plotly_font_config


def plot_radar(ranges, requested_objectives, all_objectives, sorted_objectives, iter_over, col_name, full_df, pcn_idx,
               get_representative_subset, polar_range, seeds, processes, chunk_size, save_dir, file_name,
               split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population,
               skip_subtitle, plot_all, plot_legend_as_subtitles, plot_single_objective,
               env_name, print_repr_policies_table=False, plot_policies_different_colours=False,
               plot_dashed_lines=False, image=True,
               use_uniform_spread_subset=False):
    # Establish the colour palette
    requested_objectives = [["SBF" if o == "SBS" else o for o in requested_objectives[0]]]
    requested_objectives = [["ABFTU" if o == "ABFTA" else o for o in requested_objectives[0]]]
    all_objectives = ["SBF" if o == "SBS" else o for o in all_objectives]
    all_objectives = ["ABFTU" if o == "ABFTA" else o for o in all_objectives]
    budget_symbol_map = {0: "circle", 2: "square", 3: "diamond",
                         4: "triangle-up", 5: "hexagon"}  # new
    budget_symbol_labels = {0: "◯", 2: "▢", 3: "⬦", 4: "△", 5: "▽"}  # new

    table_criteria = "objectives"
    extra_caption = f", when optimising for {'-'.join(requested_objectives[0])}"
    if split_per_bias:
        colour_palette = px.colors.qualitative.Plotly_r
        table_criteria = "reward biases"
    elif split_per_distance:
        colour_palette = px.colors.qualitative.Bold
        table_criteria = "individual fairness distance metrics"
    elif split_per_window:
        colour_palette = px.colors.qualitative.Set2
        table_criteria = "window sizes"
        if len(iter_over) == 4:
            colour_palette = [colour_palette[3]] + colour_palette[:3]
    elif split_per_population:
        colour_palette = px.colors.qualitative.Set1_r
        table_criteria = "populations"
    else:
        extra_caption = ""
    colour_palette = {
        2: ["dodgerblue", "steelblue", "cornflowerblue", "lightblue", "deepskyblue", "teal", "dodgerblue"],

        3: ["darkorange", "sandybrown", "peru", "gold", "chocolate", "goldenrod"],

        4: ["tomato", "darksalmon", "firebrick", "indianred", "lightcoral", "rosybrown"],

        5: ["mediumpurple", "blueviolet", "plum", "violet", "magenta", "slateblue"],

        0: ["forestgreen", "limegreen", "darkseagreen", "yellowgreen", "lime"]
    }
    cols_titles = None
    n_cols = len(iter_over)
    table_entries = []
    #table_columns = [col_name[0].upper() + col_name[1:], *all_objectives]
    table_columns = [col_name[0].upper() + col_name[1:], *all_objectives,
                     "Line Dash", "Color", "Symbol"]
    if plot_single_objective:
        n_cols = 3
        cols_titles = ["Performance", "Group Fairness", "Individual Fairness"]
    elif plot_legend_as_subtitles and not skip_subtitle:
        string_objectives = ["SBF" if o == "SBS" else o for o in requested_objectives]
        string_objectives = ["ABFTU" if o == "ABFTA" else o for o in string_objectives]
        cols_titles = ["+".join(objs) for objs in string_objectives] \
            if split_per_objective else list(iter_over.values())

    full_figure = make_subplots(rows=1, cols=n_cols, specs=[[{"type": "polar"} for _ in range(n_cols)]],
                                column_titles=cols_titles)
    font_size = 17
    font_size2 = font_size - 3
    #####
    if plot_single_objective:
        for col, obj_type in enumerate([reward_type, group_type, ind_type]):
            columns = [(f"<span style=\"color:{colour_palette[j]}\">{k}</span>"
                        if TYPE_NOTION[k] == obj_type else k) for j, k in enumerate(all_objectives)]
            full_figure.update_layout({f"polar{col + 1 if col > 0 else ''}": dict(
                radialaxis=dict(visible=True, angle=90, tickfont={"size": font_size2}, showticklabels=False), #, range=polar_range),
                angularaxis=dict(categoryarray=columns, rotation=90, tickfont={"size": font_size}))})

    for i, split in enumerate(iter_over):
        print("iterate over", iter_over)
        print("split", split)
        o_df = full_df[full_df[col_name] == split]
        print(f"{len(o_df)} non-dominated policies over {len(seeds)} seeds")

        if plot_single_objective:
            print("plot_single")
            columns = [(f"<span style=\"color:{colour_palette[j]}\">{k}</span>"
                        if TYPE_NOTION[k] == TYPE_NOTION[split] else k) for j, k in enumerate(all_objectives)]
        else:
            print("plot")
            #reqs = split.split(":") if split_per_objective else requested_objectives[0]
            reqs = split.split(":") if split_per_objective and isinstance(split, str) else requested_objectives[0]
            all_objs = [f"{o.split('_')[0]}<sub>{o.split('_')[1]}</sub>" if '_' in o else o for o in all_objectives]
            columns = [(f"<span style=\"color:{'black' if plot_policies_different_colours else colour_palette[i]}\">"
                        f"<b>{k}</b></span>" if k in reqs else k) for j, k in enumerate(all_objs)]

        if not plot_single_objective:
            full_figure.update_layout({f"polar{i + 1 if i > 0 else ''}": dict(
                radialaxis=dict(visible=True, angle=90, tickfont={"size": font_size2}, showticklabels=False), #range=polar_range),
                angularaxis=dict(categoryarray=columns, rotation=90, tickfont={"size": font_size}))
            })
        if get_representative_subset:
            objectives = ['ARI', 'ARH', 'SB', 'SBS', 'ABFTA']
            highlight_indices = find_representative_subset(
                o_df, requested_objectives, all_objectives, seeds,
                use_uniform_spread=use_uniform_spread_subset,
                processes=processes, chunk_size=chunk_size)
        else:
            highlight_indices = o_df.index
        # Ensure highlight_indices is an ordered list for consistent indexing
        highlight_indices = list(highlight_indices)


        draw_split = True
        dashes = ["dash", "dot", "dashdot", "longdash", "longdashdot", ]
        current_dash = -1
        current_repr = 0

        for l, (k, row) in enumerate(o_df.iterrows()):
            r = row.tolist()[:len(all_objectives)] + [row[0]]
            theta = columns + [columns[0]]
            # cycle through the list of colours for this budget
            marker_color = colour_palette[split][l % len(colour_palette[split])]
            # name = "+".join(split) if not (plot_single_objective or split_per_objective) else split
            name = split
            draw = True
            showlegend = draw_split and not plot_legend_as_subtitles
            if get_representative_subset:
                # Assign marker colors for highlighted policies in the order they appear
                if k in highlight_indices:
                    dash_idx = list(highlight_indices).index(k)
                    marker_color = colour_palette[split][dash_idx % len(colour_palette[split])]
                elif plot_policies_different_colours and not plot_single_objective:
                    marker_color = "gray"
                else:
                    marker_color = colour_palette[split][l % len(colour_palette[split])]

                # Representative policy styling
                if k in highlight_indices:
                    dash_idx = list(highlight_indices).index(k)
                    # Solid policies (best for each objective) full opacity, others slightly less
                    if dash_idx < len(requested_objectives[0]):
                        line_width = 2.5
                        opacity = 0.9
                    else:
                        line_width = 2
                        opacity = 0.6  # slightly reduced opacity for dashed ones
                    current_repr += 1
                    current_dash += 1
                elif plot_all:
                    line_width = 1.75
                    opacity = 0.65 * len(highlight_indices) / len(o_df)
                    showlegend = False
                else:
                    draw = False

            if draw:
                if plot_dashed_lines and (k in highlight_indices):
                    dash_idx = list(highlight_indices).index(k)
                    if dash_idx < len(requested_objectives[0]):
                        line_dash = "solid"
                    else:
                        line_dash = dashes[(dash_idx - len(requested_objectives[0])) % len(dashes)]
                    print(f"Policy index {k}: dash index = {dash_idx}, line_dash = {line_dash}")
                else:
                    line_dash = "solid"
                col = type_columns[TYPE_NOTION[split]] if plot_single_objective else i + 1
                # full_figure.add_trace(go.Scatterpolar(r=r, theta=theta, mode="markers+lines", marker_color=marker_color,
                #                                       opacity=opacity, name=name, showlegend=showlegend,
                #                                       line_width=line_width, marker_size=line_width * 2.5,
                #                                       line_dash=line_dash), row=1, col=col)
                font = get_radar_font()
                full_figure.add_trace(
                    go.Scatterpolar(
                        r=r,
                        theta=theta,
                        mode="markers+lines",
                        marker=dict(color=marker_color,
                                    symbol=budget_symbol_map.get(split, "circle"),
                                    size=line_width * 2.5),
                        opacity=opacity,
                        name=name,
                        showlegend=showlegend,
                        line_width=line_width,
                        line_dash=line_dash),
                    row=1, col=col)
                if showlegend:
                    draw_split = False

            if print_repr_policies_table and ((not get_representative_subset) or (k in highlight_indices)):
                # Remove trailing zeros after decimal points
                round_number = 5
                objs = [str(round(o, 1)) if o == int(o) else str(round(o, round_number)) for o in r[:-1]]
                obj_name = name.split("_")[1] if (split_per_window and "_" in str(name)) else name
                if split_per_population or split_per_bias:
                    obj_name = iter_over[split]
                #table_entries.append([obj_name, *objs])
                table_entries.append([obj_name, *objs, line_dash,
                                      marker_color,
                                      budget_symbol_map.get(split, "circle")])

    # Update figure
    size = 360
    subscript_names = any([('_' in o) for o in all_objectives])
    mm = 23 if subscript_names else 20   # 20
    mmm = mm * 1.5
    margins = dict(t=mm + 35, b=mm + 3, l=mm + 5, r=mm + 3)
    full_figure.update_layout({"margin": margins, "legend_font_size": font_size2})
    full_figure.update_annotations({"font_size": font_size})  # Subplot titles

    objectives_label = [", ".join(objs) for objs in requested_objectives][0]
    # Set the title as normal
    if split == 0:
        split_label = "∞"
    else:
        split_label = split
    full_figure.update_layout(
        title_text=f"Budget {split_label} - {objectives_label} - {budget_symbol_labels.get(split, '')}",
        title_font_size=20,
        font=dict(
            family="serif",
            size=14,
            color="black"
        ),
        title_pad=dict(b=30)  # Slightly more bottom padding to make space for underline
    )
    add_range_labels(full_figure, ranges)
    add_underline(full_figure, requested_objectives[0])

    if image:
        full_figure.write_image(f"{save_dir}/{file_name}{'_' + str(pcn_idx) if pcn_idx not in [-1, None] else ''}.jpg",
                                width=size * 0.75 * n_cols + margins["l"] * 4,
                                height=size,
                                scale=6)  # 4
    # time.sleep(1)
    # full_figure.write_image(f"{save_dir}/{file_name}{'_' + str(pcn_idx) if pcn_idx not in [-1, None] else ''}.pdf",
    #                         width=size * 0.75 * n_cols + margins["l"] * 4,
    #                         height=size,
    #                         scale=2)  # 4
    if print_repr_policies_table:
        latex_df = pd.DataFrame(table_entries, columns=table_columns)
        latex = latex_df.to_latex(index=False, label=f"table:{env_name.split('_')[0]}{file_name}",
                                  longtable=plot_single_objective,
                                  caption=f"The representative subset of {' '.join(env_name.split('_'))} policies, for "
                                          f"different {table_criteria}{extra_caption}. "
                                          f"Results rounded to {round_number} decimals.")
        lines = latex.splitlines()
        new_lines = []
        prev_str = None
        for line in lines:
            if "\\\\" in line and "&" in line:
                if prev_str is None:
                    prev_str = line
                elif line.split(" & ")[0] != prev_str.split(" & ")[0]:
                    new_lines.append("\\midrule")
                    prev_str = line
            new_lines.append(line)
        latex = "\n\\begin{landscape}\n" + "\n".join(new_lines) + "\n\\end{landscape}"
        print(latex)

    return table_entries
