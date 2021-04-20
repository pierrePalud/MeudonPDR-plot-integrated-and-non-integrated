import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import re
import os 
from tqdm.auto import tqdm


def read_extraction_result_dat(filename):
    """extract integrated data. The file to be read must be the result of 
    an extraction of a .stat file.

    Parameters
    ----------
    filename : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # read header
    with open(filename) as f:
        list_cols = []
        line_content = f.readline()
        while line_content[0] == "#":
            list_cols.append(line_content)
            line_content = f.readline()
        
    nrows_header = len(list_cols)
    cols_names = ''.join(list_cols).replace("#", "").replace("\n", "").split("  ")
    cols_names = [col for col in cols_names if len(col) > 6]
    cols_names = [col[1:] if col[0] == " " else col for col in cols_names]

    # read rest of file
    df = pd.read_csv(
        filename, header=0, names=cols_names, 
        engine="python", skiprows=nrows_header, sep="  "
    )

    return df




def read_dat_file(filename):
    first_lines = []
    num_lines_to_skip = 3
    with open(f"./data/extraction_result/{filename}") as f:
        for _ in range(num_lines_to_skip + 1):
            line = f.readline()
        while line[0] == "#":
            first_lines.append(line)
            line = f.readline()
            num_lines_to_skip += 1

    # withdraw two last
    first_lines = first_lines[:-2]
    # withdraw # (first) and \n (last)
    first_lines = [line[1:-2] for line in first_lines]

    cols = re.compile(r"\s+\[\d+\]").split(' '.join(first_lines))
    cols[0] = cols[0][3:]

    # if rwo lines or more have the same name
    a,b = np.unique(cols, return_counts=True)
    if a[b >= 2].size > 0:
        for elt in a[b >= 2]:
            cols = [f"{col}_{i}" if col == elt else col for i,col in enumerate(cols)] 
    
    grid = pd.read_csv(
        f"./data/extraction_result/{filename}", 
        sep="   ", skiprows=num_lines_to_skip, engine='python', header=0, names=cols)
    return grid


def extract_simulation_parameters(filename):
    p = re.compile("[nrA]")

    params = filename.split('_')[1]
    params_str = p.split(params)[1:]
    params_float = [float(param.replace('p', '.')) for param in params_str] 
    return params_float # return [nH, G0, Avmax]

def is_increasing(df_inter, idx_col):
    diffs = np.diff(df_inter.iloc[:,idx_col], 1)
    return np.all(diffs > -1e-8)

def attains_lim(df_inter, idx_col, lim=1):
    return np.max(df_inter.iloc[:,idx_col]) >= lim

def where_attains_lim(df_inter, idx_col, lim=1):
    idx_attains = np.argmin(np.abs(df_inter.iloc[:, idx_col] - lim))
    return df_inter.index[idx_attains]


def scrape_all_files(Avmax="1e1"):
    list_files = os.listdir("./data/extraction_result")
    list_opticaldepth_files = [filename for filename in list_files if f"A{Avmax}_a_20.pop" in filename]
    
    list_results = []
    for filename_pop in tqdm(list_opticaldepth_files):
        simulation_params = extract_simulation_parameters(filename_pop)

        df_simulation_pop = read_dat_file(filename_pop)

        filename_optdepth = filename_pop.replace(".pop", ".OptDepth")
        df_simulation_optdepth = read_dat_file(filename_optdepth)

        df_simulation = pd.merge(df_simulation_pop, df_simulation_optdepth, on='AV')

        df_simulation['nH'] = simulation_params[0]
        df_simulation['radm'] = simulation_params[1]

        list_results.append(df_simulation)

    df_final = pd.concat(list_results, axis=0)
#    df_final = df_final.set_index(['nH', 'radm', 'AVmax', "line"])
    return df_final



##* Extract and compute

def extract_all_transition_data(df, transition, species, upper_state):
    cols_to_keep = ["nH", 'radm', 'AV', "Temperature"]

    # add density of upper state
    list_density_col = [col for col in list(df.columns) if (f"n({species} {upper_state})" in col)]
    assert len(list_density_col) == 1
    density_col = list_density_col[0]
    cols_to_keep += [density_col]

    # add line optical depth
    list_line_optdepth_col = [
        col for col in list(df.columns) 
        if (f"Line optical depth observer side({transition})" in col) and (transition in col)
    ]
    assert len(list_line_optdepth_col) == 1
    line_optdepth_col = list_line_optdepth_col[0]
    cols_to_keep += [line_optdepth_col]

    df_species = df[cols_to_keep]
    return df_species, density_col, line_optdepth_col


def get_density_peak_data(df_species, density_col):
    idx_density_peak = (
        df_species.groupby(['nH', 'radm'])[density_col].transform(max) == df_species[density_col]
    )

    df_species_inter = df_species[idx_density_peak]

    df_species_inter = df_species_inter.set_index(['nH', 'radm'])
    df_species_inter.columns = [f"{col}_density_peak" for col in df_species_inter.columns]
    df_species_inter = df_species_inter.reset_index()
    return df_species_inter


def get_optically_thick_transition_data(df_species, line_optdepth_col):
    df_species_inter2 = df_species[df_species[line_optdepth_col] >= 1]

    idx_optically_thick = (
        df_species_inter2.groupby(['nH', 'radm'])["AV"].transform(min) == df_species_inter2["AV"]
    )
    df_species_inter2 = df_species_inter2[idx_optically_thick]

    df_species_inter2 = df_species_inter2.set_index(['nH', 'radm'])
    df_species_inter2.columns = [f"{col}_optically_thick" for col in df_species_inter2.columns]
    df_species_inter2 = df_species_inter2.reset_index()
    return df_species_inter2


def combine_dataframes(df_species_inter, df_species_inter2):
    df_species_gb = pd.merge(df_species_inter, df_species_inter2, on=['nH', "radm"], how="left")

    df_species_gb["AV_eff"] = df_species_gb.apply(
        lambda row: np.nanmin([row["AV_density_peak"], row["AV_optically_thick"]]), 
        axis=1
    )
    
    df_species_gb = df_species_gb.groupby(["nH", "radm"]).mean().reset_index()
    assert df_species_gb.groupby(["nH", "radm"])["AV_density_peak"].count().max() == 1
    return df_species_gb


def get_data_at_effective_emission_depth(df_species_gb, df_species, density_col, line_optdepth_col):
    df_species_final = pd.merge(
        df_species_gb, df_species, 
        left_on=['nH', 'radm', 'AV_eff'], right_on=['nH', 'radm', 'AV']
    )

    df_species_final = df_species_final.drop('AV', 1)
    df_species_final = df_species_final.rename(columns={
        "Temperature":"Temperature_at_AV_eff",
        density_col:f"{density_col}_at_AV_eff",
        line_optdepth_col:f"{line_optdepth_col}_at_AV_eff",
    })

    df_species_final = np.log10(df_species_final)
    df_species_final = df_species_final.set_index(["nH", "radm"]).sort_index()
    return df_species_final



def add_missing_points_in_grid(df_incomplete_grid):
    X = np.unique(df_incomplete_grid.index.get_level_values(0))
    Y = np.unique(df_incomplete_grid.index.get_level_values(1))
    xx, yy = np.meshgrid(X, Y)
    xx = xx.T
    yy = yy.T

    new_index = pd.MultiIndex.from_product(
        [X, Y], names=df_incomplete_grid.index.names)

    df_new_index = pd.DataFrame(index=new_index)

    df_filled_grid = pd.merge(df_incomplete_grid, df_new_index, left_index=True, right_index=True, how="outer")
    return df_filled_grid, xx, yy


def extract_data_process(df, transition, species, upper_state):
    df_species, density_col, line_optdepth_col = extract_all_transition_data(df, transition, species, upper_state)

    df_density_peak = get_density_peak_data(df_species, density_col)
    df_opt_thick = get_optically_thick_transition_data(df_species, line_optdepth_col)

    df_species_gb = combine_dataframes(df_density_peak, df_opt_thick)
    df_species_final = get_data_at_effective_emission_depth(df_species_gb, df_species, density_col, line_optdepth_col)
    return df_species_final, density_col, line_optdepth_col


def plot_all(n_levels, transition,species,upper_state,density_col, line_optdepth_col, df_filled_grid_integrated,xx_integrated,     
        yy_integrated,df_filled_grid_non_integrated, xx_non_integrated, yy_non_integrated, list_transitions):
    n_rows = 4
    n_cols = 4

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12,8), sharex=True, sharey=True)


    titles = [[None, r"$\log$ cd" + f" {species}", r"$\log$ " + f"I({transition})", None]]
    partial_cols_plotted = [[None, f"cd {species}", f"I({species}", None]]


    id_row = 0
    for id_col in [1,2]:
        ax[id_row, id_col].set_title(titles[id_row][id_col])

        col_raw = partial_cols_plotted[id_row][id_col]
        cols = [col for col in df_filled_grid_integrated.columns if col_raw in col]
        
        if len(cols) > 1:
            list_transitions_species = [col for col in list_transitions if species in col]
            idx = list_transitions_species.index(transition)
            col_name = cols[idx]

        else:
            col_name = cols[0]

        Z = df_filled_grid_integrated[col_name].values.reshape(*xx_integrated.shape)
        im = ax[id_row, id_col].contour(xx_integrated, yy_integrated, Z, levels=n_levels)
        ax[id_row, id_col].clabel(im, inline=True, fontsize=10)



    titles = [
        [None, None, None, None],
        [r"$\log A_{V, eff}$", r"$T$ at $A_{V, eff}$", r"$n_u$ at $A_{V, eff}$", r"$\log \tau$ at $A_{V, eff}$"],
        [r"$\log A_V$ where $\tau = 1$", r"$T$ at $\tau = 1$", r"$n_u$ at $\tau = 1$", r"$\log \tau$ at $\tau = 1$"],
        [r"$\log A_V$ at density peak", r"$T$ at density peak", r"$n_u$ at density peak", r"$\log \tau$ at density peak"],
    ]

    cols_plotted = [
        [None, None, None, None],
        ["AV_eff", "Temperature_at_AV_eff", f"{density_col}_at_AV_eff", f"{line_optdepth_col}_at_AV_eff"],
        ["AV_optically_thick", "Temperature_optically_thick", f"{density_col}_optically_thick", f"{line_optdepth_col}_optically_thick"],
        ["AV_density_peak", "Temperature_density_peak", f"{density_col}_density_peak", f"{line_optdepth_col}_density_peak"],
    ]

    for id_row in range(1, n_rows):
        for id_col in range(n_cols):
            ax[id_row, id_col].set_title(titles[id_row][id_col])
            #ax[id_row, id_col].plot(df_new_index.index.get_level_values(0), df_new_index.index.get_level_values(1), 'k+', ms=3)

            Z = df_filled_grid_non_integrated[cols_plotted[id_row][id_col]].values.reshape(*xx_non_integrated.shape)
            im = ax[id_row, id_col].contour(xx_non_integrated, yy_non_integrated, Z, levels=n_levels)
            ax[id_row, id_col].clabel(im, inline=True, fontsize=10)


    for i in range(n_rows):
        ax[i,0].set_ylabel(r"$\log G_0$")

    for j in range(n_cols):
        ax[n_rows-1, j].set_xlabel(r"$\log n_H$")


    fig.tight_layout()
    plt.show()
    


def extract_and_plot(df, df_integrated, list_transitions, transition, n_levels):
    species = transition.split(" ")[0]
    upper_state = transition.split(" ")[1].split("->")[0]

    # extract and shape all the necessary data for non integrated data
    df_species_final, density_col, line_optdepth_col = extract_data_process(df, transition, species, upper_state)

    # the grid might have some missing points, and we need to have a complete (though maybe not uniform) grid
    df_filled_grid_integrated, xx_integrated, yy_integrated = add_missing_points_in_grid(df_integrated)
    df_filled_grid_non_integrated, xx_non_integrated, yy_non_integrated = add_missing_points_in_grid(df_species_final)

    # final plots
    plot_all(
        n_levels=n_levels,
        transition=transition,
        species=species,
        upper_state=upper_state,
        density_col=density_col, 
        line_optdepth_col=line_optdepth_col, 
        df_filled_grid_integrated=df_filled_grid_integrated,
        xx_integrated=xx_integrated, 
        yy_integrated=yy_integrated,
        df_filled_grid_non_integrated=df_filled_grid_non_integrated, 
        xx_non_integrated=xx_non_integrated, 
        yy_non_integrated=yy_non_integrated,
        list_transitions=list_transitions
    )


if __name__ == '__main__':
    # import integrated data
    filename_integrated = "./data/PDR17G1E20_n_cte_with_cd.dat"

    df_integrated = read_extraction_result_dat(filename_integrated)
    df_integrated = df_integrated.apply(lambda x: np.log10(x))
    df_integrated = df_integrated[df_integrated["[003]Avmax"] == 1.]
    df_integrated = df_integrated.set_index(["[001]nH", "[002]radm"])


    # import non integrated data
    df_non_integrated_all = scrape_all_files()

    list_transitions = [
        col.replace("Line optical depth observer side(", "")[:-1] 
        for col in df_non_integrated_all.columns if "Line optical " in col
    ]
    transition = list_transitions[0]

    # extract relevant non integrated features and plot result
    extract_and_plot(df_non_integrated_all, df_integrated, list_transitions, transition, n_levels=10)


