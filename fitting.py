import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

import errors as err


def read_data(file_paths, selected_country, start_year, end_year):
    dataframes_list = []

    for path in file_paths:
        file_name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, skiprows=4)
        df = df.rename(columns={'Country Name': 'Country'})
        df = df.set_index('Country')
        df_selected_country = df.loc[selected_country, str(start_year):str(end_year)].transpose().reset_index()
        df_selected_country = df_selected_country.rename(columns={'index': 'Year', selected_country: file_name})
        dataframes_list.append(df_selected_country)

    # Concatenate all DataFrames based on the 'Year' column
    result_df = pd.concat(dataframes_list, axis=1)

    # Replace null values with the mean of each column
    result_df = result_df.apply(lambda col: col.fillna(col.mean()))

    return result_df


def exp_growth(t, scale, growth):
    """Computes exponential function with scale and growth as free parameters"""
    f = scale * np.exp(growth * t)
    return f


def logistics(t, a, k, t0):
    """Computes logistics function with scale and incr as free parameters"""
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f


def fit_and_plot_growth(file_paths, selected_country, start_year, end_year, model_func, plot_title, save_filename):
    df = read_data(file_paths, selected_country, start_year, end_year)

    df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
    df["Agricultural land"] = pd.to_numeric(df["Agricultural land"], errors='coerce')

    initial_guess = [1.0, 0.02]
    popt, pcovar = opt.curve_fit(model_func, df["Year"], df["Agricultural land"], p0=initial_guess, maxfev=10000)

    df["pop_model"] = model_func(df["Year"], *popt)

    plt.figure()
    plt.plot(df["Year"], df["Agricultural land"], label="data")
    plt.plot(df["Year"], df["pop_model"], label="fit")
    plt.legend()
    plt.title(plot_title)
    plt.show()

    # Call function to calculate upper and lower limits with extrapolation
    years = np.linspace(start_year, end_year + 10)
    pop_model_growth = model_func(years, *popt)
    sigma = err.error_prop(years, model_func, popt, pcovar)
    low = pop_model_growth - sigma
    up = pop_model_growth + sigma

    plt.figure()
    plt.title(f"Agricultural land of {selected_country} in {end_year + 10}", fontweight='bold')
    plt.plot(df["Year"], df["Agricultural land"], label="data")
    plt.plot(years, pop_model_growth, label="fit")
    plt.fill_between(years, low, up, alpha=0.3, color="y", label="95% Confidence Interval")
    plt.legend(loc="upper left")
    plt.xlabel("Year", fontweight='bold')
    plt.ylabel("Agricultural land", fontweight='bold')
    plt.savefig(save_filename, dpi=300)
    plt.show()

    # Predict future values
    pop_future = model_func(np.array([end_year + 10]), *popt)
    sigma_future = err.error_prop(np.array([end_year + 10]), model_func, popt, pcovar)
    print(f"Agricultural land in {end_year + 10}: {pop_future[0] / 1.0e6} Mill.")

    # For the next 10 years
    print(f"Agricultural land in the next 10 years:")
    for year in range(end_year + 1, end_year + 11):
        pop_year = model_func(year, *popt) / 1.0e6
        print(f"{year}: {pop_year} Mill.")


# Example usage for "Arab World"
file_paths_arab_world = ['Agricultural land.csv']
fit_and_plot_growth(file_paths_arab_world, "Arab World", 1960, 2021, exp_growth,
                    "Data Fit Attempt for Arab World", 'Agricultural_land_Arab_World.png')

# Example usage for "Pakistan"
file_paths_pakistan = ['Agricultural land.csv']
fit_and_plot_growth(file_paths_pakistan, "Pakistan", 1960, 2021, exp_growth,
                    "Data Fit Attempt for Pakistan", 'Agricultural_land_Pakistan.png')
