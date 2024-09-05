from django.db import models

# Create your models here.


import warnings

warnings.filterwarnings("ignore")

# import pandas
import json
import numpy as np
import matplotlib.pyplot as plt


data_source = "https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename="

if False:
    print("loading datasources")
    print("   world infected")
    data_infected_by_region = pandas.read_csv(
        "https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv"
    )

    print("   world recovered")
    data_recovered_by_region = pandas.read_csv(
        "https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv"
    )

    print("   world deaths")
    data_deaths_by_region = pandas.read_csv(
        "https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv"
    )
else:
    print("loading datasources")
    print("   world infected")
    data_infected_by_region = pandas.read_csv(
        "/home/mauriciomatera/mysite/covid/data/time_series_covid19_confirmed_global.csv"
    )

    print("   world recovered")
    data_recovered_by_region = pandas.read_csv(
        "/home/mauriciomatera/mysite/covid/data/time_series_covid19_recovered_global.csv"
    )

    print("   world deaths")
    data_deaths_by_region = pandas.read_csv(
        "/home/mauriciomatera/mysite/covid/data/time_series_covid19_deaths_global.csv"
    )


print("   Argentina confirmed ")
data_infected_arg_by_region = pandas.read_csv(
    "/home/mauriciomatera/mysite/covid/data/covid19-argentina-confirmed.csv"
)
print("   Argentina recovered ", type(data_infected_arg_by_region))
data_recovered_arg_by_region = pandas.read_csv(
    "/home/mauriciomatera/mysite/covid/data/covid19-argentina-recovered.csv"
)
data_deaths_arg_by_region = pandas.read_csv(
    "/home/mauriciomatera/mysite/covid/data/covid19-argentina-deaths.csv"
)

print("   Populations ")
populations = pandas.read_csv(
    "https://raw.githubusercontent.com/datasets/population/master/data/population.csv"
)

populations[populations["Country Name"] == "United States"]
# populations=pandas.concat([populations]+[pandas.DataFrame([["US","USA",2020, 323127513]],
#                                                          columns=["Country Name", "Country Code","Year","Value"])],
#                          ignore_index=True)
# populations.append({"Country Name":"US","Country Code":"USA", "Year":2020,"Value":323127513.},ignore_index=True )
# populations=populations.append({"Country Name":"Korea, South","Country Code":"SKorea", "Year":2020,"Value":51606633},ignore_index=True )
# populations=pandas.concat([populations]+[pandas.DataFrame([["Korea, South","SKorea",2020, 51606633]],
#                                                          columns=["Country Name", "Country Code","Year","Value"])],
#                          ignore_index=True)


countries = {
    "Argentina",
    "France",
    "Germany",
    "Schwitzeland",
    "Israel",
    "New Zealand",
    "Russia",
    "Kenya",
    "Congo (Kinshasa)",
    "Egypt",
    "Madagascar",
    "South Africa",
    "Ethiopia",
    "Australia",
    "India",
    "Korea, South",
    "Finland",
    "Norway",
    "Uruguay",
    "Venezuela",
    "Columbia",
    "Chile",
    "Brazil",
    "Bolivia",
    "Paraguay",
    "China",
    "Spain",
    "Italy",
    "United Kingdom",
    "US",
    "Sweden",
}


nodes = {}

argprovinces = (
    "Argentina",
    "Buenos Aires",
    "Catamarca",
    "Chaco",
    "Chubut",
    "Córdoba",
    "Corrientes",
    "Entre Ríos",
    "Formosa",
    "Jujuy",
    "La Pampa",
    "La Rioja",
    "Mendoza",
    "Misiones",
    "Neuquén",
    "Río Negro",
    "Salta",
    "San Juan",
    "San Luis",
    "Santa Cruz",
    "Santa Fe",
    "Santiago del Estero",
    "Tierra del Fuego",
    "Tucumán",
    "CABA",
)

for province in argprovinces:
    nodename = province
    if nodename == "Argentina":
        nodename = "Argentina-total"
    else:
        nodename = "Argentina-" + nodename
    nodes[nodename] = {
        "coordinates": (0, 0),
        "population": 4000000,
        "initial_infected": None,
        "initial_day": 125,
        "aexp": None,
        "err_aexp": None,
        "initial_immunized": 0,
        "peak": None,
        "tinc": None,
        "curve_confirmed": data_infected_arg_by_region[province].to_numpy(),
        "curve_recovered": data_recovered_arg_by_region[province].to_numpy(),
        "curve_deaths": data_deaths_arg_by_region[province].to_numpy(),
    }


for country in countries:
    data_infected_country = data_infected_by_region[
        data_infected_by_region["Country/Region"] == country
    ]
    regions = data_infected_country.filter(regex="Province/State")
    regions = regions.to_numpy()[:, 0]
    # print("country:",country)
    data_infected_country = data_infected_by_region[
        data_infected_by_region["Country/Region"] == country
    ]
    data_deaths_country = data_deaths_by_region[
        data_deaths_by_region["Country/Region"] == country
    ]
    data_recovered_country = data_recovered_by_region[
        data_recovered_by_region["Country/Region"] == country
    ]
    for region in regions:
        key = country
        if region is not np.nan:
            population = np.nan
            key = key + "-"
            key = key + str(region)
            # data = data_deaths_by_region[data_deaths_by_region["Province/State"]==region]
            data_infected = data_infected_country[
                data_infected_country["Province/State"] == region
            ]
            data_deaths = data_deaths_country[
                data_deaths_country["Province/State"] == region
            ]
            data_recovered = data_recovered_country[
                data_recovered_country["Province/State"] == region
            ]
        else:
            try:
                population = (
                    populations[populations["Country Name"] == country]
                    .sort_values("Year")
                    .filter(["Value"])
                    .iloc[[-1]]
                    .to_numpy()[0, 0]
                )
            except:
                population = -1
            data_infected = data_infected_country[
                data_infected_country["Province/State"].isnull()
            ]
            data_deaths = data_deaths_country[
                data_deaths_country["Province/State"].isnull()
            ]
            data_recovered = data_recovered_country[
                data_recovered_country["Province/State"].isnull()
            ]

        coords = np.array(
            [data_infected.Long.to_numpy()[0], data_infected.Lat.to_numpy()[0]]
        )
        data_confirmed = np.array(data_infected.to_numpy()[0, 4:], dtype=float)
        data_deaths = np.array(data_deaths.to_numpy()[0, 4:], dtype=float)
        data_recovered = np.array(data_recovered.to_numpy()[0, 4:], dtype=float)

        # From the day 559, recovered are not reported anymore. Estimate them by the infected 30 days before.
        data_recovered[559:] = data_confirmed[529:-30]
        data_infected = data_confirmed - data_recovered - data_deaths

        # Curvas de nuevos recuperados e infectados diarios
        newrecovered = (
            data_recovered[1:]
            - data_recovered[:-1]
            + data_deaths[1:]
            - data_deaths[:-1]
        )
        newinfecteds = data_confirmed[1:] - data_confirmed[:-1]
        # Para calcular el tiempo de recuperación, asumo que la curva de nuevos infectados y
        # de recuperados tienen la misma distribución, ya que en el modelo T son la misma cantidad
        # desplazada en el tiempo en un plazo igual al tiempo de recuperación.
        # Luego, el tiempo de recuperación puedo estimarlo como el valor medio de $t$
        # pesado con la curva de recuperados, menos el valor medio pesado con la curva de nuevos
        # infectados.
        days = np.array(range(len(newrecovered)))
        try:
            tinc = np.dot(newrecovered, days) / np.sum(newrecovered) - np.dot(
                newinfecteds, days
            ) / np.sum(newinfecteds)
            tinc = int(round(tinc))
            if tinc < 3:
                tinc = 12
        except:
            tinc = 12

        history = np.array(
            [np.array(range(len(data_infected))), data_infected]
        ).transpose()
        peak = sorted(history, key=lambda x: -x[1])[0]
        tmax = peak[0]
        history = history[history[:, 1] != 0]
        initial_day = int(history[0][0])
        last_day = int(history[-1][0])

        newinfecteds = newinfecteds[initial_day + 7 : last_day]

        data_infected_exp = (
            1.0 + newinfecteds / data_infected[initial_day + 7 : last_day]
        )
        data_infected_exp = data_infected_exp[data_infected_exp is not np.nan][0]
        data_infected_exp = data_infected_exp[data_infected_exp != np.inf][0]
        infection_rate = np.average(data_infected_exp)
        if infection_rate == np.inf:
            print(data_infected_exp)
            infection_rate = np.nan
        else:
            err_infection_rate = np.sqrt(
                np.average(data_infected_exp**2) / infection_rate**2 - 1
            )
            if err_infection_rate > 0.05:
                pass  # print("Warning: for ",key," ",infection_rate," +/-",100*err_infection_rate, "%. Probably there was a change of exponent")
                # print(data_infected_exp)
                # plt.plot(data_infected_exp,label=key)
                # plt.legend()
                # plt.show()

        nodes[key] = {
            "coordinates": coords,
            "population": -1 if np.isnan(population) else int(population),
            "initial_infected": data_infected[initial_day],
            "initial_day": initial_day,
            "aexp": infection_rate,
            "err_aexp": err_infection_rate,
            "initial_immunized": 0,
            "peak": peak,
            "tinc": tinc,
            "curve_confirmed": data_confirmed,
            "curve_recovered": data_recovered,
            "curve_deaths": data_deaths,
        }

try:
    nodes["China-Guangdong"]["population"] = 104303132
    nodes["China-Shandong"]["population"] = 95793065
    nodes["China-Henan"]["population"] = 94023567
    nodes["China-Sichuan"]["population"] = 80418200
    nodes["China-Jiangsu"]["population"] = 78659903
    nodes["China-Hebei"]["population"] = 71854202
    nodes["China-Hunan"]["population"] = 65683722
    nodes["China-Anhui"]["population"] = 59500510
    nodes["China-Hubei"]["population"] = 57237740
    nodes["China-Hubei"]["population"] = 57237740
    nodes["China-Zhejiang"]["population"] = 54426891
    nodes["China-Hong Kong"]["population"] = 7392000
    nodes["China-Beijing"]["population"] = 21540000
    nodes["China-Fujian"]["population"] = 28560000
    nodes["China-Gansu"]["population"] = 28560000

    nodes["China-Chongqing"]["population"] = 30480000
    nodes["China-Guangxi"]["population"] = 48380000
    nodes["China-Guizhou"]["population"] = 34750000
    nodes["China-Hainan"]["population"] = 9258000
    nodes["China-Heilongjiang"]["population"] = 38310000
    nodes["China-Inner Mongolia"]["population"] = 24710000
    nodes["China-Jiangxi"]["population"] = 45200000
    nodes["China-Jilin"]["population"] = 27460000
    nodes["China-Liaoning"]["population"] = 43900000
    nodes["China-Macau"]["population"] = 622567
    nodes["China-Ningxia"]["population"] = 6300000
    nodes["China-Qinghai"]["population"] = 5627000
    nodes["China-Shaanxi"]["population"] = 37330000
    nodes["China-Shanghai"]["population"] = 24280000
    nodes["China-Shanxi"]["population"] = 36500000
    nodes["China-Tianjin"]["population"] = 11558000
    nodes["China-Tibet"]["population"] = 3180000
    nodes["China-Xinjiang"]["population"] = 21810000
    nodes["China-Yunnan"]["population"] = 45970000
except:
    pass

# Salvar los datos a un array de json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open("/home/mauriciomatera/mysite/covid/data/data.json", "w") as f:
    f.write(json.dumps(nodes, sort_keys=True, cls=NumpyEncoder))


def mostrar_curva2(region, mostrar_ajuste, start, end):
    """Inversa de la Tasa de infección"""
    data_confirmed = nodes[region]["curve_confirmed"]
    data_death = nodes[region]["curve_deaths"]
    data_recovered = nodes[region]["curve_recovered"]
    activos = data_confirmed - data_death - data_recovered
    newrecovered = (
        data_recovered[1:] - data_recovered[:-1]
    )  # +data_deaths[1:]-data_deaths[:-1]
    newdeaths = data_death[1:] - data_death[:-1]  # +data_deaths[1:]-data_deaths[:-1]
    newinfected = data_confirmed[1:] - data_confirmed[:-1]
    newconfirmed = data_confirmed[1:] - data_confirmed[:-1]
    print(len(activos), len(newinfected))
    ts = np.array(range(len(newinfected)))
    values = newconfirmed / activos[1:]
    activos = activos[:-1]
    assert len(values) == len(
        activos
    ), "activos y values deberían tener la misma longitud."
    print([start, end])
    mask = (
        (ts >= start)
        & (ts <= end)
        & (~np.isnan(values))
        & (~np.isnan(activos))
        & (values > 0)
    )
    ts = ts[mask]
    values = np.log(2) / values[mask]
    activos = activos[mask]
    print(len(activos), len(ts), len(values))
    print(ts)
    print(values)
    # ts = ts[start:end]
    # values = values[start:end]
    # ts = ts[(~np.isnan(values))]
    # values = values[(~np.isnan(values))]
    # ts = ts[values>0]
    # values = np.log(2)/values[values>0]

    # activos = activos[1:]
    # ts = ts[start:end]
    # activos = activos[start:end]
    # ts = ts[(~np.isnan(activos))]
    # activos = values[(~np.isnan(activos))]

    result = {
        "ts": [int(t) for t in ts],
        "values": list(values),
        "activos": list(activos[1:]),
    }

    plt.scatter(ts, values, label=region)
    plt.plot(ts, 0 * ts + 21.0, ls="-.")
    try:
        lnvalues = np.log(values)
        fit = np.polyfit(ts - ts[0], lnvalues, 1, cov=False)
        result["Tau"] = 1 / fit[0]
        result["y0"] = fit[1]
        if mostrar_ajuste:
            result["fit"] = list(np.exp(fit[0] * (ts - ts[0]) + fit[1]))
    except:
        pass
    return result


def mostrar_curva3(region, start, end, tipoescalal, tipoescalar, mostrar_ajuste):
    """Correlación entre $\\tau$ estimado y crecimiento de casos activos"""
    data_confirmed = nodes[region]["curve_confirmed"]
    data_death = nodes[region]["curve_deaths"]
    data_recovered = nodes[region]["curve_recovered"]
    activos = data_confirmed - data_death - data_recovered
    newrecovered = (
        data_recovered[1:] - data_recovered[:-1]
    )  # +data_deaths[1:]-data_deaths[:-1]
    newdeaths = data_deaths[1:] - data_deaths[:-1]  # +data_deaths[1:]-data_deaths[:-1]
    newinfected = data_confirmed[1:] - data_confirmed[:-1]
    newconfirmed = data_confirmed[1:] - data_confirmed[:-1]
    ts = np.array(range(len(newinfected)))

    values = newconfirmed / activos[1:]
    ts = ts[start:end]
    values = values[start:end]
    ts = ts[(~np.isnan(values))]
    values = values[(~np.isnan(values))]
    ts = ts[values > 0]
    values = np.log(2) / values[values > 0]
    lnvalues = np.log(values)
    fig, ax1 = plt.subplots()
    ln1 = ax1.scatter(ts, values, label="$\\tau$", color="blue")
    ax1.plot(ts, 0 * ts + 21.0, ls="-.")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("$\\tau$")

    ax2 = ax1.twinx()

    activos = data_confirmed - data_death - data_recovered
    ts = np.array(range(len(activos)))
    values = activos[start:end]
    ts = ts[start:end]
    ts = ts[(~np.isnan(values))]
    values = values[(~np.isnan(values))]
    ts = ts[values > 0]
    values = values[values > 0]
    lnvalues = np.log(values)
    ln2 = ax2.scatter(ts, values, label="casos activos", c="red")
    ax2.set_ylabel("Casos activos")

    try:
        fit = np.polyfit(ts - ts[0], lnvalues, 1, cov=False)
        print(
            "Tiempo característico :",
            1 / fit[0],
            "dias.  Ordenada a t_0:",
            np.exp(fit[1]),
        )
        if mostrar_ajuste:
            plt.plot(
                ts, np.exp(fit[0] * (ts - ts[0]) + fit[1]), label="ajuste", color="red"
            )
    except:
        pass
    ax1.set_yscale(tipoescalal)
    ax2.set_yscale(tipoescalar)
    # plt.ylabel("$T_{duplicaci\'on}$")
    # plt.plot(np.log(newrecovered),label="nuevos recuperados")
    # plt.plot(np.log(newdeaths),label="nuevos muertos")
    ax1.set_title("Correlación entre $\\tau$ estimado y crecimiento de casos activos")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.show()
