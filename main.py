import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.stats import ttest_ind

#Ejercicio 1

np.random.seed(42)


EFECTO_ANUAL = {
    2023: 1000,
    2024: 1500,
    2025: 2000
}

EFECTO_MENSUAL = {
    1: 1000,
    2: 1500,
    3: 2000,
    4: 2000,
    5: 2500,
    6: 2500,
    7: 3000,
    8: 2500,
    9: 2500,
    10: 2000,
    11: 1500,
    12: 1000
}

EFECTO_DIARIO = {
    0: 2000,
    1: 3000,
    2: 3500,
    3: 3000,
    4: 2000,
    5: 1000,
    6: 1000
}

EFECTO_TIENDA = {
    "Santa Ana": 5000,
    "La Floresta": 2000,
    "Los Cedros": 3000,
    "Palermo": 1000,
    "Córdoba": 3000
}


def calcular_lambda(fecha, tienda):
    año = fecha.year
    mes = fecha.month
    dia_semana = fecha.weekday()

    lambda_t = (
        EFECTO_ANUAL[año]
        + EFECTO_MENSUAL[mes]
        + EFECTO_DIARIO[dia_semana]
        + EFECTO_TIENDA[tienda]
    )

    return lambda_t


def generar_datos():
    fechas = pd.date_range(
        start="2023-01-01",
        end="2025-12-31",
        freq="D"
    )

    datos = []

    for fecha in fechas:
        for tienda in EFECTO_TIENDA:
            lambda_t = calcular_lambda(fecha, tienda)
            clientes = np.random.poisson(lambda_t)

            datos.append({
                "fecha": fecha,
                "anño": fecha.year,
                "mes": fecha.month,
                "dia_semana": fecha.weekday(),
                "tienda": tienda,
                "lambda": lambda_t,
                "clientes": clientes
            })

    return pd.DataFrame(datos)


def guardar_datos(df):
    df.to_csv("datos_simulados.csv", index=False, encoding="utf-8-sig")
    print("Archivo datos_simulados.csv generado correctamente.")




def main():
    df = generar_datos()

    print(df.head())
    print(df.shape)

    guardar_datos(df)

    intervalos = calcular_intervalos_santa_ana(df)

    print("\nIntervalos empíricos para Santa Ana:")
    print(intervalos)

    guardar_intervalos(intervalos)

    graficar_promedio_mensual(intervalos)
    graficar_intervalos(intervalos)

    resumen = resumen_por_tienda(df)

    boxplot_tiendas(df)
     
    media_general = calcular_media_general(df)
    ssw = calcular_ssw(df)
    ssb = calcular_ssb(df, media_general)
    sst = calcular_sst(df, media_general)
    print("\nVerificación de la identidad fundamental:")
    print(f"SST = {sst:.2f}")
    print(f"SSB + SSW = {(ssb + ssw):.2f}")
    print(f"Diferencia = {sst - (ssb + ssw):.8f}")
    f_anova, p_valor_anova = realizar_anova(df, ssb, ssw)

    

    datos_mayor, datos_menor, tienda_mayor, tienda_menor = obtener_extremos(df)

    prueba_t_student(
      datos_mayor,
      datos_menor,
      tienda_mayor,
      tienda_menor
)

#Ejercicio 2

def calcular_intervalos_santa_ana(df):
    df_santa_ana = df[df["tienda"] == "Santa Ana"]

    intervalos = (
        df_santa_ana
        .groupby("mes")["clientes"]
        .agg(
            promedio="mean",
            limite_inferior_95=lambda x: x.quantile(0.025),
            limite_superior_95=lambda x: x.quantile(0.975),
            limite_inferior_99=lambda x: x.quantile(0.005),
            limite_superior_99=lambda x: x.quantile(0.995)
        )
        .reset_index()
    )

    return intervalos


def guardar_intervalos(intervalos):
    intervalos.to_csv(
        "intervalos_santa_ana.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Archivo intervalos_santa_ana.csv generado correctamente.")


def graficar_promedio_mensual(intervalos):
    nombres_meses = [
        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(
        nombres_meses,
        intervalos["promedio"],
        marker="o",
        linewidth=2
    )

    plt.title("Promedio mensual de clientes - Santa Ana")
    plt.xlabel("Mes")
    plt.ylabel("Clientes promedio")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("informe/promedio_mensual_santa_ana.png", dpi=300)

    plt.close()

def graficar_intervalos(intervalos):

    nombres_meses = [
        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"
    ]

    plt.figure(figsize=(10,5))

    plt.plot(
        nombres_meses,
        intervalos["promedio"],
        marker="o",
        label="Promedio"
    )

    plt.fill_between(
        nombres_meses,
        intervalos["limite_inferior_95"],
        intervalos["limite_superior_95"],
        alpha=0.25,
        label="Intervalo empírico 95%"
    )

    plt.title("Clientes diarios - Santa Ana")
    plt.xlabel("Mes")
    plt.ylabel("Clientes")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("informe/intervalo_95_santa_ana.png", dpi=300)

    plt.close()


#Ejercicio 3:

from scipy.stats import f_oneway


def resumen_por_tienda(df):
    resumen = (
        df.groupby("tienda")["clientes"]
        .agg(
            promedio="mean",
            desvio="std",
            minimo="min",
            maximo="max",
            cantidad="count"
        )
        .round(2)
        .reset_index()
    )

    print("\nResumen estadístico por tienda:")
    print(resumen)

    resumen.to_csv(
        "resumen_por_tienda.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return resumen

def boxplot_tiendas(df):

    plt.figure(figsize=(10,6))

    df.boxplot(
        column="clientes",
        by="tienda",
        grid=False
    )

    plt.title("Distribución de clientes por supermercado")
    plt.suptitle("")
    plt.xlabel("Supermercado")
    plt.ylabel("Clientes diarios")

    plt.tight_layout()

    plt.savefig(
        "informe/boxplot_supermercados.png",
        dpi=300
    )

    plt.close()

def calcular_media_general(df):

    media_general = df["clientes"].mean()

    print(f"\nMedia general: {media_general:.2f}")

    return media_general

def calcular_ssw(df):
    medias_por_tienda = df.groupby("tienda")["clientes"].transform("mean")

    ssw = ((df["clientes"] - medias_por_tienda) ** 2).sum()

    print(f"SSW - Variabilidad dentro de las tiendas: {ssw:.2f}")

    return ssw

def calcular_ssb(df, media_general):

    resumen = (
        df.groupby("tienda")["clientes"]
        .agg(
            media="mean",
            n="count"
        )
        .reset_index()
    )

    resumen["diferencia"] = resumen["media"] - media_general

    resumen["diferencia2"] = resumen["diferencia"] ** 2

    resumen["contribucion"] = (
        resumen["n"] * resumen["diferencia2"]
    )

    print("\nConstrucción de SSB")
    print(resumen)

    ssb = resumen["contribucion"].sum()

    print(f"\nSSB = {ssb:.2f}")

    return ssb

def calcular_sst(df, media_general):

    sst = ((df["clientes"] - media_general) ** 2).sum()

    print(f"\nSST = {sst:.2f}")

    return sst

def realizar_anova(df, ssb, ssw):
    k = df["tienda"].nunique()
    n = len(df)

    gl_entre = k - 1
    gl_dentro = n - k

    msb = ssb / gl_entre
    msw = ssw / gl_dentro

    f_manual = msb / msw

    grupos = [
        df[df["tienda"] == tienda]["clientes"]
        for tienda in df["tienda"].unique()
    ]

    f_scipy, p_valor = f_oneway(*grupos)

    print("\nANOVA")
    print(f"Grados de libertad entre grupos: {gl_entre}")
    print(f"Grados de libertad dentro de grupos: {gl_dentro}")
    print(f"MSB: {msb:.2f}")
    print(f"MSW: {msw:.2f}")
    print(f"F calculado manualmente: {f_manual:.4f}")
    print(f"F calculado con scipy: {f_scipy:.4f}")
    print(f"p-valor: {p_valor:.10f}")

    if p_valor < 0.05:
        print("Conclusión: se rechaza H0. Existen diferencias significativas entre las tiendas.")
    else:
        print("Conclusión: no se rechaza H0. No hay evidencia suficiente de diferencias entre tiendas.")

    return f_manual, p_valor

#Ejercicio 4

def obtener_extremos(df):

    resumen = (
        df.groupby("tienda")["clientes"]
        .mean()
        .sort_values()
    )

    tienda_menor = resumen.index[0]
    tienda_mayor = resumen.index[-1]

    datos_mayor = df[df["tienda"] == tienda_mayor]["clientes"]
    datos_menor = df[df["tienda"] == tienda_menor]["clientes"]

    print("\nTiendas seleccionadas")
    print("----------------------")
    print(f"Mayor promedio : {tienda_mayor}")
    print(f"Menor promedio : {tienda_menor}")

    print(f"\nMedia {tienda_mayor}: {datos_mayor.mean():.2f}")
    print(f"Media {tienda_menor}: {datos_menor.mean():.2f}")

    return datos_mayor, datos_menor, tienda_mayor, tienda_menor

def prueba_t_student(datos_mayor,
                     datos_menor,
                     tienda_mayor,
                     tienda_menor):

    t, p = ttest_ind(
        datos_mayor,
        datos_menor,
        equal_var=True
    )

    print("\nPrueba t de Student")
    print("---------------------------")

    print(f"Tienda 1 : {tienda_mayor}")
    print(f"Tienda 2 : {tienda_menor}")

    print(f"\nEstadístico t = {t:.4f}")
    print(f"p-valor = {p:.10e}")

    alpha = 0.05

    if p < alpha:
        print("\nSe rechaza H₀.")
        print("Existe una diferencia estadísticamente significativa entre ambas tiendas.")
    else:
        print("\nNo se rechaza H₀.")
        print("No existe evidencia suficiente para afirmar que las medias sean diferentes.")


if __name__ == "__main__":
    main()