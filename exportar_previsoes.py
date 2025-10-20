# ===============================================
# Exportar Previsões SARIMAX para CSV
# Modelo: 48 observações completas
# Autor: Eduardo Piaia
# Data: 19/10/2025
# ===============================================

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("EXPORTAÇÃO DE PREVISÕES - SARIMAX(1,1,1)x(1,1,0,6)")
print("="*80)

# =============================
# CARREGAR DADOS
# =============================
data = {
    "Data": [
        "01/10/2021","01/11/2021","01/12/2021","01/01/2022","01/02/2022","01/03/2022","01/04/2022",
        "01/05/2022","01/06/2022","01/07/2022","01/08/2022","01/09/2022","01/10/2022","01/11/2022",
        "01/12/2022","01/01/2023","01/02/2023","01/03/2023","01/04/2023","01/05/2023","01/06/2023",
        "01/07/2023","01/08/2023","01/09/2023","01/10/2023","01/11/2023","01/12/2023","01/01/2024",
        "01/02/2024","01/03/2024","01/04/2024","01/05/2024","01/06/2024","01/07/2024","01/08/2024",
        "01/09/2024","01/10/2024","01/11/2024","01/12/2024","01/01/2025","01/02/2025","01/03/2025",
        "01/04/2025","01/05/2025","01/06/2025","01/07/2025","01/08/2025","01/09/2025"
    ],
    "Log_Total": [
        12.4989259,12.62505344,13.0865333,12.71736798,12.59716361,12.59900062,12.60991113,12.61402453,12.67713328,12.74550185,
        12.76396815,12.7941308,12.81536234,12.87812499,13.31000617,12.81007755,12.88327945,12.77650682,12.79090565,12.84677424,
        12.88478976,12.88450284,12.9171095,12.9357015,12.93131233,13.02418651,13.4592378,13.1107338,12.92473257,12.9493291,
        12.96094796,13.01023846,12.99892816,13.01475734,13.11740927,13.0449418,13.05688487,13.11040591,13.58346523,13.11971179,
        13.26041623,13.1169336,13.1514685,13.20695589,13.20202196,13.20592319,13.28126541,13.2466225
    ],
    "Log_Clientes": [
        5.356586275,5.370638028,5.351858133,5.433722004,5.365976015,5.398162702,5.384495063,5.38907173,5.379897354,5.38907173,
        5.393627546,5.402677382,5.420534999,5.420534999,5.416100402,5.484796933,5.365976015,5.342334252,5.351858133,5.38907173,
        5.416100402,5.407171771,5.416100402,5.429345629,5.438079309,5.472270674,5.493061443,5.59471138,5.446737372,5.451038454,
        5.455321115,5.468060141,5.484796933,5.493061443,5.549076085,5.556828062,5.587248658,5.59471138,5.609471795,5.66296048,
        5.65248918,5.648974238,5.683579767,5.690359454,5.713732806,5.707110265,5.752572639,5.755742214
    ]
}

df = pd.DataFrame(data)
df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y")
df.set_index("Data", inplace=True)
df = df.asfreq("MS")

print(f"\n📊 Dados históricos: {len(df)} observações")
print(f"   Período: {df.index[0].strftime('%m/%Y')} até {df.index[-1].strftime('%m/%Y')}")

# =============================
# TREINAR MODELO COM 48 OBS
# =============================
print("\n🔄 Treinando modelo SARIMAX(1,1,1)x(1,1,0,6) com 48 observações...")

y_full = df["Log_Total"]
X_full = df[["Log_Clientes"]]

model_full = SARIMAX(y_full, exog=X_full, order=(1,1,1), seasonal_order=(1,1,0,6),
                     enforce_stationarity=False, enforce_invertibility=False)
results_full = model_full.fit(disp=False)

print("✅ Modelo treinado com sucesso!")
print(f"   AIC: {results_full.aic:.2f}")
print(f"   BIC: {results_full.bic:.2f}")

# =============================
# GERAR PREVISÕES FUTURAS
# =============================
print("\n🔮 Gerando previsões para Out/2025 - Set/2026...")

# Criar datas futuras
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

# Projetar Log_Clientes com crescimento médio dos últimos 12 meses
recent_growth = df["Log_Clientes"].iloc[-12:].pct_change().mean()
print(f"   Crescimento médio de Log_Clientes (últimos 12 meses): {recent_growth*100:.2f}%")

last_log_clientes = df["Log_Clientes"].iloc[-1]
future_log_clientes = []
for i in range(12):
    next_value = last_log_clientes * (1 + recent_growth) ** (i + 1)
    future_log_clientes.append(next_value)

X_future = pd.DataFrame({'Log_Clientes': future_log_clientes}, index=future_dates)

# Fazer previsões
future_forecast_log = results_full.forecast(steps=12, exog=X_future)
future_forecast = np.exp(future_forecast_log)

print("✅ Previsões geradas!")

# =============================
# CRIAR DATAFRAME DE EXPORTAÇÃO
# =============================
previsoes_df = pd.DataFrame({
    'Ano': future_dates.year,
    'Mês': future_dates.month,
    'Data': future_dates.strftime('%d/%m/%Y'),
    'Mês_Nome': future_dates.strftime('%B/%Y'),
    'Log_Total_Previsto': future_forecast_log.values,
    'Total_Previsto_R$': future_forecast.values,
    'Log_Clientes_Projetado': future_log_clientes,
    'Clientes_Projetado': np.exp(future_log_clientes)
})

# Adicionar estatísticas
previsoes_df['Intervalo_Confiança_95%'] = '±' + (1.96 * results_full.resid.std()).round(4).astype(str)

# =============================
# EXPORTAR PARA CSV
# =============================
filename = 'previsoes_sarimax_48obs_out2025_set2026.csv'
previsoes_df.to_csv(filename, index=False, encoding='utf-8-sig')

print(f"\n📁 Arquivo exportado: {filename}")
print(f"   Localização: {filename}")

# =============================
# MOSTRAR PREVIEW
# =============================
print("\n" + "="*80)
print("PREVIEW DAS PREVISÕES")
print("="*80)
print(previsoes_df.to_string(index=False))

# =============================
# ESTATÍSTICAS RESUMIDAS
# =============================
print("\n" + "="*80)
print("ESTATÍSTICAS DAS PREVISÕES (Out/2025 - Set/2026)")
print("="*80)
print(f"\n📊 Total previsto (12 meses):    R$ {future_forecast.sum():>15,.2f}")
print(f"📊 Média mensal:                 R$ {future_forecast.mean():>15,.2f}")
print(f"📊 Mínimo:                       R$ {future_forecast.min():>15,.2f} ({future_dates[future_forecast.argmin()].strftime('%b/%Y')})")
print(f"📊 Máximo:                       R$ {future_forecast.max():>15,.2f} ({future_dates[future_forecast.argmax()].strftime('%b/%Y')})")
print(f"📊 Desvio padrão:                R$ {future_forecast.std():>15,.2f}")
print(f"📊 Coef. Variação:               {(future_forecast.std()/future_forecast.mean()*100):>15.2f}%")

print("\n" + "="*80)
print("✅ EXPORTAÇÃO CONCLUÍDA COM SUCESSO!")
print("="*80)
print(f"\n💡 Próximos passos:")
print(f"   1. Abra o arquivo '{filename}' no Excel ou Power BI")
print(f"   2. As previsões já estão em escala original (R$)")
print(f"   3. Log_Total e Log_Clientes também estão disponíveis para análise")
print(f"   4. Intervalo de confiança de 95% incluído")
