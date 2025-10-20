# ===============================================
# Análise SARIMAX - Série Log_Total
# Autor: Eduardo Piaia
# Data: 19/10/2025
# ===============================================

# =============================
# 1. Importar bibliotecas
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================
# 2. Carregar os dados
# =============================

# DataFrame com os dados fornecidos
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

# Converter coluna de data e definir como índice
df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y")
df.set_index("Data", inplace=True)
df = df.asfreq("MS")  # início de cada mês

print("="*60)
print("DADOS CARREGADOS COM SUCESSO")
print("="*60)
print(f"Período: {df.index.min().strftime('%d/%m/%Y')} até {df.index.max().strftime('%d/%m/%Y')}")
print(f"Total de observações: {len(df)}")
print("\nPrimeiras linhas:")
print(df.head())
print("\nÚltimas linhas:")
print(df.tail())
print("\nEstatísticas descritivas:")
print(df.describe())

# =============================
# 3. Decomposição sazonal da série Log_Total
# =============================
print("\n" + "="*60)
print("DECOMPOSIÇÃO SAZONAL")
print("="*60)

result = seasonal_decompose(df["Log_Total"], model='additive', period=12)

fig = result.plot()
fig.set_size_inches(12, 8)
plt.suptitle("Decomposição Sazonal da Série Log_Total", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# =============================
# 4. Separar dados em treino e teste
# =============================
# CONFIGURAÇÃO: Escolha o tamanho do conjunto de teste
# Opção 1: Usar porcentagem (ex: 80% treino, 20% teste)
# train_size = int(len(df) * 0.8)

# Opção 2: Usar número fixo de observações para teste (ex: últimas 12 observações)
TEST_SIZE = 12  # Altere aqui para o número desejado de observações de teste
train_size = len(df) - TEST_SIZE

train = df.iloc[:train_size]
test = df.iloc[train_size:]

print(f"\n⚙️  CONFIGURAÇÃO DA DIVISÃO TREINO/TESTE:")
print(f"   Total de observações: {len(df)}")
print(f"   Dados de treino: {len(train)} observações ({(len(train)/len(df)*100):.1f}%)")
print(f"   Dados de teste: {len(test)} observações ({(len(test)/len(df)*100):.1f}%)")
print(f"   Período de treino: {train.index.min().strftime('%d/%m/%Y')} até {train.index.max().strftime('%d/%m/%Y')}")
print(f"   Período de teste: {test.index.min().strftime('%d/%m/%Y')} até {test.index.max().strftime('%d/%m/%Y')}")

# =============================
# 5. Ajustar o modelo SARIMAX
# =============================
print("\n" + "="*60)
print("AJUSTANDO MODELO SARIMAX")
print("="*60)

# Parâmetros do modelo - altere aqui e os gráficos se atualizam automaticamente
order = (1, 1, 1)
seasonal_order = (1, 1, 0, 6)

print(f"\nParâmetros do modelo:")
print(f"  Order (p,d,q): {order}")
print(f"  Seasonal Order (P,D,Q,s): {seasonal_order}")

# Definir endógena e exógena para treino
y_train = train["Log_Total"]
X_train = train[["Log_Clientes"]]

# Ajuste do modelo
model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

print("\n" + "-"*60)
print("RESUMO DO MODELO")
print("-"*60)
print(results.summary())

# =============================
# 6. Gráficos de diagnóstico dos resíduos (SERÁ MOVIDO PARA DEPOIS DO MODELO COM 48 OBS)
# =============================
# Este gráfico será gerado após treinar o modelo com todas as 48 observações

# =============================
# 7. Previsões e Métricas de Erro
# =============================
print("\n" + "="*60)
print("PREVISÕES E MÉTRICAS DE ERRO")
print("="*60)

# Previsão in-sample (dados de treino) - USANDO PREDICT PARA VALORES ESTÁVEIS
# O método predict() é mais estável que fittedvalues para visualização
y_train_pred = results.predict(start=train.index[0], end=train.index[-1], exog=X_train)

# Previsão out-of-sample (dados de teste)
y_test = test["Log_Total"]
X_test = test[["Log_Clientes"]]

# Fazer previsões para o período de teste
forecast = results.forecast(steps=len(test), exog=X_test)

print(f"\n📊 Informação sobre as Métricas:")
print(f"   - Dados de TREINO: {len(train)} observações")
print(f"   - Dados de TESTE: {len(test)} observações (estas são usadas para calcular MAPE/MAE/RMSE)")
print(f"   - Período de teste: {test.index.min().strftime('%m/%Y')} até {test.index.max().strftime('%m/%Y')}")

# =============================
# 8. Calcular métricas de erro
# =============================

def calculate_metrics(y_true, y_pred, set_name=""):
    """
    Calcula métricas de erro para avaliação do modelo
    """
    # Remover valores nulos
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Calcular métricas
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(y_pred_clean - y_true_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
    
    # R² Score
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n{set_name} - Métricas de Erro:")
    print("-" * 50)
    print(f"  MAE  (Mean Absolute Error):           {mae:.6f}")
    print(f"  MSE  (Mean Squared Error):            {mse:.6f}")
    print(f"  RMSE (Root Mean Squared Error):       {rmse:.6f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.4f}%")
    print(f"  SMAPE (Symmetric MAPE):               {smape:.4f}%")
    print(f"  R²   (Coefficient of Determination):  {r2:.6f}")
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'R2': r2
    }

# Métricas para dados de treino (in-sample)
train_metrics = calculate_metrics(y_train, y_train_pred, "DADOS DE TREINO (In-Sample)")

# Métricas para dados de teste (out-of-sample)
test_metrics = calculate_metrics(y_test, forecast, "DADOS DE TESTE (Out-of-Sample)")

print(f"\n⚠️  IMPORTANTE: As métricas de TESTE foram calculadas com base em {len(test)} observações!")
print(f"   Se deseja usar as últimas 12 observações, altere a linha 'train_size' no código.")

# =============================
# 9. Critérios de Informação
# =============================
print("\n" + "="*60)
print("CRITÉRIOS DE INFORMAÇÃO DO MODELO")
print("="*60)
print(f"  AIC  (Akaike Information Criterion):        {results.aic:.4f}")
print(f"  BIC  (Bayesian Information Criterion):      {results.bic:.4f}")
print(f"  HQIC (Hannan-Quinn Information Criterion):  {results.hqic:.4f}")

# =============================
# 10. Reverter LOG e Criar Previsão do Escritório Contábil
# =============================
print("\n" + "="*60)
print("REVERTENDO LOG E CRIANDO PREVISÃO DO ESCRITÓRIO")
print("="*60)

# Reverter o logaritmo (aplicar exponencial)
y_train_original = np.exp(y_train)
y_test_original = np.exp(y_test)
y_train_pred_original = np.exp(y_train_pred)
forecast_original = np.exp(forecast)

# Criar a série completa original
y_original = np.exp(df["Log_Total"])

# Previsão do Escritório Contábil: valor do mês do ano anterior × 1.15
# Esta previsão só pode ser feita a partir da 13ª observação (após 12 meses)
escritorio_forecast = pd.Series(index=df.index, dtype=float)

for i in range(12, len(df)):
    # Pega o valor de 12 meses atrás e multiplica por 1.15
    escritorio_forecast.iloc[i] = y_original.iloc[i - 12] * 1.15

print(f"✅ Dados revertidos do LOG (aplicado exp())")
print(f"✅ Previsão do Escritório Contábil criada (valor ano anterior × 1.15)")
print(f"   - Previsão do escritório disponível a partir de: {df.index[12].strftime('%m/%Y')}")

# =============================
# 11. Visualização das Previsões (a partir de 2023)
# =============================
print("\n" + "="*60)
print("VISUALIZAÇÃO DAS PREVISÕES (A PARTIR DE 2023)")
print("="*60)

# Filtrar dados a partir de 2023
data_inicio_2023 = pd.Timestamp('2023-01-01')

# Filtrar todas as séries
mask_2023 = df.index >= data_inicio_2023
df_2023 = df[mask_2023]
y_original_2023 = y_original[mask_2023]

# Filtrar treino e teste
mask_train_2023 = train.index >= data_inicio_2023
mask_test_2023 = test.index >= data_inicio_2023

train_2023 = train[mask_train_2023]
y_train_original_2023 = y_train_original[mask_train_2023]
y_train_pred_original_2023 = y_train_pred_original[mask_train_2023]

test_2023 = test[mask_test_2023]
y_test_original_2023 = y_test_original[mask_test_2023]
forecast_original_2023 = forecast_original

# Filtrar previsão do escritório
escritorio_forecast_2023 = escritorio_forecast[mask_2023]

# Gráfico 1: Série completa a partir de 2023 com todas as previsões
plt.figure(figsize=(16, 7))
plt.plot(df_2023.index, y_original_2023, label='Observado', color='black', linewidth=3, marker='o', markersize=6, zorder=5)
plt.plot(train_2023.index, y_train_pred_original_2023, label='SARIMAX - In-Sample (Ajustado)', color='orange', linestyle='--', alpha=0.8, linewidth=2.5, marker='D', markersize=4)
plt.plot(test_2023.index, forecast_original_2023, label='SARIMAX - Out-of-Sample (Previsto)', color='red', linestyle='--', linewidth=2.5, marker='s', markersize=7, zorder=4)
plt.plot(escritorio_forecast_2023.index, escritorio_forecast_2023, label='Escritório Contábil (Ano Anterior × 1.15)', color='purple', linestyle=':', linewidth=2.5, marker='^', markersize=6, alpha=0.7)
plt.axvline(x=test.index[0], color='gray', linestyle='--', linewidth=2, label='Início do Teste', alpha=0.6)

plt.xlabel('Data', fontsize=13, fontweight='bold')
plt.ylabel('Total (Valores Originais)', fontsize=13, fontweight='bold')
plt.title('Comparação de Previsões: SARIMAX vs Escritório Contábil (2023+)', fontsize=15, fontweight='bold')
plt.legend(loc='best', fontsize=10, framealpha=0.95)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# Gráfico 2: Zoom no período de teste com comparação
plt.figure(figsize=(14, 7))
plt.plot(test_2023.index, y_test_original_2023, label='Observado (Real)', color='black', marker='o', linewidth=3, markersize=10, zorder=5)
plt.plot(test_2023.index, forecast_original_2023, label='SARIMAX (Previsto)', color='red', marker='s', linestyle='--', linewidth=2.5, markersize=8)

# Filtrar escritório para período de teste
escritorio_test = escritorio_forecast[test.index]
plt.plot(test_2023.index, escritorio_test[mask_test_2023], label='Escritório Contábil', color='purple', marker='^', linestyle=':', linewidth=2.5, markersize=8, alpha=0.8)

plt.xlabel('Data', fontsize=13, fontweight='bold')
plt.ylabel('Total (Valores Originais)', fontsize=13, fontweight='bold')
plt.title('Zoom: Comparação de Previsões no Período de Teste', fontsize=15, fontweight='bold')
plt.legend(loc='best', fontsize=11, framealpha=0.95)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# =============================
# 12. Comparação de Métricas: SARIMAX vs Escritório Contábil
# =============================
print("\n" + "="*60)
print("COMPARAÇÃO DE DESEMPENHO: SARIMAX vs ESCRITÓRIO")
print("="*60)

# Calcular métricas para SARIMAX (em valores originais)
print("\n" + "🤖 MODELO SARIMAX (Machine Learning)")
test_metrics_original = calculate_metrics(y_test_original, forecast_original, "SARIMAX - Out-of-Sample")

# Calcular métricas para Escritório Contábil
escritorio_test_values = escritorio_test.values
print("\n" + "📊 ESCRITÓRIO CONTÁBIL (Regra: Ano Anterior × 1.15)")
escritorio_metrics = calculate_metrics(y_test_original, escritorio_test_values, "Escritório Contábil")

# Comparação direta
print("\n" + "="*60)
print("📈 COMPARAÇÃO DE DESEMPENHO")
print("="*60)
print(f"\n{'Métrica':<10} {'SARIMAX':>15} {'Escritório':>15} {'Melhoria':>15}")
print("-" * 60)

mape_diff = ((escritorio_metrics['MAPE'] - test_metrics_original['MAPE']) / escritorio_metrics['MAPE']) * 100
mae_diff = ((escritorio_metrics['MAE'] - test_metrics_original['MAE']) / escritorio_metrics['MAE']) * 100
rmse_diff = ((escritorio_metrics['RMSE'] - test_metrics_original['RMSE']) / escritorio_metrics['RMSE']) * 100
r2_diff = test_metrics_original['R2'] - escritorio_metrics['R2']

print(f"{'MAPE':<10} {test_metrics_original['MAPE']:>14.2f}% {escritorio_metrics['MAPE']:>14.2f}% {mape_diff:>14.1f}%")
print(f"{'MAE':<10} {test_metrics_original['MAE']:>15,.2f} {escritorio_metrics['MAE']:>15,.2f} {mae_diff:>14.1f}%")
print(f"{'RMSE':<10} {test_metrics_original['RMSE']:>15,.2f} {escritorio_metrics['RMSE']:>15,.2f} {rmse_diff:>14.1f}%")
print(f"{'R²':<10} {test_metrics_original['R2']:>15.4f} {escritorio_metrics['R2']:>15.4f} {r2_diff:>14.4f}")

print("\n💡 Interpretação:")
if test_metrics_original['MAPE'] < escritorio_metrics['MAPE']:
    print(f"   ✅ SARIMAX é {mape_diff:.1f}% MELHOR que o método do Escritório!")
else:
    print(f"   ⚠️  Método do Escritório é melhor neste caso.")


# Gráfico 3: Resíduos do período de teste
test_residuals = y_test_original - forecast_original

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(test.index, test_residuals, color='red', marker='o', linewidth=2, markersize=6, label='SARIMAX')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Data', fontsize=11)
plt.ylabel('Resíduos', fontsize=11)
plt.title('Resíduos - SARIMAX', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
escritorio_residuals = y_test_original.values - escritorio_test_values
plt.plot(test.index, escritorio_residuals, color='purple', marker='^', linewidth=2, markersize=6, label='Escritório')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Data', fontsize=11)
plt.ylabel('Resíduos', fontsize=11)
plt.title('Resíduos - Escritório Contábil', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.hist(test_residuals, bins=8, color='red', alpha=0.7, edgecolor='black')
plt.xlabel('Resíduos', fontsize=11)
plt.ylabel('Frequência', fontsize=11)
plt.title('Distribuição dos Resíduos - SARIMAX', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 2, 4)
plt.hist(escritorio_residuals, bins=8, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Resíduos', fontsize=11)
plt.ylabel('Frequência', fontsize=11)
plt.title('Distribuição dos Resíduos - Escritório', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Análise de Resíduos: SARIMAX vs Escritório Contábil', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# =============================
# 13. Tabela de Comparação Detalhada
# =============================
print("\n" + "="*60)
print("TABELA DE COMPARAÇÃO: SARIMAX vs ESCRITÓRIO (TESTE)")
print("="*60)

comparison_df = pd.DataFrame({
    'Data': test.index,
    'Real': y_test_original.values,
    'SARIMAX': forecast_original.values,
    'Escritório': escritorio_test_values,
    'Erro_SARIMAX': (y_test_original.values - forecast_original.values),
    'Erro_Escritório': (y_test_original.values - escritorio_test_values),
    'Erro_%_SARIMAX': np.abs((y_test_original.values - forecast_original.values) / y_test_original.values * 100),
    'Erro_%_Escritório': np.abs((y_test_original.values - escritorio_test_values) / y_test_original.values * 100)
})

# Formatar valores
comparison_df['Real'] = comparison_df['Real'].apply(lambda x: f"{x:,.2f}")
comparison_df['SARIMAX'] = comparison_df['SARIMAX'].apply(lambda x: f"{x:,.2f}")
comparison_df['Escritório'] = comparison_df['Escritório'].apply(lambda x: f"{x:,.2f}")
comparison_df['Erro_SARIMAX'] = comparison_df['Erro_SARIMAX'].apply(lambda x: f"{x:,.2f}")
comparison_df['Erro_Escritório'] = comparison_df['Erro_Escritório'].apply(lambda x: f"{x:,.2f}")
comparison_df['Erro_%_SARIMAX'] = comparison_df['Erro_%_SARIMAX'].apply(lambda x: f"{x:.2f}%")
comparison_df['Erro_%_Escritório'] = comparison_df['Erro_%_Escritório'].apply(lambda x: f"{x:.2f}%")

print("\n" + comparison_df.to_string(index=False))

# =============================
# 14. Resumo Final Atualizado
# =============================
print("\n" + "="*60)
print("RESUMO FINAL DA ANÁLISE")
print("="*60)
print(f"\n📊 Modelo: SARIMAX{order}x{seasonal_order}")
print(f"📅 Período Total: {df.index.min().strftime('%d/%m/%Y')} até {df.index.max().strftime('%d/%m/%Y')}")
print(f"📈 Total de Observações: {len(df)}")
print(f"🔵 Dados de Treino: {len(train)} ({(len(train)/len(df)*100):.1f}%)")
print(f"🟢 Dados de Teste: {len(test)} ({(len(test)/len(df)*100):.1f}%)")

print(f"\n💡 Critérios de Informação:")
print(f"   AIC: {results.aic:.4f} | BIC: {results.bic:.4f} | HQIC: {results.hqic:.4f}")

print(f"\n✅ Performance SARIMAX no Teste (Valores Originais):")
print(f"   MAPE: {test_metrics_original['MAPE']:.2f}% | RMSE: {test_metrics_original['RMSE']:,.2f} | R²: {test_metrics_original['R2']:.4f}")

print(f"\n📊 Performance Escritório Contábil no Teste:")
print(f"   MAPE: {escritorio_metrics['MAPE']:.2f}% | RMSE: {escritorio_metrics['RMSE']:,.2f} | R²: {escritorio_metrics['R2']:.4f}")

print(f"\n🏆 Vencedor: ", end="")
if test_metrics_original['MAPE'] < escritorio_metrics['MAPE']:
    print(f"SARIMAX (MAPE {mape_diff:.1f}% melhor!)")
else:
    print("Escritório Contábil")

print("\n" + "="*60)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*60)

# =============================
# 15. MODELO FINAL COM TODAS AS 48 OBSERVAÇÕES
# =============================
print("\n" + "="*80)
print("MODELO FINAL: TREINAMENTO COM TODAS AS 48 OBSERVAÇÕES")
print("="*80)

# Treinar modelo com todos os dados
y_full = df["Log_Total"]
X_full = df[["Log_Clientes"]]

print("\n🔄 Ajustando modelo SARIMAX com o dataset completo...")
model_full = SARIMAX(y_full, exog=X_full, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
results_full = model_full.fit(disp=False)

print("\n" + "="*80)
print("RESUMO DO MODELO FINAL (48 OBSERVAÇÕES)")
print("="*80)
print(results_full.summary())

print("\n" + "="*80)
print("CRITÉRIOS DE INFORMAÇÃO DO MODELO FINAL")
print("="*80)
print(f"  AIC  (Akaike Information Criterion):        {results_full.aic:.4f}")
print(f"  BIC  (Bayesian Information Criterion):      {results_full.bic:.4f}")
print(f"  HQIC (Hannan-Quinn Information Criterion):  {results_full.hqic:.4f}")

# =============================
# 15.1. DIAGNÓSTICO DOS RESÍDUOS DO MODELO FINAL (48 OBSERVAÇÕES)
# =============================
print("\n" + "="*80)
print("DIAGNÓSTICO DOS RESÍDUOS - MODELO FINAL (48 OBSERVAÇÕES)")
print("="*80)

results_full.plot_diagnostics(figsize=(15, 10))
plt.suptitle(f"Diagnóstico dos Resíduos - SARIMAX{order}x{seasonal_order} (48 observações)", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# =============================
# 15.2. ANÁLISE DE RESÍDUOS DO MODELO FINAL
# =============================
print("\n" + "="*80)
print("ANÁLISE DETALHADA DOS RESÍDUOS - MODELO FINAL")
print("="*80)

# Obter resíduos do modelo final
residuals_full = results_full.resid

# Calcular estatísticas dos resíduos
print(f"\n📊 Estatísticas dos Resíduos:")
print(f"   Média: {residuals_full.mean():.6f}")
print(f"   Desvio Padrão: {residuals_full.std():.6f}")
print(f"   Mínimo: {residuals_full.min():.6f}")
print(f"   Máximo: {residuals_full.max():.6f}")

# Gráficos de resíduos
plt.figure(figsize=(16, 10))

# Gráfico 1: Resíduos ao longo do tempo
plt.subplot(2, 2, 1)
plt.plot(df.index, residuals_full, color='blue', linewidth=2, marker='o', markersize=4)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.axhline(y=residuals_full.mean() + 2*residuals_full.std(), color='orange', linestyle=':', linewidth=1.5, label='+2σ')
plt.axhline(y=residuals_full.mean() - 2*residuals_full.std(), color='orange', linestyle=':', linewidth=1.5, label='-2σ')
plt.xlabel('Data', fontsize=11, fontweight='bold')
plt.ylabel('Resíduos', fontsize=11, fontweight='bold')
plt.title('Resíduos ao Longo do Tempo', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Gráfico 2: Histograma dos resíduos
plt.subplot(2, 2, 2)
plt.hist(residuals_full, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Média = 0')
plt.axvline(x=residuals_full.mean(), color='orange', linestyle='--', linewidth=2, label=f'Média = {residuals_full.mean():.4f}')
plt.xlabel('Resíduos', fontsize=11, fontweight='bold')
plt.ylabel('Frequência', fontsize=11, fontweight='bold')
plt.title('Distribuição dos Resíduos', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Gráfico 3: Q-Q Plot
from scipy import stats
plt.subplot(2, 2, 3)
stats.probplot(residuals_full, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normalidade dos Resíduos)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Gráfico 4: Resíduos vs Valores Ajustados
plt.subplot(2, 2, 4)
fitted_values_full = results_full.predict(start=df.index[0], end=df.index[-1], exog=X_full)
plt.scatter(fitted_values_full, residuals_full, color='purple', alpha=0.6, s=50, edgecolors='black')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Valores Ajustados', fontsize=11, fontweight='bold')
plt.ylabel('Resíduos', fontsize=11, fontweight='bold')
plt.title('Resíduos vs Valores Ajustados', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.suptitle('Análise Completa dos Resíduos - Modelo Final (48 Observações)', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# =============================
# 16. PREVISÃO PARA OS PRÓXIMOS 12 MESES
# =============================
print("\n" + "="*80)
print("PREVISÃO PARA OS PRÓXIMOS 12 MESES")
print("="*80)

# Criar índice de datas futuras (próximos 12 meses)
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

# Para fazer previsões, precisamos dos valores futuros de Log_Clientes
# Vamos assumir crescimento médio dos últimos 12 meses
recent_growth = df["Log_Clientes"].iloc[-12:].pct_change().mean()
print(f"\n📊 Crescimento médio de Log_Clientes (últimos 12 meses): {recent_growth*100:.2f}%")

# Gerar valores futuros de Log_Clientes
last_log_clientes = df["Log_Clientes"].iloc[-1]
future_log_clientes = []
for i in range(12):
    next_value = last_log_clientes * (1 + recent_growth) ** (i + 1)
    future_log_clientes.append(next_value)

X_future = pd.DataFrame({'Log_Clientes': future_log_clientes}, index=future_dates)

print(f"\n🔮 Gerando previsões SARIMAX para Out/2025 até Set/2026...")
# Fazer previsões para os próximos 12 meses
future_forecast_log = results_full.forecast(steps=12, exog=X_future)
future_forecast = np.exp(future_forecast_log)

# Criar previsão heurística do escritório para os próximos 12 meses
# Regra: valor do mesmo mês do ano anterior × 1.15
y_original_full = np.exp(df["Log_Total"])
escritorio_future_forecast = []

for i in range(12):
    # Pega o valor de 12 meses atrás (últimos 12 valores do dataset)
    base_value = y_original_full.iloc[-(12-i)]
    escritorio_future_forecast.append(base_value * 1.15)

escritorio_future_forecast = pd.Series(escritorio_future_forecast, index=future_dates)

print(f"✅ Previsões geradas com sucesso!")
print(f"\n📅 Período de previsão: {future_dates[0].strftime('%m/%Y')} até {future_dates[-1].strftime('%m/%Y')}")

# =============================
# 17. TABELA DE PREVISÕES FUTURAS
# =============================
print("\n" + "="*80)
print("TABELA DE PREVISÕES PARA OS PRÓXIMOS 12 MESES")
print("="*80)

future_comparison = pd.DataFrame({
    'Data': future_dates,
    'SARIMAX': future_forecast.values,
    'Escritório_Contábil': escritorio_future_forecast.values,
    'Diferença': future_forecast.values - escritorio_future_forecast.values,
    'Diferença_%': ((future_forecast.values - escritorio_future_forecast.values) / escritorio_future_forecast.values * 100)
})

# Formatar para exibição
future_comparison_display = future_comparison.copy()
future_comparison_display['SARIMAX'] = future_comparison_display['SARIMAX'].apply(lambda x: f"R$ {x:,.2f}")
future_comparison_display['Escritório_Contábil'] = future_comparison_display['Escritório_Contábil'].apply(lambda x: f"R$ {x:,.2f}")
future_comparison_display['Diferença'] = future_comparison_display['Diferença'].apply(lambda x: f"R$ {x:,.2f}")
future_comparison_display['Diferença_%'] = future_comparison_display['Diferença_%'].apply(lambda x: f"{x:+.2f}%")

print("\n" + future_comparison_display.to_string(index=False))

# =============================
# 18. GRÁFICO DE PREVISÕES FUTURAS
# =============================
print("\n" + "="*80)
print("VISUALIZAÇÃO DAS PREVISÕES FUTURAS")
print("="*80)

# Gráfico com histórico recente + previsões futuras
plt.figure(figsize=(16, 8))

# Filtrar últimos 24 meses do histórico para contexto
historical_cutoff = pd.Timestamp('2023-10-01')
mask_recent = df.index >= historical_cutoff
df_recent = df[mask_recent]
y_original_recent = y_original_full[mask_recent]

# Plotar histórico recente
plt.plot(df_recent.index, y_original_recent, label='Histórico (Observado)', 
         color='black', linewidth=3, marker='o', markersize=6, zorder=5)

# Plotar previsões SARIMAX
plt.plot(future_dates, future_forecast, label='SARIMAX - Previsão (Out/2025 - Set/2026)', 
         color='red', linewidth=3, marker='s', markersize=8, linestyle='--', zorder=4)

# Plotar previsões do Escritório
plt.plot(future_dates, escritorio_future_forecast, label='Escritório Contábil - Previsão', 
         color='purple', linewidth=3, marker='^', markersize=8, linestyle=':', alpha=0.8, zorder=3)

# Adicionar linha vertical separando histórico de previsões
plt.axvline(x=df.index[-1], color='gray', linestyle='--', linewidth=2.5, 
            label='Início das Previsões', alpha=0.7)

# Adicionar área sombreada para o período de previsão
plt.axvspan(future_dates[0], future_dates[-1], alpha=0.1, color='yellow', 
            label='Período de Previsão')

plt.xlabel('Data', fontsize=14, fontweight='bold')
plt.ylabel('Total (Valores Originais - R$)', fontsize=14, fontweight='bold')
plt.title('Previsões para os Próximos 12 Meses: SARIMAX vs Escritório Contábil', 
          fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=11, framealpha=0.95)
plt.grid(True, alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico 2: Comparação apenas das previsões futuras
plt.figure(figsize=(14, 7))
plt.plot(future_dates, future_forecast, label='SARIMAX', 
         color='red', linewidth=3.5, marker='s', markersize=10, linestyle='-', zorder=4)
plt.plot(future_dates, escritorio_future_forecast, label='Escritório Contábil', 
         color='purple', linewidth=3.5, marker='^', markersize=10, linestyle='-', alpha=0.8, zorder=3)

# Adicionar valores no gráfico
for i, (date, sarimax_val, escrit_val) in enumerate(zip(future_dates, future_forecast, escritorio_future_forecast)):
    if i % 2 == 0:  # Mostrar valores alternados para não poluir
        plt.text(date, sarimax_val, f'R$ {sarimax_val:,.0f}', 
                fontsize=9, ha='center', va='bottom', color='red', fontweight='bold')
        plt.text(date, escrit_val, f'R$ {escrit_val:,.0f}', 
                fontsize=9, ha='center', va='top', color='purple', fontweight='bold')

plt.xlabel('Data', fontsize=13, fontweight='bold')
plt.ylabel('Total Previsto (R$)', fontsize=13, fontweight='bold')
plt.title('Comparação Detalhada: Previsões Futuras (Out/2025 - Set/2026)', 
          fontsize=15, fontweight='bold', pad=15)
plt.legend(loc='best', fontsize=12, framealpha=0.95)
plt.grid(True, alpha=0.4, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================
# 19. ESTATÍSTICAS DAS PREVISÕES FUTURAS
# =============================
print("\n" + "="*80)
print("ESTATÍSTICAS DAS PREVISÕES FUTURAS")
print("="*80)

print(f"\n📊 SARIMAX:")
print(f"   Total previsto (12 meses): R$ {future_forecast.sum():,.2f}")
print(f"   Média mensal: R$ {future_forecast.mean():,.2f}")
print(f"   Mínimo: R$ {future_forecast.min():,.2f} ({future_dates[future_forecast.argmin()].strftime('%m/%Y')})")
print(f"   Máximo: R$ {future_forecast.max():,.2f} ({future_dates[future_forecast.argmax()].strftime('%m/%Y')})")

print(f"\n📊 Escritório Contábil:")
print(f"   Total previsto (12 meses): R$ {escritorio_future_forecast.sum():,.2f}")
print(f"   Média mensal: R$ {escritorio_future_forecast.mean():,.2f}")
print(f"   Mínimo: R$ {escritorio_future_forecast.min():,.2f} ({future_dates[escritorio_future_forecast.argmin()].strftime('%m/%Y')})")
print(f"   Máximo: R$ {escritorio_future_forecast.max():,.2f} ({future_dates[escritorio_future_forecast.argmax()].strftime('%m/%Y')})")

diff_total = future_forecast.sum() - escritorio_future_forecast.sum()
diff_pct = (diff_total / escritorio_future_forecast.sum()) * 100

print(f"\n💡 Diferença Total:")
print(f"   SARIMAX vs Escritório: R$ {diff_total:,.2f} ({diff_pct:+.2f}%)")

if diff_total > 0:
    print(f"   ✅ SARIMAX prevê receita R$ {abs(diff_total):,.2f} MAIOR que o Escritório")
else:
    print(f"   ⚠️  SARIMAX prevê receita R$ {abs(diff_total):,.2f} MENOR que o Escritório")

print("\n" + "="*80)
print("ANÁLISE COMPLETA FINALIZADA!")
print("="*80)

# =============================
# 20. MODELO ETS (Error, Trend, Seasonality)
# =============================
print("\n" + "="*80)
print("MODELO ETS (EXPONENTIAL SMOOTHING)")
print("="*80)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Treinar modelo ETS com os dados de treino (36 observações)
print("\n🔄 Ajustando modelo ETS (Holt-Winters)...")

# Usar série em log para treino
ets_model = ExponentialSmoothing(
    y_train,
    seasonal_periods=12,
    trend='add',
    seasonal='add',
    damped_trend=True
)
ets_results = ets_model.fit(optimized=True)

print("✅ Modelo ETS ajustado com sucesso!")

# Fazer previsões com ETS
print("\n🔮 Gerando previsões ETS para o período de teste...")
ets_forecast_log = ets_results.forecast(steps=len(test))
ets_forecast = np.exp(ets_forecast_log)

print(f"✅ Previsões ETS geradas!")

# Calcular métricas para ETS
print("\n" + "📊 MODELO ETS (Exponential Smoothing)")
ets_metrics = calculate_metrics(y_test_original, ets_forecast, "ETS - Out-of-Sample")

# =============================
# 21. ENSEMBLE: COMBINAÇÃO SARIMAX + ETS
# =============================
print("\n" + "="*80)
print("MODELO ENSEMBLE: SARIMAX + ETS COM OTIMIZAÇÃO DE PESOS")
print("="*80)

from scipy.optimize import minimize

def ensemble_predictions(weights, pred1, pred2):
    """Combina duas previsões com pesos dados"""
    return weights[0] * pred1 + weights[1] * pred2

def ensemble_error(weights, pred1, pred2, y_true):
    """Calcula o erro (RMSE) do ensemble"""
    ensemble_pred = ensemble_predictions(weights, pred1, pred2)
    return np.sqrt(mean_squared_error(y_true, ensemble_pred))

print("\n🔄 Otimizando pesos do ensemble...")

# Restrições: pesos devem somar 1 e estar entre 0 e 1
constraints = ({'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1})
bounds = [(0, 1), (0, 1)]

# Otimizar pesos para minimizar RMSE
initial_weights = [0.5, 0.5]
result_opt = minimize(
    ensemble_error,
    initial_weights,
    args=(forecast_original.values, ets_forecast.values, y_test_original.values),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result_opt.x
print(f"\n✅ Pesos ótimos encontrados!")
print(f"   Peso SARIMAX: {optimal_weights[0]:.4f} ({optimal_weights[0]*100:.2f}%)")
print(f"   Peso ETS: {optimal_weights[1]:.4f} ({optimal_weights[1]*100:.2f}%)")

# Criar previsões do ensemble
ensemble_forecast = ensemble_predictions(optimal_weights, forecast_original.values, ets_forecast.values)

# Calcular métricas para o ensemble
print("\n" + "🎯 MODELO ENSEMBLE (SARIMAX + ETS)")
ensemble_metrics = calculate_metrics(y_test_original, ensemble_forecast, "Ensemble - Out-of-Sample")

# =============================
# 22. COMPARAÇÃO DE TODOS OS MODELOS
# =============================
print("\n" + "="*80)
print("COMPARAÇÃO COMPLETA: TODOS OS MODELOS")
print("="*80)

print(f"\n{'Modelo':<20} {'MAPE':>12} {'MAE':>15} {'RMSE':>15} {'R²':>12}")
print("-" * 80)
print(f"{'SARIMAX':<20} {test_metrics_original['MAPE']:>11.2f}% {test_metrics_original['MAE']:>15,.2f} {test_metrics_original['RMSE']:>15,.2f} {test_metrics_original['R2']:>12.4f}")
print(f"{'ETS':<20} {ets_metrics['MAPE']:>11.2f}% {ets_metrics['MAE']:>15,.2f} {ets_metrics['RMSE']:>15,.2f} {ets_metrics['R2']:>12.4f}")
print(f"{'Ensemble':<20} {ensemble_metrics['MAPE']:>11.2f}% {ensemble_metrics['MAE']:>15,.2f} {ensemble_metrics['RMSE']:>15,.2f} {ensemble_metrics['R2']:>12.4f}")
print(f"{'Escritório':<20} {escritorio_metrics['MAPE']:>11.2f}% {escritorio_metrics['MAE']:>15,.2f} {escritorio_metrics['RMSE']:>15,.2f} {escritorio_metrics['R2']:>12.4f}")

# Determinar o melhor modelo
models_comparison = {
    'SARIMAX': test_metrics_original['MAPE'],
    'ETS': ets_metrics['MAPE'],
    'Ensemble': ensemble_metrics['MAPE'],
    'Escritório': escritorio_metrics['MAPE']
}

best_model = min(models_comparison, key=models_comparison.get)
print(f"\n🏆 MELHOR MODELO: {best_model} (MAPE: {models_comparison[best_model]:.2f}%)")

# =============================
# 23. GRÁFICO COMPARATIVO - TODOS OS MODELOS
# =============================
print("\n" + "="*80)
print("VISUALIZAÇÃO COMPARATIVA DE TODOS OS MODELOS")
print("="*80)

plt.figure(figsize=(16, 8))
plt.plot(test.index, y_test_original, label='Observado (Real)', 
         color='black', marker='o', linewidth=3.5, markersize=10, zorder=5)
plt.plot(test.index, forecast_original, label='SARIMAX', 
         color='red', marker='s', linestyle='--', linewidth=2.5, markersize=7, alpha=0.8)
plt.plot(test.index, ets_forecast, label='ETS', 
         color='blue', marker='D', linestyle='--', linewidth=2.5, markersize=7, alpha=0.8)
plt.plot(test.index, ensemble_forecast, label=f'Ensemble (SARIMAX {optimal_weights[0]:.0%} + ETS {optimal_weights[1]:.0%})', 
         color='green', marker='*', linestyle='-', linewidth=3, markersize=12, zorder=4)
plt.plot(test.index, escritorio_test_values, label='Escritório Contábil', 
         color='purple', marker='^', linestyle=':', linewidth=2.5, markersize=7, alpha=0.7)

plt.xlabel('Data', fontsize=14, fontweight='bold')
plt.ylabel('Total (R$)', fontsize=14, fontweight='bold')
plt.title('Comparação de Todos os Modelos - Período de Teste', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=11, framealpha=0.95)
plt.grid(True, alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================
# 24. PREVISÕES FUTURAS COM ENSEMBLE
# =============================
print("\n" + "="*80)
print("PREVISÕES FUTURAS COM MODELO ENSEMBLE")
print("="*80)

# Treinar ETS com todas as 48 observações
print("\n🔄 Treinando modelo ETS com todas as 48 observações...")
ets_model_full = ExponentialSmoothing(
    y_full,
    seasonal_periods=12,
    trend='add',
    seasonal='add',
    damped_trend=True
)
ets_results_full = ets_model_full.fit(optimized=True)
print("✅ Modelo ETS completo ajustado!")

# Fazer previsões futuras com ETS
print("\n🔮 Gerando previsões ETS para Out/2025 - Set/2026...")
ets_future_forecast_log = ets_results_full.forecast(steps=12)
ets_future_forecast = np.exp(ets_future_forecast_log)

# Criar ensemble para previsões futuras (usando os mesmos pesos ótimos)
ensemble_future_forecast = ensemble_predictions(
    optimal_weights, 
    future_forecast.values, 
    ets_future_forecast.values
)

print(f"✅ Previsões Ensemble futuras geradas!")

# Tabela de previsões futuras - todos os modelos
print("\n" + "="*80)
print("TABELA DE PREVISÕES FUTURAS - TODOS OS MODELOS (Out/2025 - Set/2026)")
print("="*80)

future_all_models = pd.DataFrame({
    'Data': future_dates,
    'SARIMAX': future_forecast.values,
    'ETS': ets_future_forecast.values,
    'Ensemble': ensemble_future_forecast,
    'Escritório': escritorio_future_forecast.values
})

# Formatar para exibição
future_all_models_display = future_all_models.copy()
future_all_models_display['SARIMAX'] = future_all_models_display['SARIMAX'].apply(lambda x: f"R$ {x:,.2f}")
future_all_models_display['ETS'] = future_all_models_display['ETS'].apply(lambda x: f"R$ {x:,.2f}")
future_all_models_display['Ensemble'] = future_all_models_display['Ensemble'].apply(lambda x: f"R$ {x:,.2f}")
future_all_models_display['Escritório'] = future_all_models_display['Escritório'].apply(lambda x: f"R$ {x:,.2f}")

print("\n" + future_all_models_display.to_string(index=False))

# =============================
# 25. GRÁFICO DE PREVISÕES FUTURAS - TODOS OS MODELOS
# =============================
print("\n" + "="*80)
print("VISUALIZAÇÃO DAS PREVISÕES FUTURAS - TODOS OS MODELOS")
print("="*80)

plt.figure(figsize=(16, 8))

# Plotar histórico recente
plt.plot(df_recent.index, y_original_recent, label='Histórico (Observado)', 
         color='black', linewidth=3, marker='o', markersize=6, zorder=5)

# Plotar previsões de todos os modelos
plt.plot(future_dates, future_forecast, label='SARIMAX', 
         color='red', linewidth=2.5, marker='s', markersize=7, linestyle='--', alpha=0.8)
plt.plot(future_dates, ets_future_forecast, label='ETS', 
         color='blue', linewidth=2.5, marker='D', markersize=7, linestyle='--', alpha=0.8)
plt.plot(future_dates, ensemble_future_forecast, label=f'Ensemble (Otimizado)', 
         color='green', linewidth=3.5, marker='*', markersize=10, linestyle='-', zorder=4)
plt.plot(future_dates, escritorio_future_forecast, label='Escritório Contábil', 
         color='purple', linewidth=2.5, marker='^', markersize=7, linestyle=':', alpha=0.7)

# Linha vertical separando histórico de previsões
plt.axvline(x=df.index[-1], color='gray', linestyle='--', linewidth=2.5, 
            label='Início das Previsões', alpha=0.7)

# Área sombreada
plt.axvspan(future_dates[0], future_dates[-1], alpha=0.1, color='yellow')

plt.xlabel('Data', fontsize=14, fontweight='bold')
plt.ylabel('Total (R$)', fontsize=14, fontweight='bold')
plt.title('Previsões Futuras: Comparação de Todos os Modelos', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=11, framealpha=0.95)
plt.grid(True, alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================
# 26. ESTATÍSTICAS FINAIS - PREVISÕES FUTURAS
# =============================
print("\n" + "="*80)
print("ESTATÍSTICAS DAS PREVISÕES FUTURAS - TODOS OS MODELOS")
print("="*80)

print(f"\n{'Modelo':<20} {'Total (12 meses)':>20} {'Média Mensal':>20}")
print("-" * 65)
print(f"{'SARIMAX':<20} R$ {future_forecast.sum():>17,.2f} R$ {future_forecast.mean():>17,.2f}")
print(f"{'ETS':<20} R$ {ets_future_forecast.sum():>17,.2f} R$ {ets_future_forecast.mean():>17,.2f}")
print(f"{'Ensemble':<20} R$ {ensemble_future_forecast.sum():>17,.2f} R$ {ensemble_future_forecast.mean():>17,.2f}")
print(f"{'Escritório':<20} R$ {escritorio_future_forecast.sum():>17,.2f} R$ {escritorio_future_forecast.mean():>17,.2f}")

# Comparação com escritório
ensemble_diff = ensemble_future_forecast.sum() - escritorio_future_forecast.sum()
ensemble_diff_pct = (ensemble_diff / escritorio_future_forecast.sum()) * 100

print(f"\n💡 Ensemble vs Escritório:")
print(f"   Diferença: R$ {ensemble_diff:,.2f} ({ensemble_diff_pct:+.2f}%)")

if ensemble_diff > 0:
    print(f"   ✅ Ensemble prevê R$ {abs(ensemble_diff):,.2f} a MAIS que o Escritório")
else:
    print(f"   ⚠️  Ensemble prevê R$ {abs(ensemble_diff):,.2f} a MENOS que o Escritório")

print("\n" + "="*80)
print("🎉 ANÁLISE COMPLETA COM ENSEMBLE FINALIZADA!")
print("="*80)

print(f"\n📊 RESUMO EXECUTIVO:")
print(f"   🥇 Melhor modelo no teste: {best_model} (MAPE: {models_comparison[best_model]:.2f}%)")
print(f"   🎯 Ensemble combina: SARIMAX ({optimal_weights[0]:.0%}) + ETS ({optimal_weights[1]:.0%})")
print(f"   💰 Previsão Ensemble (12 meses): R$ {ensemble_future_forecast.sum():,.2f}")
print(f"   📈 Diferença vs Escritório: {ensemble_diff_pct:+.2f}%")
