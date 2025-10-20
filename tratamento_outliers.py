# ===============================================
# ESTRATÉGIAS PARA TRATAR RESÍDUOS INICIAIS
# Autor: Eduardo Piaia
# Data: 19/10/2025
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

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

df_original = pd.DataFrame(data)
df_original["Data"] = pd.to_datetime(df_original["Data"], format="%d/%m/%Y")
df_original.set_index("Data", inplace=True)
df_original = df_original.asfreq("MS")

print("="*80)
print("ESTRATÉGIAS PARA TRATAR OUTLIERS E MELHORAR RESÍDUOS INICIAIS")
print("="*80)
print(f"\n📊 Dados: {len(df_original)} observações de {df_original.index[0].strftime('%m/%Y')} a {df_original.index[-1].strftime('%m/%Y')}")

# =============================
# FUNÇÃO DE MÉTRICAS
# =============================
def calculate_metrics_simple(y_true, y_pred):
    """Calcula métricas de erro"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# =============================
# ESTRATÉGIA 1: MODELO ORIGINAL (BASELINE)
# =============================
print("\n" + "="*80)
print("ESTRATÉGIA 1: MODELO ORIGINAL (BASELINE)")
print("="*80)

df1 = df_original.copy()
y_full1 = df1["Log_Total"]
X_full1 = df1[["Log_Clientes"]]

model1 = SARIMAX(y_full1, exog=X_full1, order=(1,1,1), seasonal_order=(1,1,0,6),
                 enforce_stationarity=False, enforce_invertibility=False)
results1 = model1.fit(disp=False)
residuals1 = results1.resid

print(f"✅ Modelo ajustado")
print(f"   Resíduo máximo: {residuals1.max():.2f}")
print(f"   Resíduo mínimo: {residuals1.min():.2f}")
print(f"   Desvio padrão: {residuals1.std():.2f}")
print(f"   AIC: {results1.aic:.2f} | BIC: {results1.bic:.2f}")

# =============================
# ESTRATÉGIA 2: REMOVER PRIMEIRAS OBSERVAÇÕES
# =============================
print("\n" + "="*80)
print("ESTRATÉGIA 2: REMOVER PRIMEIROS 6 MESES (Out/2021 - Mar/2022)")
print("="*80)
print("💡 Remove o período inicial problemático, treinando apenas com dados 'estáveis'")

# Remover as primeiras 6 observações
df2 = df_original.iloc[6:].copy()
y_full2 = df2["Log_Total"]
X_full2 = df2[["Log_Clientes"]]

model2 = SARIMAX(y_full2, exog=X_full2, order=(1,1,1), seasonal_order=(1,1,0,6),
                 enforce_stationarity=False, enforce_invertibility=False)
results2 = model2.fit(disp=False)
residuals2 = results2.resid

print(f"✅ Modelo ajustado com {len(df2)} observações (removeu {len(df_original) - len(df2)} iniciais)")
print(f"   Período: {df2.index[0].strftime('%m/%Y')} até {df2.index[-1].strftime('%m/%Y')}")
print(f"   Resíduo máximo: {residuals2.max():.2f}")
print(f"   Resíduo mínimo: {residuals2.min():.2f}")
print(f"   Desvio padrão: {residuals2.std():.2f}")
print(f"   AIC: {results2.aic:.2f} | BIC: {results2.bic:.2f}")

# =============================
# ESTRATÉGIA 3: USAR VARIÁVEIS DUMMY PARA OUTLIERS
# =============================
print("\n" + "="*80)
print("ESTRATÉGIA 3: ADICIONAR VARIÁVEIS DUMMY PARA OUTLIERS")
print("="*80)
print("💡 Identifica outliers automaticamente e adiciona dummies para controlá-los")

df3 = df_original.copy()

# Treinar modelo preliminar para identificar outliers
model_temp = SARIMAX(df3["Log_Total"], exog=df3[["Log_Clientes"]], order=(1,1,1), 
                     seasonal_order=(1,1,0,6), enforce_stationarity=False, enforce_invertibility=False)
results_temp = model_temp.fit(disp=False)
residuals_temp = results_temp.resid

# Identificar outliers (resíduos > 3 desvios padrão)
threshold = 3 * residuals_temp.std()
outlier_indices = np.where(np.abs(residuals_temp) > threshold)[0]

print(f"🔍 Outliers identificados (|resíduo| > {threshold:.2f}):")
for idx in outlier_indices:
    date = df3.index[idx]
    residual = residuals_temp.iloc[idx]
    print(f"   - {date.strftime('%m/%Y')}: resíduo = {residual:.2f}")

# Criar variáveis dummy para cada outlier
X_full3 = df3[["Log_Clientes"]].copy()
for idx in outlier_indices:
    dummy_name = f"Dummy_{df3.index[idx].strftime('%Y%m')}"
    X_full3[dummy_name] = 0
    X_full3.iloc[idx, X_full3.columns.get_loc(dummy_name)] = 1

y_full3 = df3["Log_Total"]

model3 = SARIMAX(y_full3, exog=X_full3, order=(1,1,1), seasonal_order=(1,1,0,6),
                 enforce_stationarity=False, enforce_invertibility=False)
results3 = model3.fit(disp=False)
residuals3 = results3.resid

print(f"\n✅ Modelo ajustado com {len(outlier_indices)} variáveis dummy")
print(f"   Resíduo máximo: {residuals3.max():.2f}")
print(f"   Resíduo mínimo: {residuals3.min():.2f}")
print(f"   Desvio padrão: {residuals3.std():.2f}")
print(f"   AIC: {results3.aic:.2f} | BIC: {results3.bic:.2f}")

# =============================
# ESTRATÉGIA 4: WINSORIZAÇÃO (LIMITAR VALORES EXTREMOS)
# =============================
print("\n" + "="*80)
print("ESTRATÉGIA 4: WINSORIZAÇÃO DOS DADOS")
print("="*80)
print("💡 Substitui valores extremos pelos percentis 5% e 95%")

from scipy.stats import mstats

df4 = df_original.copy()

# Aplicar winsorização na variável Log_Total
log_total_winsorized = mstats.winsorize(df4["Log_Total"], limits=[0.05, 0.05])
df4["Log_Total_Wins"] = log_total_winsorized

# Mostrar quais valores foram alterados
changed_indices = np.where(df4["Log_Total"] != df4["Log_Total_Wins"])[0]
if len(changed_indices) > 0:
    print(f"\n📝 Valores alterados pela winsorização:")
    for idx in changed_indices:
        date = df4.index[idx]
        original = df4["Log_Total"].iloc[idx]
        winsorized = df4["Log_Total_Wins"].iloc[idx]
        print(f"   - {date.strftime('%m/%Y')}: {original:.4f} → {winsorized:.4f}")
else:
    print(f"\n📝 Nenhum valor foi alterado pela winsorização")

y_full4 = df4["Log_Total_Wins"]
X_full4 = df4[["Log_Clientes"]]

model4 = SARIMAX(y_full4, exog=X_full4, order=(1,1,1), seasonal_order=(1,1,0,6),
                 enforce_stationarity=False, enforce_invertibility=False)
results4 = model4.fit(disp=False)
residuals4 = results4.resid

print(f"\n✅ Modelo ajustado com dados winzorizados")
print(f"   Resíduo máximo: {residuals4.max():.2f}")
print(f"   Resíduo mínimo: {residuals4.min():.2f}")
print(f"   Desvio padrão: {residuals4.std():.2f}")
print(f"   AIC: {results4.aic:.2f} | BIC: {results4.bic:.2f}")

# =============================
# ESTRATÉGIA 5: ROBUST REGRESSION (MODELO ROBUSTO)
# =============================
print("\n" + "="*80)
print("ESTRATÉGIA 5: MODELO COM TRATAMENTO ROBUSTO DE OUTLIERS")
print("="*80)
print("💡 Usa parâmetros do SARIMAX que minimizam influência de outliers")

df5 = df_original.copy()
y_full5 = df5["Log_Total"]
X_full5 = df5[["Log_Clientes"]]

# Usar cov_type='robust' para estimativas robustas
model5 = SARIMAX(y_full5, exog=X_full5, order=(1,1,1), seasonal_order=(1,1,0,6),
                 enforce_stationarity=False, enforce_invertibility=False)
results5 = model5.fit(disp=False, cov_type='robust')
residuals5 = results5.resid

print(f"✅ Modelo robusto ajustado")
print(f"   Resíduo máximo: {residuals5.max():.2f}")
print(f"   Resíduo mínimo: {residuals5.min():.2f}")
print(f"   Desvio padrão: {residuals5.std():.2f}")
print(f"   AIC: {results5.aic:.2f} | BIC: {results5.bic:.2f}")

# =============================
# COMPARAÇÃO VISUAL DE TODAS AS ESTRATÉGIAS
# =============================
print("\n" + "="*80)
print("COMPARAÇÃO VISUAL DAS ESTRATÉGIAS")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle('Comparação de Resíduos - Diferentes Estratégias de Tratamento', 
             fontsize=16, fontweight='bold', y=0.995)

# Estratégia 1: Original
ax1 = axes[0, 0]
ax1.plot(df_original.index, residuals1, color='blue', linewidth=2, marker='o', markersize=4)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax1.axhline(y=2*residuals1.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.axhline(y=-2*residuals1.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.set_title('1. ORIGINAL (Baseline)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Resíduos', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, f'σ = {residuals1.std():.2f}\nAIC = {results1.aic:.1f}', 
         transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Estratégia 2: Sem primeiras observações
ax2 = axes[0, 1]
ax2.plot(df2.index, residuals2, color='green', linewidth=2, marker='o', markersize=4)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax2.axhline(y=2*residuals2.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axhline(y=-2*residuals2.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.set_title('2. REMOVER 6 MESES INICIAIS', fontweight='bold', fontsize=12)
ax2.set_ylabel('Resíduos', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.98, f'σ = {residuals2.std():.2f}\nAIC = {results2.aic:.1f}', 
         transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Estratégia 3: Dummies
ax3 = axes[1, 0]
ax3.plot(df_original.index, residuals3, color='purple', linewidth=2, marker='o', markersize=4)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax3.axhline(y=2*residuals3.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax3.axhline(y=-2*residuals3.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax3.set_title(f'3. VARIÁVEIS DUMMY ({len(outlier_indices)} outliers)', fontweight='bold', fontsize=12)
ax3.set_ylabel('Resíduos', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.text(0.02, 0.98, f'σ = {residuals3.std():.2f}\nAIC = {results3.aic:.1f}', 
         transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Estratégia 4: Winsorização
ax4 = axes[1, 1]
ax4.plot(df_original.index, residuals4, color='brown', linewidth=2, marker='o', markersize=4)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax4.axhline(y=2*residuals4.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax4.axhline(y=-2*residuals4.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax4.set_title('4. WINSORIZAÇÃO (5%-95%)', fontweight='bold', fontsize=12)
ax4.set_ylabel('Resíduos', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.text(0.02, 0.98, f'σ = {residuals4.std():.2f}\nAIC = {results4.aic:.1f}', 
         transform=ax4.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Estratégia 5: Robusto
ax5 = axes[2, 0]
ax5.plot(df_original.index, residuals5, color='darkred', linewidth=2, marker='o', markersize=4)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax5.axhline(y=2*residuals5.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax5.axhline(y=-2*residuals5.std(), color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax5.set_title('5. MODELO ROBUSTO', fontweight='bold', fontsize=12)
ax5.set_ylabel('Resíduos', fontweight='bold')
ax5.set_xlabel('Data', fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.text(0.02, 0.98, f'σ = {residuals5.std():.2f}\nAIC = {results5.aic:.1f}', 
         transform=ax5.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Comparação de distribuições
ax6 = axes[2, 1]
ax6.boxplot([residuals1, residuals2, residuals3, residuals4, residuals5],
            labels=['Original', 'Sem 6\ninicial', 'Dummy', 'Winsor', 'Robusto'],
            patch_artist=True)
ax6.set_title('Distribuição dos Resíduos', fontweight='bold', fontsize=12)
ax6.set_ylabel('Resíduos', fontweight='bold')
ax6.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# =============================
# TABELA RESUMO COMPARATIVA
# =============================
print("\n" + "="*80)
print("RESUMO COMPARATIVO DAS ESTRATÉGIAS")
print("="*80)

comparison_data = {
    'Estratégia': [
        '1. Original (Baseline)',
        '2. Remover 6 meses iniciais',
        '3. Variáveis Dummy',
        '4. Winsorização',
        '5. Modelo Robusto'
    ],
    'Resíduo_Máx': [
        residuals1.max(),
        residuals2.max(),
        residuals3.max(),
        residuals4.max(),
        residuals5.max()
    ],
    'Resíduo_Mín': [
        residuals1.min(),
        residuals2.min(),
        residuals3.min(),
        residuals4.min(),
        residuals5.min()
    ],
    'Desvio_Padrão': [
        residuals1.std(),
        residuals2.std(),
        residuals3.std(),
        residuals4.std(),
        residuals5.std()
    ],
    'AIC': [
        results1.aic,
        results2.aic,
        results3.aic,
        results4.aic,
        results5.aic
    ],
    'BIC': [
        results1.bic,
        results2.bic,
        results3.bic,
        results4.bic,
        results5.bic
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + df_comparison.to_string(index=False))

# Identificar melhor estratégia
best_idx = df_comparison['Desvio_Padrão'].idxmin()
best_strategy = df_comparison.loc[best_idx, 'Estratégia']

print("\n" + "="*80)
print("RECOMENDAÇÃO")
print("="*80)
print(f"\n🏆 MELHOR ESTRATÉGIA (menor desvio padrão dos resíduos):")
print(f"   {best_strategy}")
print(f"   - Desvio Padrão: {df_comparison.loc[best_idx, 'Desvio_Padrão']:.4f}")
print(f"   - AIC: {df_comparison.loc[best_idx, 'AIC']:.2f}")
print(f"   - Resíduo Máximo: {df_comparison.loc[best_idx, 'Resíduo_Máx']:.2f}")

print("\n💡 INTERPRETAÇÃO:")
print("   • Menor desvio padrão = resíduos mais homogêneos e previsíveis")
print("   • Menor AIC/BIC = melhor ajuste considerando complexidade do modelo")
print("   • Resíduos mais próximos de zero = modelo captura melhor os padrões")

print("\n📌 PRÓXIMOS PASSOS:")
print("   1. Escolha a estratégia que melhor se adequa ao seu problema")
print("   2. Considere se remover dados iniciais impacta suas previsões futuras")
print("   3. Variáveis dummy são boas para manter todos os dados e controlar outliers")
print("   4. Winsorização é conservadora e mantém a estrutura temporal")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA!")
print("="*80)
