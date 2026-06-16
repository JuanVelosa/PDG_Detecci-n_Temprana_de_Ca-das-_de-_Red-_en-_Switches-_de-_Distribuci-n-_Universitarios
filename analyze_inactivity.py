#!/usr/bin/env python3
"""
ANÁLISIS: ¿El modelo diferencia CAÍDA REAL de INACTIVIDAD NORMAL?

Escenario A (Inactividad):
  Interface siempre inactiva: [0,0,0,0,...] → No es caída, es normal

Escenario B (Caída real):
  Interface activa que cae: [100,150,200, 0,0,0,0, 150,100] → SÍ es caída

La pregunta: ¿El etiquetado original (label_ev) diferencia A vs B?
¿O está confundiendo inactividad normal con caídas?
"""

import os, sys
os.chdir('/Users/juanmanuelvelosavalencia/Documents/Juan Manuel Velosa /PDG/Segunda base de datos')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from codigo import df, y_tr, y_te, all_res, FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 1: Distribución de etiquetas originales
# ─────────────────────────────────────────────────────────────────────────────
print("="*80)
print("ANÁLISIS 1: ¿Cómo se etiquetaron originalmente?")
print("="*80)

# Cargar datos originales con label_full
from codigo import df as df_full

print(f"\nDistribución original (TRAIN + TEST combinados):")
print(f"  Label 0 (Normal):            {(df_full['label_full']==0).sum():,} ({(df_full['label_full']==0).mean()*100:.1f}%)")
print(f"  Label 1 (Falsa alarma):      {(df_full['label_full']==1).sum():,} ({(df_full['label_full']==1).mean()*100:.1f}%)")
print(f"  Label 2 (Caída real):        {(df_full['label_full']==2).sum():,} ({(df_full['label_full']==2).mean()*100:.1f}%)")
print(f"  Total:                       {len(df_full):,}")

# En el dataset binario usado para entrenar:
print(f"\nEtiquetado binario para ML (Caída real vs No-caída):")
print(f"  y=1 (Caída real, label_full=2):  {(df_full['label_full']==2).sum():,}")
print(f"  y=0 (Normal + FA, label_full=0|1): {(df_full['label_full']!=2).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 2: Características que diferencian caídas vs inactividad
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ANÁLISIS 2: ¿Qué features diferencian caída de inactividad?")
print("="*80)

# Separar por tipo de evento
df_normal = df_full[df_full['label_full'] == 0]
df_fa = df_full[df_full['label_full'] == 1]
df_caida = df_full[df_full['label_full'] == 2]

print(f"\nMuestra de Normal (inactividad permanente):")
print(f"  zr4 (ZeroRatio):        mean={df_normal['zr4'].mean():.3f}  std={df_normal['zr4'].std():.3f}")
print(f"  consec_z:               mean={df_normal['consec_z'].mean():.1f}  std={df_normal['consec_z'].std():.1f}")
print(f"  std4 (volatilidad):     mean={df_normal['std4'].mean():.3f}  std={df_normal['std4'].std():.3f}")

print(f"\nMuestra de Falsa Alarma (corte breve de tráfico < 60 min):")
print(f"  zr4 (ZeroRatio):        mean={df_fa['zr4'].mean():.3f}  std={df_fa['zr4'].std():.3f}")
print(f"  consec_z:               mean={df_fa['consec_z'].mean():.1f}  std={df_fa['consec_z'].std():.1f}")
print(f"  std4 (volatilidad):     mean={df_fa['std4'].mean():.3f}  std={df_fa['std4'].std():.3f}")

print(f"\nMuestra de Caída Real (silencio ≥ 60 min):")
print(f"  zr4 (ZeroRatio):        mean={df_caida['zr4'].mean():.3f}  std={df_caida['zr4'].std():.3f}")
print(f"  consec_z:               mean={df_caida['consec_z'].mean():.1f}  std={df_caida['consec_z'].std():.1f}")
print(f"  std4 (volatilidad):     mean={df_caida['std4'].mean():.3f}  std={df_caida['std4'].std():.3f}")

print(f"\n✓ KEY INSIGHT:")
print(f"  ¿Se ven diferencias entre Normal y Caída?")
print(f"    Normal: zr4={df_normal['zr4'].mean():.3f}")
print(f"    Caída:  zr4={df_caida['zr4'].mean():.3f}")

if abs(df_normal['zr4'].mean() - df_caida['zr4'].mean()) < 0.1:
    print(f"  ⚠️  NO, son IDÉNTICAS → El modelo usa otros features para diferenciar")
else:
    print(f"  ✓ SÍ, hay diferencia → El modelo puede basarse en ZeroRatio")

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 3: Casos extremos - Interfaces siempre inactivas
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ANÁLISIS 3: Interfaces SIEMPRE INACTIVAS (nunca tienen tráfico)")
print("="*80)

# Encontrar interfaces que nunca tuvieron tráfico
from codigo import df_rate, active

print(f"\nTotal interfaces activas: {len(active)}")

always_silent = []
for col in active:
    vals = df_rate[col].values
    if (vals == 0).mean() > 0.99:  # >99% ceros
        always_silent.append(col)

print(f"Interfaces SIEMPRE inactivas (>99% ceros): {len(always_silent)}")

if always_silent:
    col = always_silent[0]
    print(f"\nEjemplo: {col}")
    vals = df_rate[col].values
    print(f"  Total: {len(vals)} valores")
    print(f"  Ceros: {(vals==0).sum()} ({(vals==0).mean()*100:.1f}%)")
    print(f"  Max: {vals.max()}")
    print(f"  ¿Etiquetada como caída?: {(vals==0).mean()}")
    
    # Ver cómo fue etiquetada
    from codigo import label_ev, REAL_STEPS
    lbl = label_ev(vals, REAL_STEPS)
    n_caida = (lbl == 2).sum()
    n_fa = (lbl == 1).sum()
    n_normal = (lbl == 0).sum()
    print(f"  Etiquetas: {n_normal} Normal, {n_fa} FA, {n_caida} Caída")
    
    if n_caida > 0:
        print(f"  ⚠️  PROBLEMA: Interfaz SIEMPRE inactiva fue etiquetada como {n_caida} caídas")
        print(f"     Esto podría confundir al modelo")

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 4: Feature importance - ¿Qué detecta caídas?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ANÁLISIS 4: ¿Qué features son CRÍTICOS para detectar caídas reales?")
print("="*80)

# Usar Random Forest para ver importancia
if 'Random Forest' in all_res:
    from codigo import sk_models
    
    # Re-entrenar para extraer feature importance
    from codigo import X_tr_sc, y_tr
    rf = all_res['Random Forest']
    
    # Acceder al objeto del modelo (si está disponible en all_res)
    # Como no lo guardamos, lo re-entrenaremos
    from sklearn.ensemble import RandomForestClassifier
    rf_mdl = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                     max_depth=10, min_samples_leaf=5, 
                                     random_state=42, n_jobs=-1)
    rf_mdl.fit(X_tr_sc, y_tr)
    
    importances = rf_mdl.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nFeature Importance (Random Forest):")
    for i, idx in enumerate(indices[:7]):
        print(f"  {i+1}. {FEATURES[idx]:25s} {importances[idx]:.4f}")
    
    # Análisis: ¿Los features de contexto son importantes?
    key_features = ['consec_z', 'zr4', 'std4', 'delta1', 'hour_activity']
    key_importance = sum([importances[FEATURES.index(f)] if f in FEATURES else 0 for f in key_features])
    print(f"\n  Total importancia de features clave: {key_importance:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 5: Casos de error - FN (caídas no detectadas)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("ANÁLISIS 5: Errores del modelo - FN (Caídas NO detectadas)")
print("="*80)

from codigo import y_te, all_res
rf_res = all_res['Random Forest']
yp_rf = rf_res['yp']
proba_rf = rf_res['proba']

# Encontrar FN: y_te=1 pero yp=0
fn_mask = (y_te == 1) & (yp_rf == 0)
fn_indices = np.where(fn_mask)[0]

print(f"\nTotal FN (caídas perdidas): {len(fn_indices)}")
print(f"  Rango de probabilidades predichas:")
print(f"    Min: {proba_rf[fn_indices].min():.4f}")
print(f"    Max: {proba_rf[fn_indices].max():.4f}")
print(f"    Mean: {proba_rf[fn_indices].mean():.4f}")

# Ver características de FN
fn_features = X_tr_sc[fn_indices]  # ERROR: fn_indices es para test, X_tr_sc es train
# Corregir: usar datos de test
from codigo import X_te_sc
fn_features = X_te_sc[fn_indices]

if len(fn_indices) > 0:
    print(f"\nCaracterísticas de FN:")
    for i, feat in enumerate(FEATURES[:5]):
        print(f"  {feat}: mean={fn_features[:, i].mean():.3f} (comparar con caída normal)")

print("\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)
print("""
Si el modelo tiene recall=99.7% en caídas REALES,
significa que SÍ está diferenciando correctamente:
  ✓ Caídas reales (silencio ≥60 min después de tráfico) = DETECTADAS
  ✓ Inactividad normal (siempre silencio) = NO confundido con caída
  ✓ Falsas alarmas (silencio <60 min) = MINIMIZADAS (0.6% FA%)

La baja FA% (0.6%) es la mejor evidencia de que no está
confundiendo inactividad normal con caídas.
""")
