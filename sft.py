# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 10:09:19 2025

@author: Thalia Queiroz
"""

import time
import numpy as np
from scipy.stats import weibull_min
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import streamlit as st

st.markdown("""
    <style>
        /* Fundo preto */
        .stApp {
            background-color: #000000;
            color: white;
        }

        /* Títulos */
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }

        /* Textos em geral */
        .css-1cpxqw2, .css-10trblm, .css-1d391kg, .css-1v3fvcr {
            color: white;
        }

        /* Input boxes e widgets */
        .stNumberInput input {
            background-color: #111;
            color: white;
            border: 1px solid #00ff88;
        }

        /* Botões */
        .stButton > button {
            background-color: #00ff88;
            color: black;
            border: none;
            padding: 0.5em 1em;
            border-radius: 10px;
        }

        /* Logo da barra lateral (se usar) */
        .css-1aumxhk {
            background-color: #000000;
        }

        /* Remove foco azul nos inputs */
        input:focus, button:focus {
            outline: none;
            box-shadow: 0 0 0 2px #00ff88;
        }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# Layout Superior – Cabeçalho com logo e título
# =============================================================================
col1, col2 = st.columns([1, 4])
with col1:
    st.image(r"C:\Users\Thalia\Desktop\Software\logo_random.png.png", use_container_width=True)

with col2:
    st.markdown("""
        <div style='display: flex; align-items: center; height: 100%;'>
            <h1 style='color: white; text-align: left; font-size: 30px;'>
                Política de Manutenção Preventiva Oportuna em Três Fases (Política QST)
            </h1>
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Parâmetros do Modelo
# =============================================================================
st.header("📥 Parâmetros do Modelo")

col1, col2 = st.columns(2)

with col1:
    betax = st.number_input("Tempo até a chegada do defeito (X) – parâmetro de forma (Weibull)", format="%.7f", step=0.0000001)
    etax = st.number_input("Tempo até a chegada do defeito (X) – parâmetro de escala (Weibull)", format="%.7f", step=0.0000001)
    lambd = st.number_input("Taxa de chegada de oportunidades (λ)", format="%.7f", step=0.0000001)
    Cp = st.number_input("Custo de substituição preventiva programada (Cp)", format="%.7f", step=0.0000001)
    Co = st.number_input("Custo de substituição preventiva em oportunidade (Co)", format="%.7f", step=0.0000001)
    Dp = st.number_input("Tempo de parada para substituição preventiva programada (Dp)", format="%.7f", step=0.0000001)

with col2:
    betah = st.number_input("Tempo entre a chegada do defeito e a falha (H) – parâmetro de forma (Weibull)", format="%.7f", step=0.0000001)
    etah = st.number_input("Tempo entre a chegada do defeito e a falha (H) – parâmetro de escala (Weibull)", format="%.7f", step=0.0000001)
    Cf = st.number_input("Custo de substituição corretiva (Cf)", format="%.7f", step=0.0000001)
    Ci = st.number_input("Custo de inspeção (Ci)", format="%.7f", step=0.0000001)
    Df = st.number_input("Tempo de parada para substituição corretiva (Df)", format="%.7f", step=0.0000001)
        
# =============================================================================
# FUNÇÕES DE DISTRIBUIÇÃO
# =============================================================================
def fx(t, betax, etax):
    return ((betax / etax) * ((t / etax) ** (betax - 1))) * np.exp(-((t / etax) ** betax))

def Rx(t, betax, etax):
    return np.exp(-((t / etax) ** betax))

def Fx(t, betax, etax):
    return 1 - Rx(t, betax, etax)

def fh(t, betah, etah):
    return ((betah / etah) * ((t / etah) ** (betah - 1))) * np.exp(-((t / etah) ** betah))

def Rh(t, betah, etah):
    return np.exp(-((t / etah) ** betah))

def Fh(t, betah, etah):
    return 1 - Rh(t, betah, etah)

def fw(t, lambd):
    return lambd * np.exp(-lambd * t)

def Rw(t,lambd):
    return np.exp(-lambd * t)

def FW(t, lambd):
    return 1 - Rw(t, lambd)

# =============================================================================
# FUNÇÕES DOS CENÁRIOS
# =============================================================================
#cenário1
def P1(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 1"""
    return Rx(T, betax, etax) * Rw((T - S), lambd)

def EC1(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Cálculo do custo esperado do Cenário 1"""
    prob = P1(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob * (Cp + lambd * (S-Q) * Ci)

def EL1(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Cálculo da duração esperada do ciclo do Cenário 1"""
    prob = P1(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob * (T + Dp)

#cenário2
def P2(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 2"""
    integral, _ = quad(lambda x: fx(x, betax, etax) * Rh((T - x), betah, etah) * Rw((T - S), lambd), S, T)
    return integral

def EC2(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Custo esperado do Cenário 2"""
    prob = P2(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob * (Cp + lambd * (S-Q) * Ci)

def EL2(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 2"""
    prob = P2(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob * (T + Dp)

#cenário3
def P3(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 3"""
    integral, _ = quad(lambda x: fx(x, betax, etax) * Rh((T - x), betah, etah) * Rw((T - x), lambd), Q, S)
    return integral

def EC3(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Custo esperado do Cenário 3"""
    integral, _ = quad(lambda x: (Cp + lambd * (x-Q) * Ci) * fx(x, betax, etax) * Rh((T - x), betah, etah) * Rw((T - x), lambd), Q, S)
    return integral

def EL3(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 3"""
    prob = P3(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob*(T + Dp)

#cenário4
def P4(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 4 (novo)"""
    integral, _ = quad(lambda x: fx(x, betax, etax) * Rh((T - x), betah, etah) * Rw((T - Q), lambd), 0, Q)
    return integral

def EC4(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Custo esperado do Cenário 4 (novo)"""
    prob = P4(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob*Cp

def EL4(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 4(novo)"""
    prob = P4(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob*(T + Dp)

#cenário5
def P5(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 5"""
    integral, _ = quad(lambda w: fw(w, lambd) * Rx((S + w), betax, etax), 0, T - S)
    return integral

def EC5(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Custo esperado do Cenário 5"""
    prob = P5(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob * (Co + lambd * (S-Q) * Ci)

def EL5(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 5"""
    integral, _ = quad(lambda w: fw(w, lambd) * Rx((S + w), betax, etax) * (S + w + Dp), 0, T - S)
    return integral

## CENÁRIO 6 (antigo cenário 5)

def P6(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 6"""
    integral, _ = dblquad(
        lambda x, w: fw(w, lambd) * fx(x, betax, etax) * Rh((S + w - x), betah, etah),
        0, T - S,
        lambda w: S,
        lambda w: S + w
    )
    return integral

def EC6(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Custo esperado do Cenário 6"""
    prob = P6(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob * (Co + lambd * (S-Q) * Ci)

def EL6(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 6"""
    integral, _ = dblquad(
        lambda x, w: (S + w + Dp) * fw(w, lambd) * fx(x, betax, etax) * Rh((S + w - x), betah, etah),
        0, T - S,
        lambda w: S,
        lambda w: S + w
    )
    return integral

# CENÁRIO 7 (antigo cenário 6)

def P7(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 7"""
    integral, _ = dblquad(
        lambda w, x: fx(x, betax, etax) * fw(w, lambd) * Rh(w, betah, etah),
        Q, S,
        lambda x: 0,
        lambda x: T - x
    )
    return integral

def EC7(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Custo esperado do Cenário 7"""
    integral, _ = dblquad(
        lambda w, x: (Co + lambd * (x-Q) * Ci) * fx(x, betax, etax) * fw(w, lambd) * Rh(w, betah, etah),
        Q, S,
        lambda x: 0,
        lambda x: T - x
    )
    return integral

def EL7(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 6"""
    integral, _ = dblquad(
        lambda w, x: (x + w + Dp) * fx(x, betax, etax) * fw(w, lambd) * Rh(w, betah, etah),
        Q, S,
        lambda x: 0,
        lambda x: T - x
    )
    return integral

########################################
# CENÁRIO 8 (NOVO)

def P8(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 8"""
    integral, _ = dblquad(
        lambda w, x: fx(x, betax, etax) * fw(w, lambd) * Rh((Q+w-x), betah, etah),
        0, Q,
        lambda x: 0,
        lambda x: T-Q
    )
    return integral

def EC8(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    prob = P8(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob*Co

def EL8(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 8"""
    integral, _ = dblquad(
        lambda w, x: (x + w + Dp) * fx(x, betax, etax) * fw(w, lambd) * Rh((Q+w-x), betah, etah),
        0, Q,
        lambda x: 0,
        lambda x: T-Q
    )
    return integral

##############################
# CENÁRIO 9 (antigo cenário 7)

def P9(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 7"""
    integral, _ = dblquad(
        lambda h, x: fx(x, betax, etax) * fh(h, betah, etah) * Rw(x + h - S, lambd),
        S, T,
        lambda x: 0,
        lambda x: T - x
    )
    return integral

def EC9(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Custo esperado do Cenário 7"""
    prob = P9(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return prob * (Cf + lambd * (S-Q) * Ci)

def EL9(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 7"""
    integral, _ = dblquad(
        lambda h, x: (x + h + Df) * fx(x, betax, etax) * fh(h, betah, etah) * Rw(x + h - S, lambd),
        S, T,
        lambda x: 0,
        lambda x: T - x
    )
    return integral

########################
# CENÁRIO 10 - antigo cenário 8


def P10(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 8"""
    integral, _ = dblquad(
        lambda h, x: fx(x, betax, etax) * fh(h, betah, etah) * Rw(h, lambd),
        Q, S,
        lambda x: 0,
        lambda x: T - x
    )
    return integral

def EC10(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Custo esperado do Cenário 8"""
    integral, _ = dblquad(
        lambda h, x: (Cf + lambd * (x-Q) * Ci) * fx(x, betax, etax) * fh(h, betah, etah) * Rw(h, lambd),
        Q, S,
        lambda x: 0,
        lambda x: T - x
    )
    return integral

def EL10(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 8"""
    integral, _ = dblquad(
        lambda h, x: (x + h + Df) * fx(x, betax, etax) * fh(h, betah, etah) * Rw(h, lambd),
        Q, S,
        lambda x: 0,
        lambda x: T - x
    )
    return integral

################################################
###### CENÁRIO 11 (NOVO)

def P11(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 8"""
    integral, _ = dblquad(
        lambda h, x: fx(x, betax, etax) * fh(h, betah, etah) * Rw((x+h-Q), lambd),
        0, Q,
        lambda x: Q - x,
        lambda x: T - x
    )
    return integral

def EC11(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    prob = P11(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return Cf*prob

def EL11(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 8"""
    integral, _ = dblquad(
        lambda h, x: (x + h + Df) * fx(x, betax, etax) * fh(h, betah, etah) * Rw((x+h-Q), lambd),
        0, Q,
        lambda x: Q - x,
        lambda x: T - x
    )
    return integral

##############################################
# CENÁRIO 12 (NOVO)

def P12(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Probabilidade de ocorrência do Cenário 8"""
    integral, _ = dblquad(
        lambda h, x: fx(x, betax, etax) * fh(h, betah, etah),
        0, Q,
        lambda x: 0,
        lambda x: Q - x
    )
    return integral

def EC12(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    prob = P12(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return Cf*prob

def EL12(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    """Duração esperada do ciclo do Cenário 8"""
    integral, _ = dblquad(
        lambda h, x: (x + h + Df) * fx(x, betax, etax) * fh(h, betah, etah),
        0, Q,
        lambda x: 0,
        lambda x: Q - x
    )
    return integral

# =============================================================================
# CÁLCULOS
# =============================================================================
def P_total (Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    p_total = 0
    p_total = p_total + P1(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P2(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P3(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P4(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P5(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P6(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P7(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P8(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P9(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P10(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P11(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_total = p_total + P12(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return p_total

def P_falha(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    p_falha = 0
    p_falha = p_falha + P9(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_falha = p_falha + P10(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_falha = p_falha + P11(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    p_falha = p_falha + P12(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return p_falha
    
def EC_ciclo (Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    EC = 0
    EC = EC + EC1(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC2(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC3(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC4(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC5(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC6(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC7(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC8(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC9(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC10(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC11(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EC = EC + EC12(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return EC

def EL_ciclo (Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    EL = 0
    EL = EL + EL1(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL2(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL3(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL4(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL5(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL6(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL7(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL8(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL9(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL10(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL11(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL = EL + EL12(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return EL

def taxa_custo(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    EC_ = EC_ciclo (Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    EL_ = EL_ciclo (Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return (EC_/EL_)

def MTBOF(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df):
    EL_ = EL_ciclo (Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    pf_ = P_falha(Q, S, T, betax,  etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
    return (EL_/pf_)
# =============================================================================
# OTIMIZAÇÃO COM DIFFERENTIAL EVOLUTION
# =============================================================================
if st.button("🚀 Otimizar"):
    
    cenarios = [(betax, etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)]
    matriz_resultados = [[None]*5 for _ in range(len(cenarios))]
    
    # Iteração sobre os cenários 
    for i in range(len(cenarios)):
        betax_i, etax_i, betah_i, etah_i, lambd_i, Ci_i, Co_i, Cp_i, Cf_i, Dp_i, Df_i = cenarios[i]
        
        # Função objetivo: mapeia x = [pQ, pS, T] para Q, S e calcula a taxa de custo
        def objetivo(x):
            pQ, pS, T_val = x
            S_val = T_val * pS
            Q_val = S_val * pQ
            return taxa_custo(Q_val, S_val, T_val, betax_i, etax_i, betah_i, etah_i, lambd_i, Ci_i, Co_i, Cp_i, Cf_i, Dp_i, Df_i)
        
        # Chute inicial e limites:
        # pQ e pS variam entre 0 e 1; T varia entre 0 e (etax+etah)
        x0 = [0.5, 0.5, etax_i]
        bounds = [(0,1), (0,1), (0, etax_i+etah_i)]
        
        with st.spinner("⏳ Otimizando a política QST... Aguarde..."):
            resultado = differential_evolution(objetivo, bounds=bounds, popsize=10, maxiter=50, tol=0.1)
        
        pQ, pS, T_opt = resultado.x
        S_opt = T_opt * pS
        Q_opt = S_opt * pQ
        taxa_ot = taxa_custo(Q_opt, S_opt, T_opt, betax_i, etax_i, betah_i, etah_i, lambd_i, Ci_i, Co_i, Cp_i, Cf_i, Dp_i, Df_i)
        MTBOF_opt = MTBOF(Q_opt, S_opt, T_opt, betax_i, etax_i, betah_i, etah_i, lambd_i, Ci_i, Co_i, Cp_i, Cf_i, Dp_i, Df_i)
        
        # Armazena os resultados
        matriz_resultados[i][0] = Q_opt
        matriz_resultados[i][1] = S_opt
        matriz_resultados[i][2] = T_opt
        matriz_resultados[i][3] = taxa_ot
        matriz_resultados[i][4] = MTBOF_opt
        
        # Exibe os resultados para este cenário
        st.success("Otimização concluída!")
        st.markdown("### 🔍 Resultados Otimizados")
        col_res1, col_res2, col_res3, col_res4, col_res5 = st.columns(5)
        col_res1.metric(label="🕒 Q otimizado", value=f"{Q_opt:.2f}")
        col_res2.metric(label="🕒 S otimizado", value=f"{S_opt:.2f}")
        col_res3.metric(label="⏱️ T otimizado", value=f"{T_opt:.2f}")
        col_res4.metric(label="💰 Custo Mínimo", value=f"{taxa_ot:.4f}")
        col_res5.metric(label="📈 MTBOF (h)", value=f"{MTBOF_opt:.2f}")

# =============================================================================
# AVALIAR POLÍTICA DEFINIDA MANUALMENTE
# =============================================================================
st.header("🧪 Avaliação de Política Pré-Definida pelo Usuário")

# Entrada dos valores manualmente definidos
Q_manual = st.number_input("Valor de Q (início de inspeções oportunas)", format="%.7f", step=0.0000001)
S_manual = st.number_input("Valor de S (limite para inspeções oportunas)", format="%.7f", step=0.0000001)
T_manual = st.number_input("Valor de T (substituição programada)", format="%.7f", step=0.0000001)

# Botão para calcular o desempenho da política manual
if st.button("📊 Avaliar política pré-definida"):
    with st.spinner("🔍 Calculando desempenho da política..."):
        taxa_manual = taxa_custo(Q_manual, S_manual, T_manual, betax, etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)
        MTBOF_manual = MTBOF(Q_manual, S_manual, T_manual, betax, etax, betah, etah, lambd, Ci, Co, Cp, Cf, Dp, Df)

    st.markdown("### 🎯 Desempenho da Política Informada")
    colm1, colm2 = st.columns(2)
    colm1.metric(label="💰 Taxa de Custo", value=f"{taxa_manual:.4f}")
    colm2.metric(label="📈 MTBOF (h)", value=f"{MTBOF_manual:.2f}")

# =============================================================================
# Rodapé
# =============================================================================
st.markdown("""
<hr style="border:0.5px solid #333;" />

<div style='color: #aaa; font-size: 13px; text-align: left;'>
    <strong style="color: #ccc;">RANDOM - Grupo de Pesquisa em Risco e Análise de Decisão em Operações e Manutenção</strong><br>
    Criado em 2012, o grupo reúne pesquisadores dedicados às áreas de risco, manutenção e modelagem de operações.<br>
    <a href='http://random.org.br' target='_blank' style='color:#888;'>Acesse o site do RANDOM</a>
</div>
""", unsafe_allow_html=True)


