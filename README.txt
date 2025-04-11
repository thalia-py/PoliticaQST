# Política de Manutenção Preventiva Oportuna em Três Fases (Política QST)

Este software, desenvolvido em Python com Streamlit, realiza a modelagem e otimização da política de manutenção QST com base em distribuições de Weibull para os tempos de defeito e falha.

## Funcionalidades
- Cálculo da taxa de custo de manutenção
- Estimativa de MTBOF (Tempo Médio Entre Falhas Observadas)
- Otimização automática dos parâmetros Q, S e T
- Avaliação de desempenho para políticas definidas manualmente
- Interface amigável e intuitiva

## Requisitos
- Python 3.8+
- Bibliotecas:
  - streamlit
  - numpy
  - scipy

## Como executar
```bash
streamlit run sft.py
