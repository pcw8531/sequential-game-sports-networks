# Data Sources and Provenance

This document describes the empirical data sources used for the cross-domain validation (Figure 5) and the Ferrall and Smith comparison (SI Text S7).

## Cross-Domain Calibration Data

All empirical calibration data are drawn from Park (2026), *Scientific Reports* 16:4595, which fitted the centrality-based protection heuristic f_p = f_p0 + f_p1 · C_E across three domains.

### Banking
- **Source:** Bank for International Settlements (BIS) Basel III Monitoring Reports, 2015–2024
- **Sample:** n = 127 globally systemically important banks (G-SIBs)
- **Protection proxy:** Tier 1 capital ratio
- **Centrality measure:** Interbank network eigenvector centrality

### Epidemiology
- **Source:** WHO/Our World in Data (OWID) COVID-19 vaccination data, 2020–2023
- **Sample:** n = 43 OECD countries
- **Protection proxy:** Vaccination coverage rate
- **Centrality measure:** International travel network eigenvector centrality

### Sports
- **Source:** UEFA Financial Fair Play (FFP) Reports and NFL Health and Safety Reports, 2015–2019
- **Sample:** n = 85 teams (combined European football and American football)
- **Protection proxy:** Injury prevention investment / medical staff ratio
- **Centrality measure:** Team interaction network eigenvector centrality

## Ferrall and Smith (1999) Comparison Data

- **Source:** Ferrall, C. & Smith, A. A. Jr. (1999). A sequential game model of sports championship series: Theory and estimation. *Review of Economics and Statistics* 81, 704–719.
- **Sample:** 168 championship series across baseball (n = 85, 1905–1996), basketball (n = 43, 1947–1997), and hockey (n = 40, 1943–1997)
- **Variables:** Cost-of-effort estimates, home advantage parameters, series length distributions

## Empirical Football Network (Figure 5A)

- **Source:** Grund, T. U. (2012). Network structure and team performance: The case of English Premier League soccer teams. *Social Networks* 34, 682–690.
- **Sample:** n = 11 players, 4-3-3 formation passing network
- **Variables:** Pass frequency, eigenvector centrality, positional roles

## Note on Data Availability

Original empirical datasets are available from the cited published sources. The simulation-generated data (Figures 2–4) are fully reproducible using the code in this repository with the parameters specified in the manuscript.
