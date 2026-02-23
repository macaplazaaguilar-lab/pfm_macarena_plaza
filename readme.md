# Greeks Computation by Monte Carlo Techniques

**Author:** Macarena Plaza
  
**Institution:** Universidade da Coruña (Carlos V.) Banco Santander (Beatriz Salvador, Daniel Arrieta)

## Overview

This project studies the computation of option sensitivities (Greeks) using Monte Carlo techniques. In quantitative finance, Greeks measure the impact of market movements and model parameters on the value of derivatives and are central to risk management.

The work focuses on comparing different Monte Carlo-based methodologies for pricing and sensitivity analysis, with applications to both standard option pricing and counterparty risk metrics.

## Objectives

1. **Option Pricing**
   - Price European options under the Black–Scholes model using Monte Carlo simulation.
   - Extend the framework by incorporating stochastic interest rates via the Hull–White model.

2. **Pathwise Sensitivity Method**
   - Study and implement the pathwise sensitivity method.
   - Compute **Delta**, **Gamma**, and **Theta** under the considered models.

3. **Finite Difference Comparison**
   - Implement the classical finite difference (bump-and-revalue) method.
   - Compare results, efficiency, and numerical stability against the pathwise approach.

4. **Application to CVA**
   - Apply Greek computation techniques to the Credit Valuation Adjustment (CVA) of an Interest Rate Swap (IRS).
   - Under suitable assumptions, exploit the representation of IRS CVA as a linear combination of swaptions.
   - Compute sensitivities of swaptions to obtain Greeks of the IRS CVA.

## Methodology

- Monte Carlo simulation for pricing.
- Pathwise derivative estimators.
- Finite difference estimators.
- Extension to stochastic short-rate modeling (Hull–White).

## Reference

- Grzelak, L.A., & Oosterlee, C.W. (2020). *Mathematical Modeling and Computation in Finance*. World Scientific Publishing Europe Ltd.