# IRRBB Simulation & Hedging Engine for EVE and NII metrics

## Project Overview

A full end-to-end **Interest Rate Risk in the Banking Book (IRRBB)** simulation framework designed using Python, the project models a sample bank balance sheet, runs Basel IRRBB shock scenarios and aims at finding interest rate swap hedges to control:
- Economic Value of Equity (EVE)
- Net Interest Income (NII)

It is designed as a mini ALM/Treasury risk engine that is similar to production bank risk systems.

## Project Objectives
This project demonstrates how to:
- Build a modular IRRBB risk engine
- Simulate Basel interest rate shock scenarios
- Model behavioral banking products (NMDs)
- Compute EVE and NII sensitivities
- Derive key-rate DV01 exposures
- Optimize interest rate swap hedging strategies
- Apply realistic optimization constraints (soft & hard constraints)

## Key Features

### **IRRBB Metrics**
- Full EVE/NII engine
- Basel shock scenarios:
    * Parallel up /down
    * Short rate up / down
    * Steepener / Flattener

### **Behavioral Modelling**
The project includes simulating an ALM behavoral modelling by incorporating Non-MAturity Deposit (NMD) model.
- Core / non-core split
- Deposit stickiness (beta)
- Average life modelling

### **Balance Sheet**
A simplified balance sheet supporting multiple banking instruments 
**Assets**
- Fixed rate loans
- Floating rate loans

**Liabilities**
- Non-maturity deposits

**Hedges**
- IR Swaps across multiple tenors

### **Hedging Engine**
A set of hedge optimizers are implemented in this project:

#### **1. DV01 Neutral Hedge**
Minimizing the total bank DV01 exposure

#### **2. Scenario-based EVE Hedge**
Minimizing $\Delta EVE$ across Basel shock scenarios considering key-rate DV01 distribution

#### **3. Joint EVE & NII IRRBB Optimizer**
A joint IRRBB optimizer that:

Minimizes:
- $\Delta EVE$ and $\Delta NII$ risks
- Hedge Size (regularization term)

Subject to:
- EVE / NII risk limits
- Allowable hedge budget limit

### **Optimization Framework**
Two generations of optimizers are introduced in this project:

#### **1. Scenario Optimizer**

It targets minimizing the IRRBB risk metrics concurrently based on the Basel shock scenarios. 

$$
\textbf{min} \: w_{EVE}\sum_{s} (\Delta EVE_s^{2}) + w_{NII}\sum_{s} (\Delta NII_s^{2}) + \lambda \sum_{i} h_i^{2}
$$

**Subject to**

$$
\begin{aligned}
\sum_{i} |h_i| &\le B \\
|h_i| &\le T_i \quad \forall i \in \text{HedgeTenors}
\end{aligned}
$$

---

#### **2. Joint IRRBB Optimizer**

Rather than minimizing IRRBB risks, this approach minimizes the **total cost of hedging**.

$$
\textbf{min} \: \sum_i h_i \cdot c_i
$$

**Subject to**

$$
\begin{aligned}
|\Delta EVE_s| &\le L_{EVE} \quad \forall s \in \text{BaselScenarios} \\
|\Delta NII_s| &\le L_{NII} \quad \forall s \in \text{BaselScenarios} \\
|h_i| &\le T_i \quad \forall i \in \text{HedgeTenors}
\end{aligned}
$$


