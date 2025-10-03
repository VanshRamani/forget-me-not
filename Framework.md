# A Framework for Principled Machine Unlearning

This document outlines two related frameworks for machine unlearning: a classical PAC-style framework for evaluating unlearning algorithms and a Bayesian extension that recasts unlearning as a principled statistical inference problem.

## 1. The PAC Unlearning Framework

The Probably Approximately Correct (PAC) Unlearning framework provides a theoretical lens to evaluate the performance of any unlearning algorithm. It defines two key metrics that capture the dual objectives of unlearning: forgetting the specified data while retaining overall model utility.

### 1.1. Core Concepts

The framework operates on a distributional level. We consider a machine learning model as representing a probability distribution over an instance space `X`.

*   **Instance Space `X`**: The space of all possible data points.
*   **Target Concept `h`**: A function `h: X -> {0, 1}` that partitions the data space into two regions.
    *   **Forget Region `X₁`**: The set of data points where `h(x) = 1`. This corresponds to the data or concept to be unlearned.
    *   **Retain Region `X₀`**: The set of data points where `h(x) = 0`. This corresponds to the data that should be remembered.
*   **Ideal Retain Distribution `P₀`**: The "ground truth" distribution over the retain region `X₀`. This represents the ideal state of the model after unlearning—a model trained only on the retained data.
*   **Output Distribution `Q`**: The distribution produced by an unlearning algorithm.

### 1.2. The PAC Conditions

The quality of the output distribution `Q` is measured by two conditions, parameterized by `ε ≥ 0` and `λ ≥ 0`.

#### **1.2.1. The `ε`-Matching Condition (`ε-MC`)**

This condition measures **model utility**. It requires that the output distribution `Q` remains statistically close to the ideal retain distribution `P₀`.

> **Definition (`ε-MC`):** A distribution `Q` satisfies the `ε`-Matching Condition if:
>
> `d(Q, P₀) ≤ ε`

where `d(·, ·)` is a statistical distance measure between distributions. The standard choice is the **Total Variation (TV) distance**:

`d_TV(Q, P₀) = sup_{A ⊆ X} |Q(A) - P₀(A)|`

A small `ε` indicates high utility, as the unlearned model is statistically similar to a model retrained from scratch on the retained data.

#### **1.2.2. The `λ`-Unlearning Condition (`λ-UC`)**

This condition measures **forget quality**. It requires that the output distribution `Q` assigns negligible probability mass to the forget region `X₁`.

> **Definition (`λ-UC`):** A distribution `Q` satisfies the `λ`-Unlearning Condition if:
>
> `Q(X₁) ≤ λ`

where `Q(X₁) = ∫_{x ∈ X₁} Q(x) dx`. A small `λ` indicates successful forgetting, as the model is unlikely to generate or positively classify data from the forgotten concept.

### 1.3. The Fundamental Implication: Utility Implies Forgetting

A key property of this framework is that the `ε-MC` directly implies a bound on the `λ-UC`.

**Theorem:** If a distribution `Q` satisfies the `ε-Matching Condition` with respect to the Total Variation distance, i.e., `d_TV(Q, P₀) ≤ ε`, then it also satisfies the `λ-Unlearning Condition` for `λ = ε`.

**Proof:**
1.  By definition, `d_TV(Q, P₀) = sup_A |Q(A) - P₀(A)| ≤ ε`.
2.  This inequality must hold for the specific event `A = X₁`, so `|Q(X₁) - P₀(X₁)| ≤ ε`.
3.  The ideal retain distribution `P₀` has no support on the forget region, thus `P₀(X₁) = 0`.
4.  Substituting this gives `|Q(X₁) - 0| ≤ ε`, which simplifies to `Q(X₁) ≤ ε`.
5.  Therefore, we satisfy the `λ-UC` with `λ = ε`.

This result establishes that an unlearning algorithm's forgetting quality is fundamentally limited by its utility.

## 2. The Bayesian PAC Unlearning Framework

This framework recasts unlearning from an algorithmic procedure to a problem of Bayesian statistical inference. The goal is to infer a posterior distribution over possible retain distributions, using the retain data as positive evidence and the forget data as negative evidence.

### 2.1. The Statistical Model

We perform inference over the space of probability distributions `P` on `X`.

*   **Prior Distribution `π(P)`**: Our belief about the retain distribution `P` *before* observing any data. A flexible choice is a **Dirichlet Process**, `P ~ DP(α, H)`, where `H` is a base measure (prior guess) and `α` is a concentration parameter.
*   **Observed Data**: We are given two sets of samples:
    *   `D₀ = {x₀,₁, ..., x₀,ₙ}`: Samples from the retain region.
    *   `D₁ = {x₁,₁, ..., x₁,ₘ}`: Samples from the forget region.

*   **Likelihood Function `L(D₀, D₁ | P)`**: This is the core of the model. It combines positive and negative evidence.
    *   **Retain Likelihood `L(D₀ | P)`**: The standard likelihood term that rewards distributions for explaining the retain data well.
        `L(D₀ | P) = Π_{x ∈ D₀} P(x)`
    *   **Forget Likelihood `L(D₁ | P)`**: A "tilted" likelihood term that penalizes distributions for explaining the forget data.
        `L(D₁ | P) ∝ [Π_{x ∈ D₁} P(x)]^(-λ_hyper)`
        Here, `λ_hyper ≥ 0` is a hyperparameter controlling the **strength of forgetting**.

*   **Posterior Distribution `π(P | D₀, D₁)`**: Using Bayes' rule, our posterior belief over retain distributions is:

    > `π(P | D₀, D₁) ∝ L(D₀ | P) * L(D₁ | P) * π(P)`
    >
    > `π(P | D₀, D₁) ∝ [Π_{x ∈ D₀} P(x)] * [Π_{x ∈ D₁} P(x)]^(-λ_hyper) * π(P)`

This posterior distribution is the "unlearned model." It captures our full uncertainty about the true retain distribution after considering all evidence.

### 2.2. Recovering the PAC Conditions

To connect back to the classical framework, we derive a single point-estimate distribution `Q` from our posterior and analyze it.

*   **Output Distribution `Q_Bayes`**: The standard choice is the **posterior predictive distribution**, which is the posterior mean of `P`.
    `Q_Bayes(x_new) = E_{P ~ π(P | D₀, D₁)} [P(x_new)] = ∫ P(x_new) π(P | D₀, D₁) dP`

#### **2.2.1. The Bayesian `ε`-Matching Condition**

The utility of the Bayesian model is captured by the **concentration of the posterior**. As the amount of retain data `|D₀|` increases, the posterior distribution `π(P | D₀, D₁)` will concentrate around the true data-generating distribution `P₀`. Consequently, the distance `d(Q_Bayes, P₀)` will decrease. The `ε` can be formally bounded using PAC-Bayesian theory, and it will be a function of `|D₀|`, the complexity of the prior, and the failure probability `δ`.

#### **2.2.2. The Bayesian `λ`-Unlearning Condition**

The forgetting quality is explicitly controlled and directly calculated.

`λ = Q_Bayes(X₁) = E_{P ~ π(P | D₀, D₁)} [P(X₁)]`

This is the **posterior expected probability mass in the forget region**. The forget likelihood term `L(D₁ | P)` was constructed precisely to suppress the posterior mass of any distribution `P` that has a large `P(X₁)`. The `λ` of the final model is therefore directly and monotonically controlled by the `λ_hyper` hyperparameter we set in the model. A larger `λ_hyper` leads to a smaller `λ`.