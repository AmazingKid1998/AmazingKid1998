<div align="center">

# Jamilus Sheium
**Machine Learning Researcher & Neural Architect**  
*Mechanistic LLM Safety • Custom System-1 Foundation Models • Adversarial & Privacy-Preserving ML*

[![Role](https://img.shields.io/badge/Horizon_Europe-Lead_ML_RA_(XTRUST--6G)-0F172A?style=flat-square&labelColor=1E293B)](https://cordis.europa.eu/)
[![Venture](https://img.shields.io/badge/Voxlith-Co--Founder_%26_ML_Lead-0F172A?style=flat-square&labelColor=1E293B)](https://voxlith.com)
[![Project](https://img.shields.io/badge/UKRI_CyberASAP-PromptGuard_ML_Lead-0F172A?style=flat-square&labelColor=1E293B)](https://promptguard-cyberasap.github.io)
[![Standing](https://img.shields.io/badge/Academic_Standing-First_Class_(85%25_%7C_4.0_GPA)-1D4ED8?style=flat-square&labelColor=1E293B)](#)

[![Email](https://img.shields.io/badge/Email-jamilus.sheium1%40port.ac.uk-334155?style=flat-square&logo=minutemailer&logoColor=white)](mailto:jamilus.sheium1@port.ac.uk)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jamilus--sheium-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jamilus-sheium)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--8926--5370-A6CE39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0009-0003-8926-5370)
[![CV](https://img.shields.io/badge/Curriculum_Vitae-Download_PDF-DC2626?style=flat-square&logo=adobeacrobatreader&logoColor=white)](Jamilus_Sheium_CV.pdf)

</div>

---

### Research Architecture Overview

My work bridges **first-principles deep learning theory** (latent subspace geometry, quantization error propagation, and constrained continual optimization) with **high-throughput adversarial security and privacy systems**. Across three externally funded European (**Horizon Europe**) and UK (**UKRI CyberASAP**) grants, I serve as the lead mathematical and machine-learning architect.

```mermaid
graph LR
    subgraph Foundations["1. Mathematical Foundations"]
        A1["Low-Rank Subspace Geometry<br/>& Refusal Directions"]
        A2["Loss-Landscape Curvature<br/>& Quantisation Mechanics"]
        A3["Constrained Continual Optimization<br/>& Non-IID Federated Gradients"]
    end

    subgraph Architectures["2. Novel Neural Architectures"]
        B1["Refusal-Guided Mixed-Precision<br/>PTQ Allocator (4-bit LLMs)"]
        B2["Custom Cyber Foundation Model<br/>+ System-1 TCN & RLCD Head"]
        B3["Provenance-Constrained Update Gate<br/>(Self-Improving LoRA Agents)"]
    end

    subgraph Deployments["3. Real-World Security & Privacy Systems"]
        C1["Voxlith Spin-Out<br/>(2M+ Multi-Modal OOD Threat Engine)"]
        C2["XTRUST-6G Consortium<br/>(Federated & Encrypted-Flow NIDS)"]
        C3["PromptGuard Client-Side DLP<br/>(Quantised Edge SLMs)"]
    end

    A1 --> B1
    A2 --> B1
    A1 --> B3
    A3 --> B2
    A3 --> B3
    B1 --> C3
    B2 --> C1
    B2 --> C2
    B3 --> C1
```

---

### Interactive Technical Deep-Dives
*Click any research pillar below to expand the underlying mathematical formulation, neural architecture diagram, and empirical evaluation methodology.*

<details open>
<summary><b>Pillar I: Custom Cyber Foundation Model, System-1 Decision Head & RLCD (Voxlith / UKRI Locknest)</b></summary>
<br/>

Adversarial web and network threat detection violates standard IID assumptions and operates under strict sub-millisecond latency constraints where slow autoregressive reasoning fails.

```mermaid
flowchart LR
    In["Multi-Modal Input<br/>(DOM + URL + Headers + Runtime)"] --> FM["Custom Pre-Trained<br/>Cyber Foundation Backbone"]
    FM --> TCN["Dilated Causal TCN<br/>(Exponential Receptive Field)"]
    FM --> JE["Joint-Embedding<br/>Latent Projection"]
    TCN --> Sys1["Fast System-1<br/>Decision Head"]
    JE --> Sys1
    Sys1 --> RLCD["RLCD Policy Boundary<br/>(Asymmetric Cost Optimization)"]
    RLCD --> Out["Zero-Day Containment<br/>(<1ms Inference | 96% OOD Acc)"]
```

* **Multi-Modal Pre-Training:** Pre-trained from scratch over a metric-balanced, heterogeneous corpus exceeding **2,000,000 samples** (structural DOM trees, URLs, HTTP headers, and live runtime execution traces) to learn invariant threat manifolds.
* **System-1 Joint-Embedding & Dilated TCNs:** Coupled the foundation backbone with dilated causal **Temporal Convolutional Networks (TCNs)** and a fast **System-1 joint-embedding decision head** to capture long-range sequential dependencies in a single forward pass.
* **RLCD Post-Training:** Formulated a **Reinforcement Learning from Contrastive/Contextual Decisions (RLCD)** objective to optimize the decision boundary under asymmetric false-positive/false-negative operational costs.
* **Kit-Disjoint OOD Partitioning:** Proved that standard IID random train/test splits leak shared phishing-kit templates across partitions; formulated a **kit-family-disjoint splitting protocol** to evaluate true out-of-distribution (OOD) generalisation.

</details>

<details>
<summary><b>Pillar II: Low-Rank Refusal Subspaces & Mixed-Precision PTQ Mechanics (First-Author Paper)</b></summary>
<br/>

Standard Post-Training Quantisation (PTQ) minimizes layer-wise reconstruction error under a calibration distribution, treating all latent directions as equally critical:

$$\min_{\hat{W}_\ell} \left\Vert{} W_\ell X_\ell - \hat{W}_\ell X_\ell \right\Vert{}_F^2 \quad \text{where} \quad \Delta W_\ell = \hat{W}_\ell - W_\ell$$

* **Mechanistic Failure Mode:** In *"When Does Post-Training Quantisation Harm LLM Safety?"* (*Submitted to IEEE Access*), I demonstrate that uniform 4-bit discretisation noise $\Delta W_\ell$ disproportionately perturbs low-rank **causal refusal subspaces** ($\mathbf{v}_{\text{refusal}} \in \mathbb{R}^{d_{\text{model}}}$) even when general perplexity remains stable.
* **Refusal-Direction-Guided Mixed-Precision Allocator:** Formulated a layer-wise sensitivity metric measuring projection drift along the extracted refusal direction:

$$\mathcal{S}_\ell(\mathbf{v}_{\text{refusal}}) = \mathbb{E}_{x \sim \mathcal{D}_{\text{harm}}} \left[ \left\vert{} \left\langle \Delta h_\ell(x), \, \hat{\mathbf{v}}_{\text{refusal}} \right\rangle \right\vert{} \right]$$

  Layers exceeding the critical subspace perturbation threshold are selectively promoted in precision under a fixed memory budget, recovering 16-bit safety reliability at 4-bit inference speeds.
* **Variance Decomposition:** Pre-registered, multi-seed evaluation across 3 instruction-tuned model families, 4 PTQ recipes, and 3 languages, isolating measurement confounders in prior safety benchmarks.

</details>

<details>
<summary><b>Pillar III: Corrigibility & Update Gating in Recursive Self-Improving LLM Agents (Honours Thesis)</b></summary>
<br/>

When an autonomous LLM agent iteratively modifies its own scaffold, memory policy, or low-rank adapters ($\theta_{t+1} = \theta_t + B_t A_t$) across generations $t \in \{1, \dots, T\}$ to maximize task utility $\mathcal{U}(\theta_t)$, capability gains frequently induce latent alignment drift and evaluator reward hacking.

```mermaid
flowchart LR
    Agent["Agent State (θ_t)"] --> Prop["Proposed Self-Edit<br/>(Prompt / Tool / LoRA Δθ_t)"]
    Prop --> Gate{"External Provenance &<br/>Corrigibility Gate"}
    Gate -- "ΔU > 0 AND D_KL ≤ ε<br/>AND Corrigibility Retained" --> Accept["Accept Update (θ_{t+1})"]
    Gate -- "Drift / Reward Hacking<br/>or Provenance Violation" --> Reject["Reject & Rollback (θ_t)"]
```

* **Constrained Trajectory Optimization:** Formulating recursive self-improvement as a constrained optimization problem where an update $\Delta \theta_t$ is accepted if and only if:

$$\mathcal{U}(\theta_t + \Delta \theta_t) > \mathcal{U}(\theta_t) \quad \text{subject to} \quad \mathcal{C}_{\text{corr}}(\theta_t + \Delta \theta_t) \ge \tau_{\text{corr}} \;\land\; D_{\text{KL}}\!\left(\pi_{\theta_{t+1}} \,\Vert{}\, \pi_{\text{safe}}\right) \le \epsilon$$

* **Early Warning Indicators:** Testing whether static behavioral safety scores miss earlier structural deterioration in **corrective leverage** (the agent's responsiveness to human override and policy correction).

</details>

<details>
<summary><b>Pillar IV: Non-IID Federated Optimization, Encrypted-Flow ML & Semantic DLP (XTRUST-6G & PromptGuard)</b></summary>
<br/>

* **Non-IID Federated Intrusion Detection (*XTRUST-6G — €8M Horizon Europe*):** Designing collaborative federated optimization protocols across 18+ European partner institutions, mitigating client gradient drift over skewed, non-IID network telemetry without raw traffic exchange.
* **Encrypted-Traffic Sequence Representation (*XTRUST-6G*):** Architecting deep temporal sequence models that identify malicious sessions directly from packet-timing inter-arrival manifolds and directional burst dynamics without payload decryption.
* **Client-Side Semantic DLP (*PromptGuard — UKRI CyberASAP*):** Formulated sensitive disclosure detection as a contextual entailment and latent representation classification task using quantised local SLMs (`vLLM` / `Ollama`), outperforming brittle regex/NER baselines on implicit confidential leaks.

</details>

---

### Funded Grants, Spin-Outs & Publications Matrix

| Initiative / Venue | Role & Funding Body | Core Mathematical & Systems Contribution | Status / Artifact |
| :--- | :--- | :--- | :--- |
| **IEEE Access** *(Manuscript)* | **First Author** *(with Dr. M. Ghahremani & Prof. M. Bader-El-Den)* | Pre-registered 4-bit PTQ safety characterisation & refusal-direction mixed-precision allocator | **Under Review** |
| **XTRUST-6G** | **Lead ML RA** *(€8M Horizon Europe / SNS JU)* | Deep zero-day sequence anomaly models, non-IID Federated Learning & encrypted-flow ML | **Active (2026–Present)** |
| **Voxlith (Locknest)** | **Co-Founder & ML Lead** *(UKRI CyberASAP Spin-Out)* | Custom Cyber Foundation Model (2M+ corpus), System-1 TCN/RLCD head & kit-disjoint OOD splits | **[voxlith.com](https://voxlith.com)** *(2 Live Pilots)* |
| **PromptGuard** | **Lead ML RA** *(UKRI CyberASAP)* | Quantised edge SLMs for semantic DLP, synthetic benchmark construction & evaluation harness | **[Project Site](https://promptguard-cyberasap.github.io)** |
| **Honours Thesis** | **Sole Investigator** *(University of Portsmouth)* | Corrigibility retention & provenance-constrained update gating in recursive self-improving agents | **In Progress (2026–2027)** |

> **Note on Proprietary Codebases & NDAs:** Source code, custom pre-training pipelines, and production weights for **XTRUST-6G**, **Voxlith**, and **PromptGuard** are governed by Horizon Europe consortium agreements, UKRI CyberASAP IP protections, and commercial NDAs. Reproducibility harnesses for academic publications are released upon peer-review clearance.

---

### Technical & Compute Stack

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace_Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![TensorFlow](https://img.shields.io/badge/TensorFlow_%2F_Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![CUDA](https://img.shields.io/badge/NVIDIA_CUDA_%26_L40S-76B900?style=flat-square&logo=nvidia&logoColor=white)
![vLLM](https://img.shields.io/badge/vLLM_%26_Ollama-0F172A?style=flat-square)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![C](https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=c&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL_(Multi--Million_Row)-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Linux](https://img.shields.io/badge/Linux_HPC-FCC624?style=flat-square&logo=linux&logoColor=black)

</div>

---

### Selected Public & Earlier Engineering Repositories

<table>
  <tr>
    <td width="50%" valign="top">
      <h4><a href="https://github.com/AmazingKid1998/buet-ces-website">BUET Centre for Energy Studies Platform</a></h4>
      <p>Full-stack conference paper submission and academic portal engineered for the Centre for Energy Studies at the Bangladesh University of Engineering and Technology (BUET).</p>
      <p><a href="http://iesd.buet.ac.bd/"><b>Live Institutional Deployment →</b></a></p>
    </td>
    <td width="50%" valign="top">
      <h4><a href="https://github.com/AmazingKid1998/process-scheduler">OS Process Scheduler & Metrics Engine</a></h4>
      <p>Algorithmic scheduling simulator and Gantt metric evaluation engine implementing First-Come-First-Served (FCFS), Shortest-Job-First (SJF), and Round-Robin (RR) policies.</p>
      <p><a href="https://github.com/AmazingKid1998/process-scheduler"><b>View Source Repository →</b></a></p>
    </td>
  </tr>
</table>
