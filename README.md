# Energy Based Model trained with Noise Contrastive Estimation

This repository implements an **Energy-Based Model (EBM)** trained using **Noise Contrastive Estimation (NCE)**.  

## Project Structure
 * loss.py -> NCE loss implementations
 * model.py ->  Energy-Based Model (EBM) implementations
 * train.py ->  Training loop with early stopping & checkpointing
 * sampling.py ->  Langevin dynamics sampling for generation
 * plots.py ->  Evaluation and visualization utilities

## Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.0
* scikit-learn ≥ 1.2
* matplotlib ≥ 3.7

Install all dependencies via:

```bash
pip install torch torchvision scikit-learn matplotlib
```

---
## **loss.py**

Implements two NCE variants:

* `nce_loss`: log-sum-exp based formulation
* `nce_loss2`: logistic version using `logsigmoid`

The model discriminates between **real data** and **noise samples**:
$$
x \sim p_{\text{data}}, \quad \tilde{x} \sim p_n
$$

---

## **sampling.py**

Performs **Langevin Dynamics** in latent space to sample from the learned energy distribution:
$$
x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x f_\theta(x_t) + \sqrt{\epsilon},\eta_t, \quad
\eta_t \sim \mathcal{N}(0, I)
$$


---
## ⚠️ Notice

The **choice of the noise distribution** ( p_n(x) ) has a **strong impact on model performance**.
A well-chosen noise helps the model correctly estimate the energy boundaries between real and synthetic samples.
Conversely, a noise distribution that is too simple or too distant from the data distribution can lead to unstable training and poor generalization.

---

## 🧾 References

* Gutmann, M., & Hyvärinen, A. (2010).
  *Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics.*
* LeCun, Y., Chopra, S., & Hadsell, R. (2006).

---

👨‍💻 **Author:** Davide De Angelis
📍 *University of Naples “Parthenope” — Master’s Degree in Machine Learning & Big Data*




