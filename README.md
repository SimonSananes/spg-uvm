
# Stochastic Policy Gradient Methods in the Uncertain Volatility Model

This repository accompanies the research paper **“Stochastic Policy Gradient Methods in the Uncertain Volatility Model”** ([arXiv:2605.06670](https://arxiv.org/abs/2605.06670)).

It provides a **self-contained Jupyter notebook** illustrating stochastic policy gradient (SPG) methods for robust option pricing and hedging under joint volatility and correlation uncertainty.

## Notebook

The notebook studies a **geometric outperformer option** with **2 to 5 underlying assets**. It implements:

- Backward actor-critic training using stochastic policies and Proximal Policy Optimization (PPO).
- A classical value-function critic and a separately trained Sobolev critic.
- Sobolev gradient targets obtained by centered finite differences at the terminal backward step, and by automatic differentiation at earlier steps.
- Independent Monte Carlo primal price estimates with 95% confidence intervals.
- Approximate pathwise dual evaluations based on the learned critic gradients.

The notebook includes all model, network, training, pricing, and dual-evaluation code. No local Python modules or external data files are required.

## Requirements

- Python with Jupyter Notebook or JupyterLab
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)

For example, install the dependencies with:

```bash
pip install torch matplotlib notebook
```

A CUDA-enabled GPU is recommended for the full experiments. The notebook automatically uses CUDA when available and otherwise runs on the CPU.

## Running the experiment

1. Open the `.ipynb` notebook in this repository.
2. In the **Experiment configuration** cell, set `d` to `2`, `3`, `4`, or `5`.
3. Run the cells in order.

The notebook trains the classical and Sobolev models independently, evaluates their primal prices, computes approximate dual estimates, and displays a comparison.

The default training and dual-evaluation settings can be computationally demanding, especially in higher dimensions. No random seed is fixed, so numerical results may vary between runs.

## Interpretation

The reported primal confidence intervals quantify Monte Carlo sampling uncertainty. The dual evaluations rely on a finite-grid, finite-sweep pathwise optimization and **are not certified upper bounds**. Their error bars do not account for optimization or discretization bias.

The notebook is an illustrative implementation of the method, not a claim that every numerical experiment in the paper can be reproduced from this single file.

## Citation

If you use this code in your research, please cite the associated
paper:

Abbas-Turki et al., *Stochastic Policy Gradient Methods in the
Uncertain Volatility Model*, arXiv:2605.06670 (2026).

Citation information is also available in
[CITATION.cff](CITATION.cff).


Citation information is also available in
[CITATION.cff](CITATION.cff).

## License

This code is distributed under the BSD 3-Clause License.
See the [LICENSE](LICENSE) file for details.

