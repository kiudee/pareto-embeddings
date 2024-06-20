# Pareto

Pareto is a Python library for Pareto-embeddings, which are used in subset choice modeling using machine learning. This repository contains the code and resources required to use and extend Pareto-embeddings in your own projects.


## Installation

### Prerequisites

- Python 3.7 or higher

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/kiudee/pareto.git
    cd pareto
    ```

2. Install the dependencies using Poetry:
    ```bash
    poetry install
    ```

3. (Optional) Install additional dependencies for PostgreSQL support:
    ```bash
    poetry install --extras "pgsql"
    ```

4. (Optional) Install additional dependencies for experimental features:
    ```bash
    poetry install --extras "exp"
    ```

## Usage

### Running Experiments

To run an experiment, use the provided `run_experiment` script:
```bash
poetry run run_experiment
```

### Fetching Jobs

To fetch a job, use the `fetch_job` script:
```bash
poetry run fetch_job
```

## Features

- **Pareto-embeddings**: Advanced embedding models for subset choice modeling.
- **Experiment scripts**: Easily run and fetch experiments with provided scripts.

## How to Cite Us

If you use Pareto in your research, please cite our paper:

```
@InProceedings{10.1007/978-3-030-58285-2_30,
author="Pfannschmidt, Karlson
and H{\"u}llermeier, Eyke",
editor="Schmid, Ute
and Kl{\"u}gl, Franziska
and Wolter, Diedrich",
title="Learning Choice Functions via Pareto-Embeddings",
booktitle="KI 2020: Advances in Artificial Intelligence",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="327--333",
abstract="We consider the problem of learning to choose from a given set of objects, where each object is represented by a feature vector. Traditional approaches in choice modelling are mainly based on learning a latent, real-valued utility function, thereby inducing a linear order on choice alternatives. While this approach is suitable for discrete (top-1) choices, it is not straightforward how to use it for subset choices. Instead of mapping choice alternatives to the real number line, we propose to embed them into a higher-dimensional utility space, in which we identify choice sets with Pareto-optimal points. To this end, we propose a learning algorithm that minimizes a differentiable loss function suitable for this task. We demonstrate the feasibility of learning a Pareto-embedding on a suite of benchmark datasets.",
isbn="978-3-030-58285-2"
}
```

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Contact Information

For any questions or feedback, please contact Karlson Pfannschmidt at [kiudee@mail.upb.de](mailto:kiudee@mail.upb.de).
