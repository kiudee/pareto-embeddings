"""Script responsible for running the actual experiment.

The plan is to fetch jobs from a database at some point.
Thus this script should be able to accept all necessary parameters from cli.

To save time this runner implements learner/dataset specific code.
"""
import json
import logging
import os
import pathlib
import socket
import time
import uuid

import click
import dill
import numpy as np
import sqlalchemy
import torch
from bask import BayesSearchCV
from csrank import (
    FATEChoiceFunction,
    FATELinearChoiceFunction,
    FETAChoiceFunction,
    FETALinearChoiceFunction,
    PairwiseSVMChoiceFunction,
    RankNetChoiceFunction,
    GeneralizedLinearModel,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline

from pareto.data import *
from pareto.exceptions import NoJobException, MissingGPUException
from pareto.metrics_np import *
from pareto.models import *
from pareto.preprocessing import StandardScaler3D
from pareto.sql import Base, Job, JobStatus, Result, get_managed_session_maker
from pareto.util import (
    all_pareto_fronts,
    get_random_state,
    is_score_prediction,
    parse_ranges,
)

DATASETS = {
    "two_parabola": TwoParabola,
    "DTLZ": DTLZ,
    "ZDT": ZDT,
}

TORCH_LEARNERS = {
    "ParetoSimple": ParetoEmbedder,
    "ParetoSimpleSGD": ParetoEmbedderSGD,
    "ParetoSimpleSGDSoft": ParetoEmbedderSGDSoft,
    "ParetoFATE": FATEParetoEmbedder,
    "PWParetoSGD": PWParetoEmbedder,
    "PWParetoSGDSoft": PWParetoEmbedderSoft,
}

TF_LEARNERS = {
    "FATEChoiceFunction": FATEChoiceFunction,
    "FATELinearChoiceFunction": FATELinearChoiceFunction,
    "FETAChoiceFunction": FETAChoiceFunction,
    "FETALinearChoiceFunction": FETALinearChoiceFunction,
    "PairwiseSVMChoiceFunction": PairwiseSVMChoiceFunction,
    "RankNetChoiceFunction": RankNetChoiceFunction,
}

CPU_LEARNERS = {
    "GLM": GeneralizedLinearModel,
    "PairwiseSVMChoiceFunction": PairwiseSVMChoiceFunction,
}

METRICS = {
    "informedness": instance_informedness_np,
    "amean": a_mean,
    "f1": f1_measure,
    "precision": precision,
    "recall": recall,
    "subset01": subset_01_loss,
    "hamming": hamming,
}


@click.command()
@click.option("--db-uri", default=None, help="sql alchemy db uri")
@click.option(
    "--model-path", default=None, help="folder to which the trained model is dumped."
)
@click.option("--max-tries", default=3, help="maximum tries to fetch a job")
@click.option(
    "--wait-time", default=60 * 3, help="wait x seconds after not finding a job"
)
@click.option("-v", "--verbose", count=True, default=0)
def fetch_job(
    db_uri=None, model_path=None, max_tries=3, wait_time=60 * 3, verbose=0,
):
    if verbose > 1:
        level = logging.DEBUG
    elif verbose > 0:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        format="%(asctime)s {} {} %(message)s".format(
            os.getenv("CCS_REQID", None), socket.gethostname()
        ),
        level=level,
    )

    if db_uri is not None:
        engine = sqlalchemy.create_engine(db_uri)
        Base.metadata.create_all(engine)
        Base.metadata.bind = engine
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=engine)
        session_maker = get_managed_session_maker(SessionMaker)
    else:
        raise ValueError("You need to provide a valid db-uri.")
    if model_path is None:
        raise ValueError(
            "You need to specify a folder path for saving the final model."
        )
    else:
        try:
            pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
        except Exception:
            logging.error(
                "The given model path could not be created. Make sure it is correct: {}".format(
                    model_path
                )
            )
            raise

    # 1. (Repeatedly) poll database for available jobs, pick a random one and try to allocate it:
    cuda_index = int(os.getenv("CUDA_VISIBLE_DEVICES", "1024"))
    logging.info(
        "CUDA_VISIBLE_DEVICES={} CCS_REQID={}".format(
            cuda_index, os.getenv("CCS_REQID", None)
        )
    )

    tries = 0
    while tries < max_tries:
        try:
            with session_maker() as session:
                if cuda_index == 1024:  # CPU job
                    job = (
                        session.query(Job)
                        .with_for_update(skip_locked=True)
                        .filter(Job.status == JobStatus.QUEUED)
                        .filter(Job.learner.in_(list(CPU_LEARNERS.keys())))
                        .order_by(Job.id)
                        .first()
                    )  # noqa
                else:  # GPU job
                    job = (
                        session.query(Job)
                        .with_for_update(skip_locked=True)
                        .filter(Job.status == JobStatus.QUEUED)
                        .filter(~Job.learner.in_(list(CPU_LEARNERS.keys())))
                        .order_by(Job.id)
                        .first()
                    )  # noqa
                job.status = JobStatus.ALLOCATED
                job.start_time = int(time.time())
                job.host_name = socket.gethostname()
                job.req_id = os.getenv("CCS_REQID", None)
                job_dict = job.to_dict()
        except Exception:
            tries += 1
            time.sleep(wait_time)
            continue
        break
    if tries == max_tries:
        raise NoJobException(
            "No job was found in the database after {} tries. Aborting.".format(tries)
        )

    # 2. We have a job now. Run the experiment:
    try:
        result_dict = _run_experiment(**job_dict)
    except Exception:
        with session_maker() as session:
            job.status = JobStatus.ERRORED
            job.end_time = int(time.time())
            _ = session.merge(job)
        raise

    # 3. The job was successful. Update job and create result:
    logging.info("Job was successful, trying to save results.")
    for i in range(5):
        try:
            with session_maker() as session:
                job = session.merge(job)
                job.end_time = int(time.time())
                job.status = JobStatus.FINISHED

                model_uuid = uuid.uuid4()

                result = Result(
                    metrics=result_dict["metrics"],
                    model_uuid=model_uuid,
                    job=job,
                    best_params=result_dict["model"].best_params_,
                )
                session.add(result)
        except Exception as e:
            logging.exception("Saving results was not successful.")
            if i < 4:
                time.sleep(60)
                logging.info("Trying again to save results.")
            else:
                logging.exception("Giving up saving results.")
                with session_maker() as session:
                    job.status = JobStatus.ERRORED
                    job.end_time = int(time.time())
                    _ = session.merge(job)
        else:
            break

    logging.info("Finally saving model now.")
    try:
        with open(
            file=pathlib.Path(model_path) / str(model_uuid), mode="wb"
        ) as file:
            dill.dump(
                result_dict["model"], file=file,
            )
    except Exception as e:
        logging.warning("Unable to save model: {}".format(e))


@click.command()
@click.option("--dataset", help="name of the dataset (generator)")
@click.option("--learner", help="name of the learning algorithm")
@click.option("--fold-id", help="which iteration of the outer cv to run")
@click.option("--outer-cv-folds", default=5, help="number of outer ShuffleSplit folds")
@click.option("--outer-cv-test", default=0.1, help="% of instances used for testing")
@click.option("--inner-cv-folds", default=1, help="number of inner ShuffleSplit folds")
@click.option("--inner-cv-val", default=0.2, help="% of instances used for validation")
@click.option("--dataset-opts", default=None, help="options for the dataset class")
@click.option("--learner-opts", default=None, help="options for the learner class")
@click.option("--hyperopt-opts", default=None, help="options for the learner class")
@click.option(
    "--search-space", default=None, help="search space for hyperparameter opt"
)
@click.option(
    "--metrics",
    default=None,
    help="metrics to evaluate the final model on (default informedness)",
)
@click.option("--random-seed", default=0, help="seed for the random number generator")
@click.option("-v", "--verbose", count=True, default=0)
def run_experiment(
    dataset,
    learner,
    fold_id,
    outer_cv_folds=5,
    outer_cv_test=0.1,
    inner_cv_folds=1,
    inner_cv_val=0.2,
    dataset_opts=None,
    learner_opts=None,
    hyperopt_opts=None,
    search_space=None,
    metrics=None,
    random_seed=0,
    verbose=0,
    db_uri=None,
):
    if verbose > 1:
        level = logging.DEBUG
    elif verbose > 0:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(format="%(asctime)s %(message)s", level=level)
    return _run_experiment(
        dataset=dataset,
        learner=learner,
        fold_id=fold_id,
        outer_cv_folds=outer_cv_folds,
        outer_cv_test=outer_cv_test,
        inner_cv_folds=inner_cv_folds,
        inner_cv_val=inner_cv_val,
        dataset_opts=dataset_opts,
        learner_opts=learner_opts,
        hyperopt_opts=hyperopt_opts,
        search_space=search_space,
        metrics=metrics,
        random_seed=random_seed,
    )


def _json_to_dict(j):
    if j is None:
        return dict()
    if isinstance(j, dict):
        return j
    return json.loads(j)


def _run_experiment(
    dataset,
    learner,
    fold_id,
    outer_cv_folds=5,
    outer_cv_test=0.1,
    inner_cv_folds=1,
    inner_cv_val=0.2,
    dataset_opts=None,
    learner_opts=None,
    hyperopt_opts=None,
    search_space=None,
    metrics=None,
    random_seed=0,
):

    logging.debug(f"Starting experiment with fold_id={fold_id}")
    # Note, that this is the new np.random.Generator:
    rand_seq = np.random.SeedSequence(random_seed)

    # * Load data / Generate data
    if dataset not in DATASETS:
        raise ValueError(f"There is no dataset with the name {dataset}")
    if not (
        learner in TORCH_LEARNERS or learner in TF_LEARNERS or learner in CPU_LEARNERS
    ):
        raise ValueError(f"There is no learner with the name {learner}")
    dataset_opts = _json_to_dict(dataset_opts)
    learner_opts = _json_to_dict(learner_opts)
    hyperopt_opts = _json_to_dict(hyperopt_opts)
    logging.info(f"Creating dataset {dataset} ...")
    ds = DATASETS[dataset](
        random_state=np.random.default_rng(rand_seq.spawn(1)[0]), **dataset_opts
    )
    X, Y = ds.get_xy()
    logging.debug(f"Data:\n{X}\n{Y}")

    # Execute outer cross validation loop
    cv = ShuffleSplit(
        n_splits=outer_cv_folds,
        test_size=float(outer_cv_test),
        random_state=get_random_state(rand_seq.spawn(1)[0]),
    )
    split = list(cv.split(X))
    logging.debug(f"Calculated outer CV split: {split}")
    logging.debug("Entering outer CV loop.")
    for i, (tra, tes) in enumerate(split):
        logging.debug(f"Entering outer loop {i}.")
        # We are only doing something, when the fold_id matches:
        if i == int(fold_id):
            logging.debug(f"Executing outer fold {i}. Initializing learner.")
            # TODO: Handle legacy/tensorflow learners here
            cuda_index = int(os.getenv("CUDA_VISIBLE_DEVICES", "1024"))
            if cuda_index == 1024:
                if learner not in CPU_LEARNERS:
                    # Raise exception
                    raise MissingGPUException(
                        f"Learner requires a GPU, but node "
                        f"{socket.gethostname()} does not have "
                        f"one."
                    )
            if learner in TF_LEARNERS:
                lrn = TF_LEARNERS[learner](**learner_opts)
            elif learner in TORCH_LEARNERS:
                # TODO: Test if this improves performance:
                torch.backends.cudnn.benchmark = True
                lrn = TORCH_LEARNERS[learner](device="cuda", **learner_opts)
            else:  # CPU learner:
                lrn = CPU_LEARNERS[learner](**learner_opts)

            pipe = Pipeline(
                steps=[("standardize", StandardScaler3D()), ("learner", lrn)]
            )
            logging.debug(lrn)
            # * Run hyper parameter optimization on (x_train, y_train)
            # TODO: Handle case where no hyperparameters are optimized
            logging.debug("Parsing hyperparameter ranges.")

            pipe_search_space = {"learner__" + k: v for k, v in search_space.items()}
            ss = parse_ranges(pipe_search_space)
            hyperopt_opts["random_state"] = get_random_state(rand_seq.spawn(1)[0])
            inner_cv = ShuffleSplit(
                n_splits=inner_cv_folds,
                test_size=float(inner_cv_val),
                random_state=get_random_state(rand_seq.spawn(1)[0]),
            )
            logging.debug("Initializing optimizer.")
            # TODO: Support passing in scoring function:
            scorer = make_scorer(score_func=a_mean)
            opt = BayesSearchCV(
                pipe, search_spaces=ss, cv=inner_cv, scoring=scorer, **hyperopt_opts
            )
            logging.debug("Running hyperparameter optimization.")
            _ = opt.fit(X[tra], Y[tra])
            logging.debug("Hyperparameter optimization finished.")
            logging.debug(
                "Will use the following set of hyperparameters: {}".format(
                    opt.best_params_
                )
            )

            Y_pred = opt.predict(X[tes])
            if is_score_prediction(Y_pred):
                Y_pred = all_pareto_fronts(Y_pred)
            logging.debug(f"Predictions: {Y_pred}")
            met_funcs = [METRICS[x] for x in metrics]
            met_results = []
            for met_func in met_funcs:
                met_results.append(met_func(Y[tes], Y_pred))
            results = dict(zip(metrics, met_results))
            logging.debug(f"Results: {results}")
            # TODO: FIX or Remove:
            # if device == "cuda":
            #     logging.info(
            #         "Maximum memory allocated: {}".format(
            #             torch.cuda.max_memory_allocated()
            #         )
            #     )

            result_dict = {"metrics": results, "model": opt}
            return result_dict
    logging.warning(
        "Outer CV loop exited without running anything. This should not happen!"
    )
