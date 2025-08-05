import logging
from typing import Any, Dict

from harddisc.drivers.dloptimizationdriver import run_dloptimization
from harddisc.drivers.dltraindriver import dltrain
from harddisc.drivers.encodedriver import encodedriver
from harddisc.drivers.generativedriver import generativedriver
from harddisc.drivers.mloptimizationdriver import run_mloptimization
from harddisc.drivers.mltraindriver import mltrain
from harddisc.drivers.topicmodeldriver import run_topicmodelling

logger = logging.getLogger(__name__)


def pomelo_driver(config: Dict[str, Any]) -> None:
    """Main runner for POMELO program

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed configuration file for entire POMELO run
    """
    # run each stage if it is listed
    stages = config["pipelinesetup"]["stages"]

    driver_dict = {
        "encoding": encodedriver,
        "mltrain": mltrain,
        "dltrain": dltrain,
        "topicmodel": run_topicmodelling,
        "mloptimization": run_mloptimization,
        "dloptimization": run_dloptimization,
        "generative": generativedriver,
    }

    logger.info(f"Running POMELO with {len(stages)} stages")

    for stage in stages:
        logger.info(f"Beginning POMELO stage: {stage}")
        driver_dict[stage](config)
        logger.info(f"Finished POMELO stage: {stage}")
