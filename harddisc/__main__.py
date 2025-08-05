import argparse
import logging
from pathlib import Path

import yaml

from harddisc.validation.pomelo import POMELOConfig
from harddisc.drivers.pomelodriver import pomelo_driver

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Creates a parser for the harddisc driver

    Returns
    -------
    argparse.ArgumentParser
        Completed parser for harddisc
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=Path, help="path to config")

    return parser


def main() -> None:
    """Runs HARDDISC: POMELO

    Raises
    ------
    ValueError
        Provided config path does not exist
    """
    parser = create_parser()

    args = parser.parse_args()

    logger.info(f"Beginning POMELO run with {args.config}")

    if not args.config.exists():
        raise ValueError(f"Config path {args.config} does not exist")  # noqa: TRY003

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pomeloconfig = POMELOConfig(**config)

    pomelo_driver(pomeloconfig.dict())

    logger.info("POMELO finished")


if __name__ == "__main__":
    main()
