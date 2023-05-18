from . import assets
from dagster import (
    AssetSelection,
    Definitions,
    define_asset_job,
    load_assets_from_modules,
    FilesystemIOManager
)
from dotenv import load_dotenv

load_dotenv()


all_assets = load_assets_from_modules([assets])

stocks_etl_job = define_asset_job(
    "stocks_etl_job",
    selection=AssetSelection.all()
)

# Definitions for jobs and resources(io_manager)
defs = Definitions(
    assets=all_assets,
    jobs=[stocks_etl_job],
)
