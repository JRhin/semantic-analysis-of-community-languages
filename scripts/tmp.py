# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import polars as pl
from tqdm.auto import tqdm


def main() -> None:
    """The main loop.
    """
    CURRENT: Path = Path(".")
    DATA_PATH: Path = CURRENT / "data"
    USERS_PATH: Path = CURRENT / "users"

    # Create directory for the users
    USERS_PATH.mkdir(exist_ok=True)

    # Variables
    platform: str = "Reddit"
    n_comments: int = 100
    remove_authors: list[str] = ["AutoModerator", "[deleted]"]
    schema_overrides: dict[str, str] = {"created_utc": pl.Float64, "score": pl.Int64}
    filter = (pl.col("body")!="[deleted]")&(pl.col("body")!="[removed]")&(pl.col("score")>0)&(pl.col("author").is_in(remove_authors).not_())
    # ids: list[str] = pl.read_parquet("data/reddit_submissions.parquet", columns=["id"])["id"].to_list()

    for path in tqdm(list((DATA_PATH / platform ).rglob("*_comments.csv"))):
        # Get language and community
        _, _, language, community = str(path).split("/")
        community = "_".join(map(str, community.split(".")[0].split("_")[:-1]))

        # DO NOT CONSIDER POLITICS
        if community == "politics": continue

        df = (
            pl.scan_csv(path, schema_overrides=schema_overrides)
			.with_columns(text=pl.col("body"),
                          language=pl.lit(language),
                    	  platform=pl.lit(platform),
                    	  community=pl.lit(community),
                    	  created_utc=pl.from_epoch(pl.col("created_utc").cast(pl.Int64), time_unit="s"),
                    	  parent_id=pl.col("parent_id").str.split("_").list.get(-1))
			.drop_nulls()
			.filter(filter)
			.unique(subset=["id"])
			.rename({"created_utc": "date"})
			# .select(pl.col("id"), pl.col("text"), pl.col("language"), pl.col("platform"), pl.col("community"), pl.col("score"), pl.col("author"), pl.col("date"), pl.col("parent_id"))
			.select(pl.col("id"), pl.col("text"), pl.col("platform"), pl.col("community"), pl.col("author"), pl.col("date"))
			# .select(pl.col("id"), pl.col("author"), pl.col("date"))
			# .group_by(by="author").len()
        )

        # Get all the users ids that had commented in the community at least n_comments times
        users_ids = df.select(pl.col("author")).group_by(by="author").len().filter(pl.col("len")>=n_comments).select(pl.col("by")).collect().get_column("by").to_list()

        # For all the selected users
        for id in users_ids:
            # Define the path towards the parquet with all the comments of the selected user
            user_df_path = USERS_PATH / (id + ".parquet")

            # Check if it exists
            if user_df_path.exists():
                user_df = pl.read_parquet(user_df_path)
            else:
                user_df = pl.DataFrame()
                
            # Update the parquet with the new comments
            user_df.vstack(df.filter(pl.col("author")==id).collect()).sort(by="date", descending=False).unique(subset=["id"]).write_parquet(user_df_path)

            # Clean the memory
            del user_df

    return None


if __name__ == "__main__":
    main()
