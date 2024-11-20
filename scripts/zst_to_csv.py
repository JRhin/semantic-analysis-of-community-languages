# modified from https://github.com/Watchful1/PushshiftDumps/blob/master/scripts/to_csv.py

# this converts a zst file to csv
#
# it's important to note that the resulting file will likely be quite large
# and you probably won't be able to open it in excel or another csv reader
#
# arguments are inputfile, outputfile
# call this like
# python to_csv.py wallstreetbets_submissions.zst wallstreetbets_submissions.csv

# output fields are in the 'fields' variable in __main__

# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import zstandard
import os
import json
import sys
import csv
import polars as pl
import logging.handlers
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime


# log = logging.getLogger("bot")
# log.setLevel(logging.DEBUG)
# log.addHandler(logging.StreamHandler())


def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	chunk = reader.read(chunk_size)
	bytes_read += chunk_size
	if previous_chunk is not None:
		chunk = previous_chunk + chunk
	try:
		return chunk.decode()
	except UnicodeDecodeError:
		if bytes_read > max_window_size:
			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
	with open(file_name, 'rb') as file_handle:
		buffer = ''
		reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
		while True:
			chunk = read_and_decode(reader, 2**27, (2**29) * 2)
			if not chunk:
				break
			lines = (buffer + chunk).split("\n")

			for line in lines[:-1]:
				yield line, file_handle.tell()

			buffer = lines[-1]
		reader.close()


def single_file_decoder(input_file_path: str,
						output_file_path: str) -> None:
	"""This function decodes the single zst file into a single csv file.

	Args:
		input_file_path : str
			The path of the original zst file.
		output_file_path : str	
			The path of the resulting csv file.

	Returns:
		None
	"""
# if __name__ == "__main__":
	# input_file_path = sys.argv[1]
	# output_file_path = sys.argv[2]
	# fields = sys.argv[3].split(",")
	
	##### all possible fields
	# fields = ['all_awardings', 'allow_live_comments', 'archived', 'author', 'author_created_utc', 'author_flair_background_color', 'author_flair_css_class', 'author_flair_template_id', 'author_flair_text', 'author_flair_text_color', 'can_gild', 'category', 'content_categories', 'contest_mode', 'created_utc', 'discussion_type', 'distinguished', 'domain', 'edited', 'gilded', 'gildings', 'hidden', 'hide_score', 'id', 'is_created_from_ads_ui', 'is_crosspostable', 'is_meta', 'is_original_content', 'is_reddit_media_domain', 'is_robot_indexable', 'is_self', 'is_video', 'link_flair_background_color', 'link_flair_css_class', 'link_flair_richtext', 'link_flair_text', 'link_flair_text_color', 'link_flair_type', 'locked', 'media', 'media_embed', 'media_only', 'name', 'no_follow', 'num_comments', 'num_crossposts', 'over_18', 'parent_whitelist_status', 'permalink', 'pinned', 'pwls', 'quarantine', 'removed_by_category', 'retrieved_utc', 'score', 'secure_media', 'secure_media_embed', 'selftext', 'send_replies', 'spoiler', 'stickied', 'subreddit', 'subreddit_id', 'subreddit_subscribers', 'subreddit_type', 'suggested_sort', 'thumbnail', 'thumbnail_height', 'thumbnail_width', 'title', 'top_awarded_type', 'total_awards_received', 'treatment_tags', 'upvote_ratio', 'url', 'whitelist_status', 'wls']
	
	##### submissions dump
	# fields = ["id", "title", "selftext", "score", "created_utc"]
	fields = ["id", "body", "score", "created_utc", "parent_id", "author"]

	file_size = os.stat(input_file_path).st_size
	file_lines = 0
	file_bytes_processed = 0
	line = None
	created = None
	bad_lines = 0
	output_file = open(output_file_path, "w", encoding='utf-8', newline="")
	writer = csv.writer(output_file)
	writer.writerow(fields)
	try:
		for line, file_bytes_processed in read_lines_zst(input_file_path):
			try:
				obj = json.loads(line)
				output_obj = []
				for field in fields:
					output_obj.append(str(obj[field]).encode("utf-8", errors='replace').decode())
				writer.writerow(output_obj)

				created = datetime.utcfromtimestamp(int(obj['created_utc']))
			except json.JSONDecodeError as err:
				bad_lines += 1
			file_lines += 1
			# if file_lines % 100000 == 0:
			# 	log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
	except KeyError as err:
		# log.info(f"Object has no key: {err}")
		# log.info(line)
		print(err)
	except Exception as err:
		# log.info(err)
		# log.info(line)
		print(err)

	output_file.close()
	# log.info(f"Complete : {file_lines:,} : {bad_lines:,}")



def main():
	"""The main loop of the script.
	"""
	# Defining paths
	CURRENT: Path = Path('.')
	DATA: Path = CURRENT / "data/Reddit/"

	# Variables
	# column = "selftext"
	column = "body"
	remove_authors = ["AutoModerator", "[deleted]"]
	# ids = pl.read_parquet("data/reddit_submissions_60000.parquet", columns=["id"])["id"].to_list()
	ids = []

	for zst_file in tqdm(list(DATA.rglob('*_comments.zst'))):
		_, platform, language, community = str(zst_file.as_posix()).split("/")
		language = language.split("_")[0]
		community = "_".join(map(str, community.split(".")[0].split("_")[:-1]))

		# if community == "worldnews" or community == "news" or community == "politics": continue

		csv_file = str(zst_file).split(".")[0]+".csv"
		
		single_file_decoder(str(zst_file), csv_file)

		continue

		# Filter:
		# - remove deleted or removed posts
		# - consider only comments with score > 0
		# - consider only authors that are not in ["AutoModerator", "[deleted]"]
		filter = (pl.col(column)!="[deleted]")&(pl.col(column)!="[removed]")&(pl.col("score")>0)&(pl.col("author").is_in(remove_authors).not_())

		# Take comments from only specific submissions
		if len(ids) > 0:
			filter = filter &(pl.col("parent_id").str.split('_').list.get(-1).is_in(ids))

		df = (
			pl.scan_csv(csv_file, truncate_ragged_lines=True, schema_overrides={"created_utc": pl.Float64, "score": pl.Int64})
			.with_columns(#text=pl.col("title") + "\n" + pl.col(column),
			              text=pl.col(column),
                          language=pl.lit(language),
                    	  platform=pl.lit(platform),
                    	  community=pl.lit(community),
                    	  created_utc=pl.from_epoch(pl.col("created_utc").cast(pl.Int64), time_unit="s"),
                    	  parent_id=pl.col("parent_id").str.split("_").list.get(-1))
			.drop_nulls()
			.unique()
			.filter(filter)
			.rename({"created_utc": "date"})
			.select(pl.col("id"), pl.col("text"), pl.col("language"), pl.col("platform"), pl.col("community"), pl.col("score"), pl.col("author"), pl.col("date"), pl.col("parent_id"))
			# .group_by(["language", "community"])
			# .agg([
			#      	pl.all().sort_by("score", descending=True).head(100)
			#      ]).explode(["id", "text", "platform", "score", "date", "parent_id"])
			# .sink_csv("final__"+csv_file)
			
		)

		# print(df.head().collect())

	return None

		

if __name__ == "__main__":
	main()
