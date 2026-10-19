import argparse
import csv
from collections import defaultdict
from pathlib import Path

from account_data.account_feature_extraction import calculate_all_features


def read_csv_rows(path):
	if not path.exists():
		return []
	with path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		return list(reader)


def group_by_author(rows):
	grouped = defaultdict(list)
	for row in rows:
		author = (row.get("author") or "").strip()
		if author:
			grouped[author].append(row)
	return grouped


def build_thread_indexes(thread_rows):
	by_root = defaultdict(list)
	by_comment_key = {}

	for row in thread_rows:
		root_author = (row.get("root_author") or "").strip()
		root_permlink = (row.get("root_permlink") or "").strip()
		if root_author and root_permlink:
			by_root[(root_author, root_permlink)].append(row)

		author = (row.get("author") or "").strip()
		permlink = (row.get("permlink") or "").strip()
		if author and permlink:
			by_comment_key[(author, permlink)] = row

	return by_root, by_comment_key


def get_author_thread_subset(author, posts, replies, thread_by_root, thread_by_comment_key):
	roots = set()

	for post in posts:
		post_author = (post.get("author") or "").strip()
		post_permlink = (post.get("permlink") or "").strip()
		if post_author == author and post_permlink:
			roots.add((post_author, post_permlink))

	for reply in replies:
		reply_author = (reply.get("author") or "").strip()
		reply_permlink = (reply.get("permlink") or "").strip()
		if reply_author != author or not reply_permlink:
			continue

		row = thread_by_comment_key.get((reply_author, reply_permlink))
		if row:
			root_author = (row.get("root_author") or "").strip()
			root_permlink = (row.get("root_permlink") or "").strip()
			if root_author and root_permlink:
				roots.add((root_author, root_permlink))

	subset = []
	seen = set()
	for root_key in roots:
		for row in thread_by_root.get(root_key, []):
			key = ((row.get("author") or "").strip(), (row.get("permlink") or "").strip())
			if not key[0] or not key[1] or key in seen:
				continue
			seen.add(key)
			subset.append(row)

	return subset


def parse_args():
	parser = argparse.ArgumentParser(
		description="Calculate all account features from collected CSV files."
	)
	parser.add_argument(
		"--input-dir",
		default="./data/steem_collection",
		help="Directory containing accounts/posts/replies/thread_comments/transfers CSV files",
	)
	parser.add_argument(
		"--output-file",
		default="./data/steem_collection/features.csv",
		help="Output CSV path for computed features",
	)
	parser.add_argument(
		"--exclude-thread-deviation",
		action="store_true",
		help=(
			"Drop thread_deviation from output when thread crawl data is unavailable. "
			"This produces the remaining 32 features."
		),
	)
	return parser.parse_args()


def main():
	args = parse_args()
	input_dir = Path(args.input_dir)
	output_file = Path(args.output_file)

	accounts_rows = read_csv_rows(input_dir / "accounts.csv")
	posts_rows = read_csv_rows(input_dir / "posts.csv")
	replies_rows = read_csv_rows(input_dir / "replies.csv")
	thread_rows = read_csv_rows(input_dir / "thread_comments.csv")
	transfers_rows = read_csv_rows(input_dir / "transfers.csv")
	thread_data_available = len(thread_rows) > 0

	accounts_by_author = {
		(row.get("author") or "").strip(): row
		for row in accounts_rows
		if (row.get("author") or "").strip()
	}

	posts_by_author = group_by_author(posts_rows)
	replies_by_author = group_by_author(replies_rows)
	transfers_by_author = group_by_author(transfers_rows)
	thread_by_root, thread_by_comment_key = build_thread_indexes(thread_rows)

	feature_rows = []

	authors = sorted(accounts_by_author.keys())
	for idx, author in enumerate(authors, start=1):
		posts = posts_by_author.get(author, [])
		replies = replies_by_author.get(author, [])
		transfers = transfers_by_author.get(author, [])
		thread_subset = get_author_thread_subset(
			author,
			posts,
			replies,
			thread_by_root,
			thread_by_comment_key,
		)

		features = calculate_all_features(
			author=author,
			account_row=accounts_by_author.get(author, {}),
			posts=posts,
			replies=replies,
			thread_comments=thread_subset,
			transfers=transfers,
		)

		if args.exclude_thread_deviation:
			features.pop("thread_deviation", None)

		features["author"] = author
		feature_rows.append(features)

		print(
			f"[{idx}/{len(authors)}] computed features for {author} "
			f"(posts={len(posts)}, replies={len(replies)}, "
			f"thread_rows={len(thread_subset)}, transfers={len(transfers)})"
		)

	output_file.parent.mkdir(parents=True, exist_ok=True)

	if not feature_rows:
		with output_file.open("w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["author"])
		print(f"No account rows found. Wrote empty output to {output_file}")
		return

	ordered_feature_keys = [k for k in feature_rows[0].keys() if k != "author"]
	fieldnames = ["author"] + ordered_feature_keys

	with output_file.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(feature_rows)

	print(f"Wrote {len(feature_rows)} rows to {output_file}")
	if not thread_data_available:
		print(
			"Thread crawl data not found: thread_deviation may be NaN. "
			"Use --exclude-thread-deviation to keep only the other 32 features."
		)


if __name__ == "__main__":
	main()
