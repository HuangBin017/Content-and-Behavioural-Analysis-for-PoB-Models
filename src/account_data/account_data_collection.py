from beem import Steem
from beem.account import Account
import csv
import json
import os
import random
import re
import time
from collections import deque
from datetime import datetime, timezone

steem = Steem(
    node=["https://api.steemyy.com"]
)

file_path = "./comments/authors.csv"
accounts_output_path = "./data/steem_collection/accounts.csv"
posts_output_path = "./data/steem_collection/posts.csv"
replies_output_path = "./data/steem_collection/replies.csv"
thread_comments_output_path = "./data/steem_collection/thread_comments.csv"
transfers_output_path = "./data/steem_collection/transfers.csv"
thread_debug_output_path = "./data/steem_collection/debug_thread_payloads.jsonl"
sample_size = 1000
random_seed = 42
MAX_POST_COUNT = 500
MIN_POST_REPLY_COUNT = 40
MAX_OPERATION_COUNT_PER_ACCOUNT = 50000
RESUME_AFTER_AUTHOR = "koko123"

# Thread crawl safeguards to prevent apparent hangs under sustained 429 throttling.
THREAD_MAX_NODES_PER_ROOT = 2000
THREAD_MAX_CONSECUTIVE_RATE_LIMITS = 20
THREAD_RATE_LIMIT_BASE_SLEEP_SECONDS = 3
# Optional heavy crawl used only for thread-deviation feature context.
ENABLE_THREAD_COLLECTION = False

def is_likely_valid_steem_account(name):
    # Simple sanity check to skip malformed rows like '-' in source CSV.
    return bool(re.fullmatch(r"[a-z][a-z0-9\-.]{1,14}[a-z0-9]", name))


def load_authors(path, offset=0, limit=None):
    authors = []
    seen = set()

    with open(path, "r", newline="", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)

        if "author" in first_line.lower().split(","):
            reader = csv.DictReader(f)
            for row in reader:
                author = str(row.get("author", "")).strip()
                if not is_likely_valid_steem_account(author):
                    continue
                if author in seen:
                    continue
                seen.add(author)
                authors.append(author)
        else:
            for line in f:
                author = line.strip()
                if not is_likely_valid_steem_account(author):
                    continue
                if author in seen:
                    continue
                seen.add(author)
                authors.append(author)

    if limit is None:
        return authors[offset:]
    return authors[offset:offset + limit]


all_authors = load_authors(file_path)
rng = random.Random(random_seed)
candidate_authors = all_authors[:]
rng.shuffle(candidate_authors)

if RESUME_AFTER_AUTHOR:
    if RESUME_AFTER_AUTHOR in candidate_authors:
        resume_index = candidate_authors.index(RESUME_AFTER_AUTHOR)
        candidate_authors = candidate_authors[resume_index + 1:]
        print(
            f"Resume enabled: starting after author '{RESUME_AFTER_AUTHOR}' "
            f"(remaining candidates={len(candidate_authors)})"
        )
    else:
        print(
            f"Resume author '{RESUME_AFTER_AUTHOR}' not found in candidate list; "
            "starting from beginning."
        )

print(
    f"Loaded {len(all_authors)} valid authors; "
    f"sampling up to {sample_size} with post_count < {MAX_POST_COUNT}, "
    f"posts+replies >= {MIN_POST_REPLY_COUNT}, "
    f"and at most {MAX_OPERATION_COUNT_PER_ACCOUNT} ops/account"
)
print(
    "Thread-deviation collection enabled "
    f"(optional heavy crawl): {ENABLE_THREAD_COLLECTION}"
)


def is_rate_limit_response(response):
    if not isinstance(response, dict):
        return False

    raw = str(response.get("raw", ""))
    warning = str(response.get("warning", ""))

    return (
        "429 Too Many Requests" in raw
        or "Upstream did not return JSON" in warning
        or "429" in raw
    )


def collect_account_data(username, dataset_end_time, creation_info=None, max_retries=5):
    created_by = ""
    created_by_steem = ""

    creation_info = creation_info or {}
    if creation_info:
        created_by = creation_info.get("creator", "")
        created_by_steem = str(created_by == "steem")

    for attempt in range(max_retries):
        try:
            account = Account(username, blockchain_instance=steem)
            follow_count = steem.rpc.get_follow_count(
                username,
                api="condenser_api"
            )

            return {
                "author": username,
                "created": account.get("created", ""),
                "followers": follow_count.get("follower_count", 0),
                "following": follow_count.get("following_count", 0),
                "dataset_end_time": dataset_end_time,
                "created_by": created_by,
                "created_by_steem": created_by_steem,
            }

        except Exception as e:
            print(
                f"[{username}] account data attempt "
                f"{attempt + 1}/{max_retries} failed: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def get_account_post_count(username, max_retries=5):
    for attempt in range(max_retries):
        try:
            account = Account(username, blockchain_instance=steem)
            value = account.get("post_count", 0)
            return int(value)
        except Exception as e:
            print(
                f"[{username}] post_count attempt "
                f"{attempt + 1}/{max_retries} failed: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def append_account_row(csv_path, row):
    headers = [
        "author",
        "created",
        "followers",
        "following",
        "dataset_end_time",
        "created_by",
        "created_by_steem",
    ]

    try:
        with open(csv_path, "x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    except FileExistsError:
        pass

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row)


def append_comment_rows(csv_path, rows):
    headers = [
        "author",
        "permlink",
        "parent_author",
        "parent_permlink",
        "parent_created",
        "title",
        "body",
        "created",
        "depth",
        "json_metadata",
        "block_num",
        "transaction_id",
    ]

    try:
        with open(csv_path, "x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    except FileExistsError:
        pass

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        for row in rows:
            writer.writerow(row)


def append_thread_comment_rows(csv_path, rows):
    headers = [
        "author",
        "permlink",
        "parent_author",
        "parent_permlink",
        "title",
        "body",
        "created",
        "depth",
        "json_metadata",
        "block_num",
        "transaction_id",
        "root_author",
        "root_permlink",
    ]

    try:
        with open(csv_path, "x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    except FileExistsError:
        pass

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        for row in rows:
            writer.writerow(row)


def append_transfer_rows(csv_path, rows):
    headers = [
        "author",
        "direction",
        "counterparty",
        "from_account",
        "to_account",
        "amount_raw",
        "amount_value",
        "asset",
        "memo",
        "created",
        "block_num",
        "transaction_id",
        "operation_index",
        "op_type",
    ]

    try:
        with open(csv_path, "x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    except FileExistsError:
        pass

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        for row in rows:
            writer.writerow(row)


def parse_amount_parts(amount_raw):
    text = str(amount_raw or "").strip()
    if not text:
        return "", ""

    parts = text.split(" ")
    if len(parts) != 2:
        return text, ""
    return parts[0], parts[1]


def get_account_creation_info(username, max_retries=5, batch_size=1000):
    start = -1
    processed_indices = set()

    while True:
        result = None
        for attempt in range(max_retries):
            try:
                result = steem.rpc.get_account_history(
                    username,
                    start,
                    batch_size,
                    api="condenser_api"
                )

                if is_rate_limit_response(result):
                    time.sleep(2)
                    result = None
                    continue

                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2)

        if result is None or not result:
            break

        oldest_index = min(item[0] for item in result)

        for item in result:
            index = item[0]
            if index in processed_indices:
                continue
            processed_indices.add(index)

            operation_data = item[1]
            operation = operation_data.get("op") if isinstance(operation_data, dict) else None
            if not operation or len(operation) != 2:
                continue

            op_type = operation[0]
            data = operation[1]

            if op_type in ("account_create", "create_claimed_account"):
                if data.get("new_account_name") == username:
                    return {
                        "creator": data.get("creator", ""),
                        "created": operation_data.get("timestamp", ""),
                        "op_type": op_type,
                        "block_num": operation_data.get("block", ""),
                        "transaction_id": operation_data.get("trx_id", ""),
                    }

        next_start = oldest_index
        if next_start < 0:
            break

        if next_start <= 100:
            start = 100
        else:
            start = next_start

    return {}


def get_account_transfers(username, max_retries=5, batch_size=100):
    transfers = {}
    processed_indices = set()

    start = -1
    end = False

    while True:
        result = None
        if start == 100:
            end = True

        for attempt in range(max_retries):
            try:
                result = steem.rpc.get_account_history(
                    username,
                    start,
                    batch_size,
                    api="condenser_api"
                )

                if is_rate_limit_response(result):
                    print(
                        f"[{username}] transfer batch start={start}, "
                        f"rate limited (429). Waiting 5 seconds before retrying."
                    )
                    time.sleep(5)
                    result = None
                    continue

                break
            except Exception as e:
                print(
                    f"[{username}] transfer batch start={start}, "
                    f"attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)

        if result is None:
            print(
                f"[{username}] Failed transfer batch start={start} "
                f"after {max_retries} attempts. Moving on."
            )
            break

        if not result:
            break

        oldest_index = min(item[0] for item in result)
        next_start = oldest_index

        if next_start < 0:
            break

        if next_start <= 100:
            start = 100
        else:
            start = next_start

        for item in result:
            index = item[0]
            if index in processed_indices:
                continue
            processed_indices.add(index)

            operation_data = item[1]
            operation = operation_data.get("op") if isinstance(operation_data, dict) else None
            if not operation or len(operation) != 2:
                continue

            op_type = operation[0]
            data = operation[1]
            if op_type != "transfer":
                continue

            from_account = data.get("from", "")
            to_account = data.get("to", "")
            amount_raw = data.get("amount", "")
            amount_value, asset = parse_amount_parts(amount_raw)

            if from_account == username:
                direction = "out"
                counterparty = to_account
            elif to_account == username:
                direction = "in"
                counterparty = from_account
            else:
                continue

            transfer_key = (
                operation_data.get("trx_id", ""),
                operation_data.get("block", ""),
                index,
            )

            if transfer_key in transfers:
                continue

            transfers[transfer_key] = {
                "author": username,
                "direction": direction,
                "counterparty": counterparty,
                "from_account": from_account,
                "to_account": to_account,
                "amount_raw": amount_raw,
                "amount_value": amount_value,
                "asset": asset,
                "memo": data.get("memo", ""),
                "created": operation_data.get("timestamp") or data.get("timestamp", ""),
                "block_num": operation_data.get("block", ""),
                "transaction_id": operation_data.get("trx_id", ""),
                "operation_index": index,
                "op_type": op_type,
            }

        if not result:
            continue

        if start < 0 or end:
            break

    return list(transfers.values())


def get_content_with_retries(author, permlink, max_retries=5):
    saw_rate_limit = False
    for attempt in range(max_retries):
        try:
            response = steem.rpc.get_content(author, permlink, api="condenser_api")

            if is_rate_limit_response(response):
                saw_rate_limit = True
                print(
                    f"[{author}/{permlink}] get_content rate-limited; "
                    f"retry {attempt + 1}/{max_retries}"
                )
                log_thread_debug_payload(
                    context="rate_limited_get_content",
                    author=author,
                    permlink=permlink,
                    payload=response,
                )
                time.sleep(THREAD_RATE_LIMIT_BASE_SLEEP_SECONDS * (attempt + 1))
                continue

            if isinstance(response, dict):
                return response

            log_thread_debug_payload(
                context="unexpected_get_content_payload",
                author=author,
                permlink=permlink,
                payload=response,
            )
            return {}
        except Exception as e:
            print(
                f"[{author}/{permlink}] get_content attempt "
                f"{attempt + 1}/{max_retries} failed: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2)

    if saw_rate_limit:
        log_thread_debug_payload(
            context="rate_limited_get_content_exhausted",
            author=author,
            permlink=permlink,
            payload={"max_retries": max_retries},
        )
    return {}


def get_content_replies_with_retries(author, permlink, max_retries=5):
    saw_rate_limit = False
    for attempt in range(max_retries):
        try:
            response = steem.rpc.get_content_replies(author, permlink, api="condenser_api")

            if is_rate_limit_response(response):
                saw_rate_limit = True
                print(
                    f"[{author}/{permlink}] get_content_replies rate-limited; "
                    f"retry {attempt + 1}/{max_retries}"
                )
                log_thread_debug_payload(
                    context="rate_limited_get_content_replies",
                    author=author,
                    permlink=permlink,
                    payload=response,
                )
                time.sleep(THREAD_RATE_LIMIT_BASE_SLEEP_SECONDS * (attempt + 1))
                continue

            if isinstance(response, list):
                return response, False

            # Some nodes return non-standard shapes on upstream errors.
            log_thread_debug_payload(
                context="unexpected_get_content_replies_payload",
                author=author,
                permlink=permlink,
                payload=response,
            )

            if isinstance(response, dict):
                result = response.get("result")
                if isinstance(result, list):
                    return result, False

            return [], False
        except Exception as e:
            print(
                f"[{author}/{permlink}] get_content_replies attempt "
                f"{attempt + 1}/{max_retries} failed: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2)

    if saw_rate_limit:
        log_thread_debug_payload(
            context="rate_limited_get_content_replies_exhausted",
            author=author,
            permlink=permlink,
            payload={"max_retries": max_retries},
        )
    return [], saw_rate_limit


def log_thread_debug_payload(context, author, permlink, payload):
    os.makedirs(os.path.dirname(thread_debug_output_path), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "context": context,
        "author": author,
        "permlink": permlink,
        "payload_type": type(payload).__name__,
        "payload": payload,
    }
    with open(thread_debug_output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def comment_to_thread_row(comment_obj, root_author, root_permlink):
    return {
        "author": comment_obj.get("author", ""),
        "permlink": comment_obj.get("permlink", ""),
        "parent_author": comment_obj.get("parent_author", ""),
        "parent_permlink": comment_obj.get("parent_permlink", ""),
        "title": comment_obj.get("title", ""),
        "body": comment_obj.get("body", ""),
        "created": comment_obj.get("created", ""),
        "depth": comment_obj.get("depth", ""),
        "json_metadata": comment_obj.get("json_metadata", ""),
        "block_num": comment_obj.get("block", ""),
        "transaction_id": comment_obj.get("trx_id", ""),
        "root_author": root_author,
        "root_permlink": root_permlink,
    }


def collect_thread_comments(root_author, root_permlink, seen_comments=None, max_retries=5):
    if seen_comments is None:
        seen_comments = set()

    thread_rows = []
    queue = deque([(root_author, root_permlink)])
    visited_nodes = set()
    nodes_processed = 0
    consecutive_rate_limits = 0

    print(f"[thread {root_author}/{root_permlink}] start")

    while queue:
        if nodes_processed >= THREAD_MAX_NODES_PER_ROOT:
            print(
                f"[thread {root_author}/{root_permlink}] "
                f"stopped: node limit {THREAD_MAX_NODES_PER_ROOT} reached"
            )
            log_thread_debug_payload(
                context="thread_node_limit_reached",
                author=root_author,
                permlink=root_permlink,
                payload={"nodes_processed": nodes_processed},
            )
            break

        current_author, current_permlink = queue.popleft()
        nodes_processed += 1

        if nodes_processed % 50 == 0:
            print(
                f"[thread {root_author}/{root_permlink}] "
                f"processed_nodes={nodes_processed} queue_size={len(queue)}"
            )

        node_key = (current_author, current_permlink)
        if node_key in visited_nodes:
            continue
        visited_nodes.add(node_key)

        current_content = get_content_with_retries(
            current_author,
            current_permlink,
            max_retries=max_retries,
        )

        if isinstance(current_content, dict):
            content_author = current_content.get("author", "")
            content_permlink = current_content.get("permlink", "")
            if content_author and content_permlink:
                key = (content_author, content_permlink)
                if key not in seen_comments:
                    seen_comments.add(key)
                    thread_rows.append(
                        comment_to_thread_row(
                            current_content,
                            root_author,
                            root_permlink,
                        )
                    )

        replies, rate_limited = get_content_replies_with_retries(
            current_author,
            current_permlink,
            max_retries=max_retries,
        )

        if rate_limited:
            consecutive_rate_limits += 1
            print(
                f"[thread {root_author}/{root_permlink}] "
                f"consecutive_rate_limits={consecutive_rate_limits}"
            )
        else:
            consecutive_rate_limits = 0

        if consecutive_rate_limits >= THREAD_MAX_CONSECUTIVE_RATE_LIMITS:
            print(
                f"[thread {root_author}/{root_permlink}] "
                f"stopped: too many consecutive rate limits "
                f"({consecutive_rate_limits})"
            )
            log_thread_debug_payload(
                context="thread_aborted_due_to_rate_limits",
                author=root_author,
                permlink=root_permlink,
                payload={
                    "consecutive_rate_limits": consecutive_rate_limits,
                    "nodes_processed": nodes_processed,
                    "queue_size": len(queue),
                },
            )
            break

        for reply in replies:
            if not isinstance(reply, dict):
                log_thread_debug_payload(
                    context="non_dict_reply_in_thread",
                    author=current_author,
                    permlink=current_permlink,
                    payload=reply,
                )
                continue

            reply_author = reply.get("author", "")
            reply_permlink = reply.get("permlink", "")
            if not reply_author or not reply_permlink:
                continue

            key = (reply_author, reply_permlink)
            if key not in seen_comments:
                seen_comments.add(key)
                thread_rows.append(
                    comment_to_thread_row(
                        reply,
                        root_author,
                        root_permlink,
                    )
                )

                queue.append((reply_author, reply_permlink))

    print(
        f"[thread {root_author}/{root_permlink}] done "
        f"processed_nodes={nodes_processed} collected_rows={len(thread_rows)}"
    )
    return thread_rows


def clean_post_body(body, json_metadata):
    try:
        metadata = json.loads(json_metadata or "{}")
    except (json.JSONDecodeError, TypeError):
        metadata = {}

    # Remove images listed in metadata
    for image in metadata.get("image", []):
        image_escaped = re.escape(image)

        body = re.sub(
            rf'!\[[^\]]*\]\(\s*{image_escaped}\s*\)',
            '',
            body
        )

    # Remove any remaining Markdown images
    body = re.sub(
        r'!\[[^\]]*\]\([^)]*\)',
        '',
        body
    )

    # Convert Markdown links to their visible text
    body = re.sub(
        r'\[([^\]]+)\]\(https?://[^)]*\)',
        r'\1',
        body
    )

    # Remove HTML images
    body = re.sub(
        r'<img[^>]*>',
        '',
        body,
        flags=re.IGNORECASE
    )

    # Remove remaining URLs
    body = re.sub(
        r'https?://\S+',
        '',
        body
    )

    # Remove HTML formatting tags
    body = re.sub(
        r'<[^>]+>',
        '',
        body
    )

    # Remove SteemX boilerplate
    body = re.sub(
        r'Posted using SteemX',
        '',
        body,
        flags=re.IGNORECASE
    )

    # Clean excessive whitespace
    body = re.sub(
        r'\n\s*\n+',
        '\n\n',
        body
    ).strip()

    return body


def get_account_posts(username, max_retries=5, batch_size=100):
    posts = {}
    replies = {}
    transfers = {}
    creation_info = {}
    parent_created_cache = {}
    processed_indices = set()
    operations_processed = 0

    start = -1
    end = False

    while True:
        if operations_processed >= MAX_OPERATION_COUNT_PER_ACCOUNT:
            print(
                f"[{username}] reached operation cap "
                f"({MAX_OPERATION_COUNT_PER_ACCOUNT}); stopping history scan."
            )
            break

        result = None
        if start == 100:
            end = True
        for attempt in range(max_retries):
            try:
                result = steem.rpc.get_account_history(
                    username,
                    start,
                    batch_size,
                    api="condenser_api"
                )

                if is_rate_limit_response(result):
                    print(
                        f"[{username}] "
                        f"Batch start={start}, "
                        f"rate limited (429). Waiting 5 seconds before retrying."
                    )
                    time.sleep(5)
                    result = None
                    continue

                break

            except Exception as e:
                print(
                    f"[{username}] "
                    f"Batch start={start}, "
                    f"attempt {attempt + 1}/{max_retries} failed: {e}"
                )

                if attempt < max_retries - 1:
                    time.sleep(2)

        if result is None:
            print(
                f"[{username}] "
                f"Failed batch start={start} after "
                f"{max_retries} attempts. Moving on."
            )
            break

        # No more history
        if not result:
            break

        oldest_index = min(item[0] for item in result)
        newest_index = max(item[0] for item in result)

        print(
            f"[{username}] "
            f"Processed operations "
            f"{oldest_index} -> {newest_index}"
        )

        next_start = oldest_index

        if next_start < 0:
            break

        if next_start <= 100:
            start = 100

        else:
            start = next_start

        stop_due_to_op_cap = False
        for item in result:

            if operations_processed >= MAX_OPERATION_COUNT_PER_ACCOUNT:
                stop_due_to_op_cap = True
                break

            index = item[0]

            if index in processed_indices:
                continue
            processed_indices.add(index)
            operations_processed += 1

            operation_data = item[1]

            try:
                op_type = operation_data["op"]
                operation = operation_data["op"]
            except (KeyError, TypeError):
                print(
                    f"[{username}] "
                    f"Skipping operation at index {index} "
                    f"due to missing 'op' field."
                    f"item: {item}"
                )
                print(f"Operation data: {operation_data}")
                continue

            op_type = operation[0]
            data = operation[1]

            if op_type in ("account_create", "create_claimed_account"):
                if data.get("new_account_name") == username and not creation_info:
                    creation_info = {
                        "creator": data.get("creator", ""),
                        "created": operation_data.get("timestamp", ""),
                        "op_type": op_type,
                        "block_num": operation_data.get("block", ""),
                        "transaction_id": operation_data.get("trx_id", ""),
                    }
                continue

            if op_type == "transfer":
                from_account = data.get("from", "")
                to_account = data.get("to", "")
                amount_raw = data.get("amount", "")
                amount_value, asset = parse_amount_parts(amount_raw)

                if from_account == username:
                    direction = "out"
                    counterparty = to_account
                elif to_account == username:
                    direction = "in"
                    counterparty = from_account
                else:
                    continue

                transfer_key = (
                    operation_data.get("trx_id", ""),
                    operation_data.get("block", ""),
                    index,
                )

                if transfer_key not in transfers:
                    transfers[transfer_key] = {
                        "author": username,
                        "direction": direction,
                        "counterparty": counterparty,
                        "from_account": from_account,
                        "to_account": to_account,
                        "amount_raw": amount_raw,
                        "amount_value": amount_value,
                        "asset": asset,
                        "memo": data.get("memo", ""),
                        "created": operation_data.get("timestamp") or data.get("timestamp", ""),
                        "block_num": operation_data.get("block", ""),
                        "transaction_id": operation_data.get("trx_id", ""),
                        "operation_index": index,
                        "op_type": op_type,
                    }
                continue

            # Only posts/replies
            if op_type != "comment":
                continue

            created = operation_data.get("timestamp") or data.get("timestamp")
            block_num = operation_data.get("block")
            transaction_id = operation_data.get("trx_id")

            # posts
            if (
                data.get("author") == username
                and data.get("parent_author", "") == ""
            ):
                permlink = data.get("permlink")

                if permlink:
                    if permlink not in posts:
                        posts[permlink] = {
                            "author": data.get("author", ""),
                            "permlink": permlink,
                            "parent_author": data.get("parent_author", ""),
                            "parent_permlink": data.get("parent_permlink", ""),
                            "title": data.get("title", ""),
                            "body": data.get("body", ""),
                            "created": created,
                            "depth": data.get("depth", ""),
                            "json_metadata": data.get("json_metadata", ""),
                            "block_num": block_num,
                            "transaction_id": transaction_id,
                        }
            # replies
            elif (
                data.get("author") == username
                and data.get("parent_author", "") != ""
            ):
                permlink = data.get("permlink")
                parent_author = data.get("parent_author", "")
                parent_permlink = data.get("parent_permlink", "")
                parent_created = ""

                parent_key = (parent_author, parent_permlink)
                if parent_author and parent_permlink:
                    if parent_key not in parent_created_cache:
                        parent_content = get_content_with_retries(
                            parent_author,
                            parent_permlink,
                        )
                        if isinstance(parent_content, dict):
                            parent_created_cache[parent_key] = parent_content.get("created", "")
                        else:
                            parent_created_cache[parent_key] = ""
                    parent_created = parent_created_cache.get(parent_key, "")

                if permlink not in replies:
                    replies[permlink] = {
                        "author": data.get("author", ""),
                        "permlink": permlink,
                        "parent_author": parent_author,
                        "parent_permlink": parent_permlink,
                        "parent_created": parent_created,
                        "title": data.get("title", ""),
                        "body": data.get("body", ""),
                        "created": created,
                        "depth": data.get("depth", ""),
                        "json_metadata": data.get("json_metadata", ""),
                        "block_num": block_num,
                        "transaction_id": transaction_id,
                    }
                continue

        if stop_due_to_op_cap:
            print(
                f"[{username}] operation cap reached during batch; "
                "ending collection for this account."
            )
            break

        if not result:
            continue

        if start < 0 or end:
            break

    return list(posts.values()), list(replies.values()), list(transfers.values()), creation_info


dataset_end_time = (
    datetime.now(timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z")
)

collected_accounts = 0
skipped_high_post_count = 0
skipped_low_post_reply_count = 0

for author in candidate_authors:
    if collected_accounts >= sample_size:
        break

    post_count = get_account_post_count(author)
    if post_count is None:
        print(f"[{author}] skipped: unable to retrieve post_count")
        continue

    if post_count >= MAX_POST_COUNT:
        skipped_high_post_count += 1
        print(
            f"[{author}] skipped: post_count={post_count} "
            f">= {MAX_POST_COUNT}"
        )
        continue

    print(
        f"[{author}] accepted for collection: post_count={post_count} "
        f"(< {MAX_POST_COUNT})"
    )

    posts, replies, transfers, creation_info = get_account_posts(author)
    post_reply_count = len(posts) + len(replies)
    if post_reply_count < MIN_POST_REPLY_COUNT:
        skipped_low_post_reply_count += 1
        print(
            f"[{author}] skipped after collection: "
            f"posts+replies={post_reply_count} < {MIN_POST_REPLY_COUNT}"
        )
        continue

    account_row = collect_account_data(
        author,
        dataset_end_time,
        creation_info=creation_info,
    )
    if account_row is None:
        continue

    thread_rows = []
    if ENABLE_THREAD_COLLECTION:
        processed_roots = set()
        seen_thread_comments = set()

        for post in posts:
            root_author = post.get("author", "")
            root_permlink = post.get("permlink", "")
            if not root_author or not root_permlink:
                continue
            root_key = (root_author, root_permlink)
            if root_key in processed_roots:
                continue
            processed_roots.add(root_key)
            thread_rows.extend(
                collect_thread_comments(
                    root_author,
                    root_permlink,
                    seen_comments=seen_thread_comments,
                )
            )

        for reply in replies:
            reply_author = reply.get("author", "")
            reply_permlink = reply.get("permlink", "")
            if not reply_author or not reply_permlink:
                continue

            reply_content = get_content_with_retries(reply_author, reply_permlink)
            root_author = reply_content.get("root_author") if isinstance(reply_content, dict) else None
            root_permlink = reply_content.get("root_permlink") if isinstance(reply_content, dict) else None

            if not root_author or not root_permlink:
                # Fallback: if root is not present, use the immediate parent as closest known thread root.
                root_author = reply.get("parent_author", "")
                root_permlink = reply.get("parent_permlink", "")

            if not root_author or not root_permlink:
                continue

            root_key = (root_author, root_permlink)
            if root_key in processed_roots:
                continue
            processed_roots.add(root_key)
            thread_rows.extend(
                collect_thread_comments(
                    root_author,
                    root_permlink,
                    seen_comments=seen_thread_comments,
                )
            )

    append_account_row(accounts_output_path, account_row)
    append_comment_rows(posts_output_path, posts)
    append_comment_rows(replies_output_path, replies)
    if ENABLE_THREAD_COLLECTION:
        append_thread_comment_rows(thread_comments_output_path, thread_rows)
    append_transfer_rows(transfers_output_path, transfers)
    print(
        f"[{author}] saved account row "
        f"(followers={account_row['followers']}, following={account_row['following']})"
    )
    print(
        f"[{author}] saved posts={len(posts)} replies={len(replies)}"
    )
    print(
        f"[{author}] saved thread_comments={len(thread_rows)}"
    )
    print(
        f"[{author}] saved transfers={len(transfers)}"
    )

    collected_accounts += 1
    print(
        f"Progress: collected_accounts={collected_accounts}/{sample_size}, "
        f"skipped_high_post_count={skipped_high_post_count}, "
        f"skipped_low_post_reply_count={skipped_low_post_reply_count}"
    )

if collected_accounts < sample_size:
    print(
        f"WARNING: collected only {collected_accounts}/{sample_size} accounts "
        f"with post_count < {MAX_POST_COUNT}, "
        f"posts+replies >= {MIN_POST_REPLY_COUNT}, "
        f"and <= {MAX_OPERATION_COUNT_PER_ACCOUNT} processed ops/account."
    )