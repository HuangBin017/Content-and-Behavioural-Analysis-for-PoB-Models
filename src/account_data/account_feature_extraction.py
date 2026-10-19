import numpy as np
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import log
from sklearn.mixture import BayesianGaussianMixture


def calculate_cdfa(texts):
    """
    Calculate the 3 CDFA features for a collection of texts
    belonging to one account.

    Returns:
        mean_mac_cdfa
        variance_mac_cdfa
        num_valid_clusters
    """

    # Remove missing / empty texts
    texts = [
        str(text.lower()).strip()
        for text in texts
        if text is not None and str(text).strip()
    ]

    m = len(texts)

    # Paper sets these values to 0 when fewer than 5 articles exist
    if m < 5:
        return 0.0, 0.0, 0

    # ---------------------------------------------------------
    # Step 1: Split each article into words
    # ---------------------------------------------------------

    word_sets = [
        set(text.split())
        for text in texts
    ]

    # ---------------------------------------------------------
    # Step 2: Construct vocabulary W
    # ---------------------------------------------------------

    vocabulary = set()

    for words in word_sets:
        vocabulary.update(words)

    vocabulary = list(vocabulary)

    n = len(vocabulary)

    if n == 0:
        return 0.0, 0.0, 0

    word_to_index = {
        word: i
        for i, word in enumerate(vocabulary)
    }

    # ---------------------------------------------------------
    # Step 3: Construct occurrence vectors V_j
    #
    # V_j[i] = 1 if word i occurs in article j
    #           0 otherwise
    # ---------------------------------------------------------

    occurrence_vectors = np.zeros((m, n), dtype=float)

    for j, words in enumerate(word_sets):
        for word in words:
            occurrence_vectors[j, word_to_index[word]] = 1

    # ---------------------------------------------------------
    # Step 4: Total occurrence vector T
    #
    # T[i] = number of articles containing word i
    # ---------------------------------------------------------

    total_occurrence = occurrence_vectors.sum(axis=0)

    # ---------------------------------------------------------
    # Step 5: Find frequent words
    #
    # Paper:
    # occurrence >= 10% of maximum occurrence
    # ---------------------------------------------------------

    max_occurrence = total_occurrence.max()

    threshold = 0.10 * max_occurrence

    frequent_word_mask = (
        total_occurrence >= threshold
    )

    # Vf_req
    v_freq = frequent_word_mask.astype(float)

    # ---------------------------------------------------------
    # Step 6: Calculate Euclidean distance between
    # Vf_req and every article vector V_j
    # ---------------------------------------------------------

    distances = np.linalg.norm(
        occurrence_vectors - v_freq,
        axis=1
    )

    # ---------------------------------------------------------
    # Step 7: Cluster distances using a Dirichlet Process
    # Gaussian Mixture Model
    #
    # Maximum number of clusters = 5
    # ---------------------------------------------------------

    X = distances.reshape(-1, 1)

    dpgmm = BayesianGaussianMixture(
        n_components=5,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        random_state=42,
        max_iter=1000
    )

    labels = dpgmm.fit_predict(X)

    # ---------------------------------------------------------
    # Step 8: Find clusters with size >= m/5
    # ---------------------------------------------------------

    minimum_cluster_size = m / 5

    valid_clusters = []

    for cluster_id in np.unique(labels):

        cluster_distances = distances[
            labels == cluster_id
        ]

        cluster_size = len(cluster_distances)

        if cluster_size >= minimum_cluster_size:
            valid_clusters.append(
                (cluster_id, cluster_distances)
            )

    # ---------------------------------------------------------
    # No valid clusters
    # ---------------------------------------------------------

    if len(valid_clusters) == 0:
        return 0.0, 0.0, 0

    # ---------------------------------------------------------
    # Step 9: Select MAC-CDFA
    #
    # MAC-CDFA = cluster with minimum average distance
    # ---------------------------------------------------------

    selected_cluster = min(
        valid_clusters,
        key=lambda x: np.mean(x[1])
    )

    mac_cdfa = selected_cluster[1]

    # ---------------------------------------------------------
    # Step 10: Extract the 3 features
    # ---------------------------------------------------------

    mean_mac_cdfa = float(
        np.mean(mac_cdfa)
    )

    variance_mac_cdfa = float(
        np.var(mac_cdfa)
    )

    num_valid_clusters = len(valid_clusters)

    return (
        mean_mac_cdfa,
        variance_mac_cdfa,
        num_valid_clusters
    )


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_day(dt):
    if dt is None:
        return None
    return dt.date().isoformat()


def _entropy_from_counts(counts):
    values = [int(v) for v in counts if int(v) > 0]
    if not values:
        return 0.0

    total = sum(values)
    if total <= 0:
        return 0.0

    probs = [v / total for v in values]
    return float(-sum(p * log(p) for p in probs if p > 0))


def _tokenize_for_entropy(text):
    if text is None:
        return []
    return re.findall(r"\b\w+\b", str(text).lower())


def _count_links(text):
    if text is None:
        return 0
    t = str(text)
    markdown_links = re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", t, flags=re.IGNORECASE)
    bare_links = re.findall(r"https?://\S+", t, flags=re.IGNORECASE)
    return len(markdown_links) + len(bare_links)


def _count_hashtags(text):
    if text is None:
        return 0
    return len(re.findall(r"(?<!\w)#([A-Za-z0-9_]+)", str(text)))


def _count_mentions(text):
    if text is None:
        return 0
    return len(re.findall(r"(?<!\w)@([A-Za-z0-9\-.]+)", str(text)))


def _get_texts(records, key="body"):
    texts = []
    for row in records:
        value = row.get(key, "") if isinstance(row, dict) else ""
        if value is None:
            continue
        texts.append(str(value))
    return texts


def _daily_counts(records, time_key="created"):
    counts = Counter()
    for row in records:
        dt = _parse_iso_datetime(row.get(time_key, "")) if isinstance(row, dict) else None
        day = _to_day(dt)
        if day:
            counts[day] += 1
    return counts


def _ensure_iterable(records):
    return records if records is not None else []


def _compute_average_response_time_seconds(author, replies, thread_comments):
    deltas = []
    for reply in _ensure_iterable(replies):
        if not isinstance(reply, dict):
            continue
        if str(reply.get("author", "")) != str(author):
            continue

        t_child = _parse_iso_datetime(reply.get("created", ""))
        t_parent = _parse_iso_datetime(reply.get("parent_created", ""))

        # Backward compatibility for older datasets that did not store parent_created.
        if t_parent is None:
            parent_author = reply.get("parent_author", "")
            parent_permlink = reply.get("parent_permlink", "")
            parent = None

            for row in _ensure_iterable(thread_comments):
                if not isinstance(row, dict):
                    continue
                if (
                    row.get("author", "") == parent_author
                    and row.get("permlink", "") == parent_permlink
                ):
                    parent = row
                    break

            if parent is not None:
                t_parent = _parse_iso_datetime(parent.get("created", ""))

        if t_child is None or t_parent is None:
            continue

        delta = (t_child - t_parent).total_seconds()
        if delta >= 0:
            deltas.append(delta)

    if not deltas:
        return 0.0
    return float(np.mean(deltas))


def _compute_thread_deviation(author, replies, thread_comments):
    # Operational definition:
    # For each target reply, compute its position among siblings under the same
    # parent in chronological order (0-based). Thread deviation is the mean
    # sibling position, where higher means deeper deviation from first responses.
    siblings = defaultdict(list)
    for row in _ensure_iterable(thread_comments):
        if not isinstance(row, dict):
            continue
        parent_key = (row.get("parent_author", ""), row.get("parent_permlink", ""))
        created = _parse_iso_datetime(row.get("created", ""))
        siblings[parent_key].append((created, row))

    for parent_key in siblings:
        siblings[parent_key].sort(key=lambda x: (x[0] is None, x[0]))

    positions = []
    for reply in _ensure_iterable(replies):
        if not isinstance(reply, dict):
            continue
        if str(reply.get("author", "")) != str(author):
            continue

        parent_key = (reply.get("parent_author", ""), reply.get("parent_permlink", ""))
        reply_key = (reply.get("author", ""), reply.get("permlink", ""))

        sibling_rows = siblings.get(parent_key, [])
        for idx, (_, row) in enumerate(sibling_rows):
            if (row.get("author", ""), row.get("permlink", "")) == reply_key:
                positions.append(idx)
                break

    if not positions:
        return float("nan")
    return float(np.mean(positions))


def _compute_entropy_rate(posts, replies):
    tokens = []
    for text in _get_texts(posts, "body") + _get_texts(replies, "body"):
        tokens.extend(_tokenize_for_entropy(text))

    if not tokens:
        return 0.0

    counts = Counter(tokens)
    entropy = _entropy_from_counts(counts.values())
    return float(entropy)


def _compute_transfer_metrics(author, transfers):
    author = str(author)
    transfer_rows = [
        t for t in _ensure_iterable(transfers)
        if isinstance(t, dict) and str(t.get("author", "")) == author
    ]

    number_of_transfers = len(transfer_rows)

    transfer_days = Counter()
    transfer_times = []
    in_counterparties = Counter()
    out_counterparties = Counter()

    for tr in transfer_rows:
        t = _parse_iso_datetime(tr.get("created", ""))
        if t is not None:
            transfer_times.append(t)
            transfer_days[_to_day(t)] += 1

        direction = str(tr.get("direction", "")).lower()
        counterparty = str(tr.get("counterparty", ""))
        if direction == "in" and counterparty:
            in_counterparties[counterparty] += 1
        elif direction == "out" and counterparty:
            out_counterparties[counterparty] += 1

    daily_time_entropy_of_transfer = _entropy_from_counts(transfer_days.values())

    if transfer_times:
        transfer_activation_time_days = (
            (max(transfer_times) - min(transfer_times)).total_seconds() / 86400.0
        )
    else:
        transfer_activation_time_days = 0.0

    daily_transfer = (
        number_of_transfers / transfer_activation_time_days
        if transfer_activation_time_days > 0
        else 0.0
    )

    in_degree_of_transfer = len(in_counterparties)
    out_degree_of_transfer = len(out_counterparties)

    entropy_in_degree_accounts = _entropy_from_counts(in_counterparties.values())
    entropy_out_degree_accounts = _entropy_from_counts(out_counterparties.values())

    return {
        "number_of_transfers": float(number_of_transfers),
        "daily_time_entropy_of_transfer": float(daily_time_entropy_of_transfer),
        "transfer_activation_time": float(transfer_activation_time_days),
        "daily_transfer": float(daily_transfer),
        "in_degree_of_transfer": float(in_degree_of_transfer),
        "out_degree_of_transfer": float(out_degree_of_transfer),
        "entropy_of_in_degree_accounts": float(entropy_in_degree_accounts),
        "entropy_of_out_degree_accounts": float(entropy_out_degree_accounts),
    }


def _compute_transfer_per_content_features(posts, replies, transfers):
    post_daily = _daily_counts(posts)
    reply_daily = _daily_counts(replies)
    transfer_daily = _daily_counts(transfers)

    all_days = sorted(set(post_daily.keys()) | set(reply_daily.keys()) | set(transfer_daily.keys()))
    if not all_days:
        return {
            "average_transfer_per_blog_or_reply": 0.0,
            "average_transfer_per_blog": 0.0,
            "average_transfer_per_reply": 0.0,
        }

    ratio_blog_or_reply = []
    ratio_blog = []
    ratio_reply = []

    for day in all_days:
        t = transfer_daily.get(day, 0)
        p = post_daily.get(day, 0)
        r = reply_daily.get(day, 0)

        denom_all = p + r
        if denom_all > 0:
            ratio_blog_or_reply.append(t / denom_all)
        if p > 0:
            ratio_blog.append(t / p)
        if r > 0:
            ratio_reply.append(t / r)

    return {
        "average_transfer_per_blog_or_reply": float(np.mean(ratio_blog_or_reply)) if ratio_blog_or_reply else 0.0,
        "average_transfer_per_blog": float(np.mean(ratio_blog)) if ratio_blog else 0.0,
        "average_transfer_per_reply": float(np.mean(ratio_reply)) if ratio_reply else 0.0,
    }


def calculate_all_features(
    author,
    account_row,
    posts,
    replies,
    thread_comments,
    transfers,
):
    posts = _ensure_iterable(posts)
    replies = _ensure_iterable(replies)
    thread_comments = _ensure_iterable(thread_comments)
    transfers = _ensure_iterable(transfers)

    # 1-9 CDFA feature group
    title_texts = _get_texts(posts, "title")
    post_texts = _get_texts(posts, "body")
    reply_texts = _get_texts(replies, "body")

    cdfa = calculate_cdfa_features(title_texts, post_texts, reply_texts)

    # 10-15 Santia-2019 style features
    all_comment_texts = _get_texts(replies, "body")
    avg_comment_length = (
        float(np.mean([len(t) for t in all_comment_texts])) if all_comment_texts else 0.0
    )

    all_texts = _get_texts(posts, "body") + all_comment_texts
    tokens = []
    for txt in all_texts:
        tokens.extend(_tokenize_for_entropy(txt))
    innovation_rate = (len(set(tokens)) / len(tokens)) if tokens else 0.0

    daily_comments = _daily_counts(replies)
    max_daily_comments = float(max(daily_comments.values())) if daily_comments else 0.0

    number_of_links = float(sum(_count_links(txt) for txt in all_texts))

    average_response_time = _compute_average_response_time_seconds(author, replies, thread_comments)
    thread_deviation = _compute_thread_deviation(author, replies, thread_comments)

    # 16-19 text-based derived features
    entropy_rate = _compute_entropy_rate(posts, replies)

    total_chars = sum(len(txt) for txt in all_texts)
    hashtag_ratio = (
        float(sum(_count_hashtags(txt) for txt in all_texts) / total_chars)
        if total_chars > 0 else 0.0
    )
    mention_ratio = (
        float(sum(_count_mentions(txt) for txt in all_texts) / total_chars)
        if total_chars > 0 else 0.0
    )
    url_ratio = (
        float(sum(_count_links(txt) for txt in all_texts) / total_chars)
        if total_chars > 0 else 0.0
    )

    # 20 FF ratio
    followers = _safe_float(account_row.get("followers", 0.0)) if isinstance(account_row, dict) else 0.0
    following = _safe_float(account_row.get("following", 0.0)) if isinstance(account_row, dict) else 0.0
    ff_ratio = followers / (followers + following) if (followers + following) > 0 else 0.0

    # 21 age of account (days)
    created_dt = _parse_iso_datetime(account_row.get("created", "")) if isinstance(account_row, dict) else None
    dataset_end_dt = _parse_iso_datetime(account_row.get("dataset_end_time", "")) if isinstance(account_row, dict) else None
    age_of_account = (
        (dataset_end_dt - created_dt).total_seconds() / 86400.0
        if created_dt and dataset_end_dt and dataset_end_dt >= created_dt else 0.0
    )

    # 22-30 transfer & blockchain features
    transfer_metrics = _compute_transfer_metrics(author, transfers)

    created_by_steem_raw = ""
    if isinstance(account_row, dict):
        created_by_steem_raw = str(account_row.get("created_by_steem", "")).strip().lower()
    steem_created_account = 1.0 if created_by_steem_raw in ("true", "1", "yes") else 0.0

    # 31-33 average transfer per content/day features
    transfer_content = _compute_transfer_per_content_features(posts, replies, transfers)

    return {
        # 1-9
        "average_mac_cdfa_t": cdfa["cdfa_t_mean"],
        "variance_mac_cdfa_t": cdfa["cdfa_t_variance"],
        "num_clusters_cdfa_t": cdfa["cdfa_t_clusters"],
        "average_mac_cdfa_c": cdfa["cdfa_c_mean"],
        "variance_mac_cdfa_c": cdfa["cdfa_c_variance"],
        "num_clusters_cdfa_c": cdfa["cdfa_c_clusters"],
        "average_mac_cdfa_r": cdfa["cdfa_r_mean"],
        "variance_mac_cdfa_r": cdfa["cdfa_r_variance"],
        "num_clusters_cdfa_r": cdfa["cdfa_r_clusters"],
        # 10-15
        "average_response_time": float(average_response_time),
        "average_comment_length": float(avg_comment_length),
        "innovation_rate": float(innovation_rate),
        "maximum_daily_comments": float(max_daily_comments),
        "number_of_links": float(number_of_links),
        # 16-21
        "entropy_rate": float(entropy_rate),
        "hashtag_ratio": float(hashtag_ratio),
        "mention_ratio": float(mention_ratio),
        "url_ratio": float(url_ratio),
        "ff_ratio": float(ff_ratio),
        "age_of_account": float(age_of_account),
        # 22-30
        "number_of_transfers": transfer_metrics["number_of_transfers"],
        "daily_time_entropy_of_transfer": transfer_metrics["daily_time_entropy_of_transfer"],
        "transfer_activation_time": transfer_metrics["transfer_activation_time"],
        "daily_transfer": transfer_metrics["daily_transfer"],
        "in_degree_of_transfer": transfer_metrics["in_degree_of_transfer"],
        "out_degree_of_transfer": transfer_metrics["out_degree_of_transfer"],
        "entropy_of_in_degree_accounts": transfer_metrics["entropy_of_in_degree_accounts"],
        "entropy_of_out_degree_accounts": transfer_metrics["entropy_of_out_degree_accounts"],
        "steem_created_account": float(steem_created_account),
        # 31-33
        "average_transfer_per_blog_or_reply": transfer_content["average_transfer_per_blog_or_reply"],
        "average_transfer_per_blog": transfer_content["average_transfer_per_blog"],
        "average_transfer_per_reply": transfer_content["average_transfer_per_reply"],
    }


def calculate_cdfa_features(titles, contents, replies):

    mean_t, var_t, clusters_t = calculate_cdfa(titles)
    mean_c, var_c, clusters_c = calculate_cdfa(contents)
    mean_r, var_r, clusters_r = calculate_cdfa(replies)

    return {
        "cdfa_t_mean": mean_t,
        "cdfa_t_variance": var_t,
        "cdfa_t_clusters": clusters_t,

        "cdfa_c_mean": mean_c,
        "cdfa_c_variance": var_c,
        "cdfa_c_clusters": clusters_c,

        "cdfa_r_mean": mean_r,
        "cdfa_r_variance": var_r,
        "cdfa_r_clusters": clusters_r,
    }