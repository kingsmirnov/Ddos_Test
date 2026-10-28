#!/usr/bin/env python3
"""
A small HTTP load tester for your own site.

It fires a fixed number of requests at a URL, optionally in parallel, and
prints latency and status-code statistics when it finishes.

Only point this at a site you own or have permission to test.

Examples:
    # 200 GET requests, 10 at a time
    python3 loadtest.py --url https://example.com/ -n 200 -c 10

    # POST JSON 50 times, one at a time
    python3 loadtest.py --url https://example.com/api/login -n 50 -c 1 \
        --method POST --data '{"user":"a","pass":"b"}' \
        --header "Content-Type: application/json"

    # send a custom header and stop waiting after 2 seconds per request
    python3 loadtest.py -u https://example.com/ -n 20 -H "Authorization: Bearer x" -t 2
"""

from __future__ import annotations

import argparse
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def parse_header(raw: str) -> tuple[str, str]:
    """Turn 'Name: value' into a (name, value) pair."""
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"header must look like 'Name: value', got {raw!r}"
        )
    name, value = raw.split(":", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"header has an empty name: {raw!r}")
    return name, value.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a fixed number of HTTP requests to a URL and report timings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:", 1)[1],
    )
    parser.add_argument(
        "-u", "--url", required=True, help="the URL to hit (http:// or https://)"
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=100,
        metavar="COUNT",
        help="how many requests to send in total (default: 100)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=10,
        metavar="COUNT",
        help="how many requests to send at the same time (default: 10)",
    )
    parser.add_argument(
        "-m",
        "--method",
        default="GET",
        help="HTTP method to use (default: GET)",
    )
    parser.add_argument(
        "-H",
        "--header",
        action="append",
        default=[],
        type=parse_header,
        metavar="'Name: value'",
        help="extra request header, repeatable",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="request body to send (use with --method POST/PUT/PATCH)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="per-request timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS certificate verification (self-signed local servers only)",
    )
    args = parser.parse_args(argv)

    if not args.url.lower().startswith(("http://", "https://")):
        parser.error("--url must start with http:// or https://")
    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def build_opener(insecure: bool) -> urllib.request.OpenerDirector:
    if not insecure:
        return urllib.request.build_opener()
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))


def send_one(
    opener: urllib.request.OpenerDirector,
    url: str,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[float, int | None, str | None]:
    """Send one request. Returns (latency_ms, status, error)."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as response:
            response.read()
        latency_ms = (time.perf_counter() - started) * 1000
        return latency_ms, response.status, None
    except urllib.error.HTTPError as exc:
        # A 4xx/5xx is still a real response with a real status code.
        try:
            exc.read()
        except Exception:
            pass
        latency_ms = (time.perf_counter() - started) * 1000
        return latency_ms, exc.code, None
    except Exception as exc:  # timeouts, DNS failures, refused connections...
        latency_ms = (time.perf_counter() - started) * 1000
        return latency_ms, None, f"{type(exc).__name__}: {exc}"


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile on an already sorted list."""
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, round(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    headers = dict(args.header)
    body = args.data.encode("utf-8") if args.data is not None else None

    has_content_type = any(key.lower() == "content-type" for key in headers)
    if body is not None and not has_content_type:
        if args.data.lstrip().startswith(("{", "[")):
            headers["Content-Type"] = "application/json"

    opener = build_opener(args.insecure)
    total = args.requests
    workers = min(args.concurrency, total)

    print(
        f"Sending {total} {args.method} request(s) to {args.url} "
        f"with concurrency {workers}..."
    )
    # Only show a progress line for runs long enough to be worth watching.
    progress_every = max(1, total // 10) if total >= 20 else 0

    results: list[tuple[float, int | None, str | None]] = []
    lock_error: str | None = None
    started = time.perf_counter()

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(send_one, opener, args.url, args.method, headers, body, args.timeout)
                for _ in range(total)
            ]
            for done, future in enumerate(futures, start=1):
                results.append(future.result())
                if progress_every and done % progress_every == 0 and done < total:
                    elapsed = time.perf_counter() - started
                    print(f"  {done}/{total} sent ({done / elapsed:,.1f} req/s)", flush=True)
    except KeyboardInterrupt:
        lock_error = "interrupted"
        print("\nStopped early by user.", file=sys.stderr)

    elapsed = time.perf_counter() - started

    if not results:
        print("No requests were completed.", file=sys.stderr)
        return 1

    latencies = sorted(latency for latency, _, _ in results)
    statuses = Counter(
        status if status is not None else "no response" for _, status, _ in results
    )
    errors = Counter(error for _, _, error in results if error)

    ok = sum(count for status, count in statuses.items() if status != "no response")

    print()
    print(f"Time elapsed : {elapsed:.2f} s")
    print(f"Requests     : {len(results)} sent, {ok} got a response")
    print(f"Throughput   : {len(results) / elapsed:,.1f} req/s")
    print()
    print("Latency (ms)")
    print(f"  min        : {latencies[0]:.1f}")
    print(f"  average    : {statistics.fmean(latencies):.1f}")
    print(f"  p50        : {percentile(latencies, 0.50):.1f}")
    print(f"  p95        : {percentile(latencies, 0.95):.1f}")
    print(f"  p99        : {percentile(latencies, 0.99):.1f}")
    print(f"  max        : {latencies[-1]:.1f}")
    print()
    print("Status codes")
    for status, count in sorted(statuses.items(), key=lambda item: str(item[0])):
        print(f"  {status}: {count}")

    if errors:
        print()
        print("Errors")
        for message, count in errors.most_common():
            print(f"  {count}x {message}")

    if lock_error:
        return 130
    return 0 if not errors and all(
        isinstance(status, int) and status < 400 for status in statuses
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
