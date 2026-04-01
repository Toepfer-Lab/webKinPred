"""
Benchmark kcat inference latency for all production API kcat-capable methods.

For each method, this command:
1) Generates 1,000 reactions from 100 unique proteins (defaults), where
   proteins have average length 400 and max length 1,000.
2) Runs one prediction job with fresh proteins (uncached embedding path).
3) Runs a second prediction job with the exact same proteins (cached path).
4) Prints a readable per-method summary:
   DLKcat - 100.00s compute (not cached), 10.00s compute (cached)

The command submits jobs to the public REST API (/api/v1/submit/) and polls
/api/v1/status/<job_id>/, reporting only server-side compute time
(`computeSeconds`) and not queue time.
"""

from __future__ import annotations

import hashlib
import os
import random
import secrets
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from django.core.management.base import BaseCommand, CommandError

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SUBSTRATE_POOL = [
    "CCO",             # ethanol
    "CC(=O)O",         # acetic acid
    "O=C=O",           # carbon dioxide
    "C1=CC=CC=C1",     # benzene
    "CCN",             # ethylamine
    "CC(O)C",          # isopropanol
    "CC(C)O",          # isobutanol-like
    "CCOC(=O)C",       # ethyl acetate
]
PRODUCT_POOL = [
    "CC=O",            # acetaldehyde
    "O",               # water
    "CO",              # methanol
    "CC(=O)O",         # acetic acid
    "C=O",             # formaldehyde
    "CCO",             # ethanol
    "O=C=O",           # carbon dioxide
    "CC(=O)N",         # acetamide
]
DEFAULT_API_BASE_URL = "https://predictor.openkinetics.org/api/v1"


@dataclass
class SingleRunResult:
    compute_seconds: float | None
    status: str
    error_message: str
    public_id: str


@dataclass
class MethodBenchmarkResult:
    method_key: str
    uncached: SingleRunResult
    cached: SingleRunResult


class Command(BaseCommand):
    help = (
        "Benchmark kcat inference time for each registered kcat-capable method, "
        "running once uncached and once cached."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--num-reactions",
            type=int,
            default=1000,
            help="Total reactions per run (default: 1000).",
        )
        parser.add_argument(
            "--num-proteins",
            type=int,
            default=100,
            help="Unique proteins to distribute across reactions (default: 100).",
        )
        parser.add_argument(
            "--avg-seq-len",
            type=int,
            default=400,
            help="Average protein length target (default: 400).",
        )
        parser.add_argument(
            "--max-seq-len",
            type=int,
            default=1000,
            help="Maximum protein length (default: 1000).",
        )
        parser.add_argument(
            "--testmode",
            action="store_true",
            help=(
                "Run the exact same benchmark flow but override workload to "
                "10 reactions and 1 unique protein."
            ),
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help=(
                "Optional fixed seed for reproducible synthetic proteins. "
                "If omitted, a fresh random seed is used each invocation."
            ),
        )
        parser.add_argument(
            "--methods",
            nargs="+",
            default=None,
            help=(
                "Optional subset of method keys to benchmark "
                '(example: --methods DLKcat UniKP "KinForm-H").'
            ),
        )
        parser.add_argument(
            "--handle-long-sequences",
            choices=["truncate", "skip"],
            default="truncate",
            help='How task pipeline handles long sequences (default: "truncate").',
        )
        parser.add_argument(
            "--api-base-url",
            type=str,
            default=os.getenv("WEBKINPRED_API_BASE_URL", DEFAULT_API_BASE_URL),
            help=(
                "Base URL for the v1 API (default: env WEBKINPRED_API_BASE_URL "
                f"or {DEFAULT_API_BASE_URL})."
            ),
        )
        parser.add_argument(
            "--api-key",
            type=str,
            default=os.getenv("WEBKINPRED_API_KEY"),
            help=(
                "Bearer API key (default: env WEBKINPRED_API_KEY). "
                "Required for authenticated submit/status endpoints."
            ),
        )
        parser.add_argument(
            "--poll-seconds",
            type=float,
            default=5.0,
            help="Polling interval in seconds for /status/ (default: 5).",
        )
        parser.add_argument(
            "--timeout-seconds",
            type=int,
            default=7200,
            help="Max wait per job before timing out (default: 7200).",
        )
        parser.add_argument(
            "--request-timeout-seconds",
            type=float,
            default=60.0,
            help="HTTP request timeout in seconds (default: 60).",
        )

    def handle(self, *args, **options):
        num_reactions = options["num_reactions"]
        num_proteins = options["num_proteins"]
        avg_seq_len = options["avg_seq_len"]
        max_seq_len = options["max_seq_len"]
        testmode = options["testmode"]
        methods_filter = options["methods"]
        handle_long_sequences = options["handle_long_sequences"]
        api_base_url = options["api_base_url"].rstrip("/")
        api_key = options["api_key"]
        poll_seconds = options["poll_seconds"]
        timeout_seconds = options["timeout_seconds"]
        request_timeout_seconds = options["request_timeout_seconds"]

        if testmode:
            num_reactions = 10
            num_proteins = 1
            self.stdout.write(
                "Test mode enabled: overriding workload to 10 reactions and 1 unique protein."
            )

        self._validate_generation_parameters(
            num_reactions=num_reactions,
            num_proteins=num_proteins,
            avg_seq_len=avg_seq_len,
            max_seq_len=max_seq_len,
        )

        if not api_key:
            raise CommandError(
                "Missing API key. Pass --api-key or set WEBKINPRED_API_KEY."
            )
        if poll_seconds <= 0:
            raise CommandError("--poll-seconds must be > 0.")
        if timeout_seconds <= 0:
            raise CommandError("--timeout-seconds must be > 0.")
        if request_timeout_seconds <= 0:
            raise CommandError("--request-timeout-seconds must be > 0.")

        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )
        kcat_methods = self._fetch_kcat_methods(
            session=session,
            api_base_url=api_base_url,
            request_timeout_seconds=request_timeout_seconds,
        )

        if methods_filter:
            requested = set(methods_filter)
            unknown = sorted(requested - set(kcat_methods))
            if unknown:
                raise CommandError(
                    "Unknown or non-kcat method key(s) for this API: "
                    + ", ".join(unknown)
                )
            selected_keys = sorted(
                key for key in requested if key in kcat_methods
            )
        else:
            selected_keys = sorted(kcat_methods)

        run_seed = options["seed"]
        if run_seed is None:
            run_seed = secrets.randbits(64)

        self.stdout.write(
            f"Benchmarking {len(selected_keys)} kcat method(s): {', '.join(selected_keys)}"
        )
        self.stdout.write(
            "Synthetic workload: "
            f"{num_reactions} reactions, {num_proteins} unique proteins, "
            f"avg length {avg_seq_len}, max length {max_seq_len}"
        )
        self.stdout.write(f"API base URL: {api_base_url}")
        self.stdout.write(
            "Reporting metric: computeSeconds from /api/v1/status/ "
            "(queueSeconds ignored)."
        )
        self.stdout.write(
            "Projected quota cost: "
            f"{len(selected_keys) * 2 * num_reactions} rows "
            "(2 jobs per method)."
        )
        self.stdout.write(
            f"Seed: {run_seed} "
            f"({'fixed' if options['seed'] is not None else 'auto-generated'})"
        )
        self.stdout.write(
            "Uncached run uses fresh proteins per method; cached run repeats "
            "the same proteins for that method."
        )
        self.stdout.write(
            "Cross-method overlap is disabled: protein sequences are unique across methods "
            "to avoid shared PLM cache effects."
        )
        self.stdout.write("")

        results: list[MethodBenchmarkResult] = []
        used_sequences: set[str] = set()

        for idx, method_key in enumerate(selected_keys, start=1):
            self.stdout.write(f"[{idx}/{len(selected_keys)}] {method_key}")
            method_seed = self._derive_method_seed(run_seed, method_key)
            rng = random.Random(method_seed)

            proteins = self._generate_unique_proteins(
                rng=rng,
                num_proteins=num_proteins,
                avg_seq_len=avg_seq_len,
                max_seq_len=max_seq_len,
                forbidden_sequences=used_sequences,
            )
            used_sequences.update(proteins)
            df_input = self._build_reaction_dataframe(
                proteins=proteins,
                num_reactions=num_reactions,
            )

            uncached_result = self._run_single_benchmark(
                session=session,
                api_base_url=api_base_url,
                method_key=method_key,
                input_df=df_input,
                handle_long_sequences=handle_long_sequences,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
                request_timeout_seconds=request_timeout_seconds,
            )
            cached_result = self._run_single_benchmark(
                session=session,
                api_base_url=api_base_url,
                method_key=method_key,
                input_df=df_input,
                handle_long_sequences=handle_long_sequences,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
                request_timeout_seconds=request_timeout_seconds,
            )

            method_result = MethodBenchmarkResult(
                method_key=method_key,
                uncached=uncached_result,
                cached=cached_result,
            )
            results.append(method_result)

            self.stdout.write(self._format_method_summary(method_result))
            self.stdout.write("")

        self.stdout.write("Final summary")
        self.stdout.write("-" * 72)
        for result in results:
            self.stdout.write(self._format_method_summary(result))

        success_count = sum(
            1
            for r in results
            if r.uncached.status == "Completed" and r.cached.status == "Completed"
        )
        self.stdout.write("-" * 72)
        self.stdout.write(
            f"Completed successfully for {success_count}/{len(results)} method(s)."
        )

    def _validate_generation_parameters(
        self,
        *,
        num_reactions: int,
        num_proteins: int,
        avg_seq_len: int,
        max_seq_len: int,
    ) -> None:
        if num_reactions <= 0:
            raise CommandError("--num-reactions must be > 0.")
        if num_proteins <= 0:
            raise CommandError("--num-proteins must be > 0.")
        if avg_seq_len <= 0:
            raise CommandError("--avg-seq-len must be > 0.")
        if max_seq_len <= 0:
            raise CommandError("--max-seq-len must be > 0.")
        if avg_seq_len > max_seq_len:
            raise CommandError("--avg-seq-len cannot be greater than --max-seq-len.")

        # Ensure we can have one max-length sequence while preserving the target average.
        target_total = num_proteins * avg_seq_len
        if num_proteins > 1:
            min_total_with_one_max = max_seq_len + (num_proteins - 1)
            if target_total < min_total_with_one_max:
                raise CommandError(
                    "Incompatible sequence settings. "
                    "Given --num-proteins, --avg-seq-len and --max-seq-len, "
                    "it is not possible to include one max-length sequence "
                    "while keeping the requested average."
                )

    def _derive_method_seed(self, run_seed: int, method_key: str) -> int:
        digest = hashlib.sha256(f"{run_seed}:{method_key}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _generate_unique_proteins(
        self,
        *,
        rng: random.Random,
        num_proteins: int,
        avg_seq_len: int,
        max_seq_len: int,
        forbidden_sequences: set[str] | None = None,
    ) -> list[str]:
        lengths = self._generate_protein_lengths(
            rng=rng,
            num_proteins=num_proteins,
            avg_seq_len=avg_seq_len,
            max_seq_len=max_seq_len,
        )

        blocked = forbidden_sequences or set()
        proteins: list[str] = []
        seen: set[str] = set()
        for length in lengths:
            for _ in range(1000):
                seq = "".join(rng.choices(AA_ALPHABET, k=length))
                if seq not in seen and seq not in blocked:
                    seen.add(seq)
                    proteins.append(seq)
                    break
            else:
                raise RuntimeError(
                    "Could not generate a unique protein sequence after many attempts."
                )
        return proteins

    def _generate_protein_lengths(
        self,
        *,
        rng: random.Random,
        num_proteins: int,
        avg_seq_len: int,
        max_seq_len: int,
    ) -> list[int]:
        if num_proteins == 1:
            # In test mode we often use a single unique protein; preserve the
            # requested average length directly in this edge case.
            return [avg_seq_len]

        target_total = num_proteins * avg_seq_len
        lengths = [avg_seq_len] * num_proteins
        lengths[0] = max_seq_len  # Ensure one max-length protein exists.

        # Rebalance to preserve exact average length.
        current_total = sum(lengths)
        delta = current_total - target_total
        idx = 1
        while delta > 0:
            if idx >= num_proteins:
                idx = 1
            if lengths[idx] > 1:
                lengths[idx] -= 1
                delta -= 1
            idx += 1

        idx = 1
        while delta < 0:
            if idx >= num_proteins:
                idx = 1
            if lengths[idx] < max_seq_len:
                lengths[idx] += 1
                delta += 1
            idx += 1

        # Add mild variability while keeping sum constant and staying in bounds.
        non_max_indices = list(range(1, num_proteins))
        if len(non_max_indices) >= 2:
            for _ in range(num_proteins * 8):
                src, dst = rng.sample(non_max_indices, 2)
                max_down = lengths[src] - 1
                max_up = max_seq_len - lengths[dst]
                if max_down <= 0 or max_up <= 0:
                    continue
                shift = rng.randint(1, min(5, max_down, max_up))
                lengths[src] -= shift
                lengths[dst] += shift

        return lengths

    def _build_reaction_dataframe(
        self,
        *,
        proteins: list[str],
        num_reactions: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        n_proteins = len(proteins)
        for i in range(num_reactions):
            substrate = SUBSTRATE_POOL[i % len(SUBSTRATE_POOL)]
            product = PRODUCT_POOL[i % len(PRODUCT_POOL)]
            rows.append(
                {
                    "Protein Sequence": proteins[i % n_proteins],
                    "Substrate": substrate,
                    "Substrates": substrate,
                    "Products": product,
                }
            )
        return pd.DataFrame(rows)

    def _run_single_benchmark(
        self,
        *,
        session: requests.Session,
        api_base_url: str,
        method_key: str,
        input_df: pd.DataFrame,
        handle_long_sequences: str,
        poll_seconds: float,
        timeout_seconds: int,
        request_timeout_seconds: float,
    ) -> SingleRunResult:
        payload = {
            "targets": ["kcat"],
            "methods": {"kcat": method_key},
            "handleLongSequences": handle_long_sequences,
            "useExperimental": False,
            "canonicalizeSubstrates": True,
            "data": input_df.to_dict(orient="records"),
        }

        submit_url = f"{api_base_url}/submit/"
        try:
            response = session.post(
                submit_url,
                json=payload,
                timeout=request_timeout_seconds,
            )
        except requests.RequestException as exc:
            return SingleRunResult(
                compute_seconds=None,
                status="Failed",
                error_message=f"Submit request failed: {exc}",
                public_id="",
            )

        if response.status_code >= 400:
            return SingleRunResult(
                compute_seconds=None,
                status="Failed",
                error_message=(
                    f"Submit failed ({response.status_code}): "
                    f"{self._extract_error_message(response)}"
                ),
                public_id="",
            )

        try:
            submit_data = response.json()
        except ValueError:
            return SingleRunResult(
                compute_seconds=None,
                status="Failed",
                error_message="Submit response was not valid JSON.",
                public_id="",
            )

        public_id = str(submit_data.get("jobId", "")).strip()
        if not public_id:
            return SingleRunResult(
                compute_seconds=None,
                status="Failed",
                error_message="Submit response did not include jobId.",
                public_id="",
            )

        return self._poll_job_until_terminal(
            session=session,
            api_base_url=api_base_url,
            public_id=public_id,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )

    def _poll_job_until_terminal(
        self,
        *,
        session: requests.Session,
        api_base_url: str,
        public_id: str,
        poll_seconds: float,
        timeout_seconds: int,
        request_timeout_seconds: float,
    ) -> SingleRunResult:
        deadline = time.monotonic() + timeout_seconds
        status_url = f"{api_base_url}/status/{public_id}/"
        last_error = ""
        last_status = "Pending"

        while True:
            if time.monotonic() > deadline:
                detail = f"Last status: {last_status}"
                if last_error:
                    detail += f" | Last poll error: {last_error}"
                return SingleRunResult(
                    compute_seconds=None,
                    status="Failed",
                    error_message=(
                        f"Timed out after {timeout_seconds}s waiting for job {public_id}. "
                        f"{detail}"
                    ),
                    public_id=public_id,
                )

            try:
                response = session.get(
                    status_url,
                    timeout=request_timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = str(exc)
                time.sleep(poll_seconds)
                continue

            if response.status_code >= 400:
                return SingleRunResult(
                    compute_seconds=None,
                    status="Failed",
                    error_message=(
                        f"Status failed ({response.status_code}) for job {public_id}: "
                        f"{self._extract_error_message(response)}"
                    ),
                    public_id=public_id,
                )

            try:
                status_data = response.json()
            except ValueError:
                return SingleRunResult(
                    compute_seconds=None,
                    status="Failed",
                    error_message=f"Status response for job {public_id} was not valid JSON.",
                    public_id=public_id,
                )

            status = str(status_data.get("status", "Unknown")).strip()
            last_status = status or "Unknown"
            compute_seconds = self._to_optional_float(status_data.get("computeSeconds"))
            error_message = str(status_data.get("error", "")).strip()

            if status in {"Completed", "Failed"}:
                return SingleRunResult(
                    compute_seconds=compute_seconds,
                    status=status,
                    error_message=error_message,
                    public_id=public_id,
                )

            time.sleep(poll_seconds)

    def _fetch_kcat_methods(
        self,
        *,
        session: requests.Session,
        api_base_url: str,
        request_timeout_seconds: float,
    ) -> list[str]:
        methods_url = f"{api_base_url}/methods/"
        try:
            response = session.get(methods_url, timeout=request_timeout_seconds)
        except requests.RequestException as exc:
            raise CommandError(f"Could not fetch methods from {methods_url}: {exc}")

        if response.status_code >= 400:
            raise CommandError(
                f"Failed to fetch methods ({response.status_code}): "
                f"{self._extract_error_message(response)}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise CommandError(f"/methods response was not valid JSON: {exc}")

        kcat_entries = data.get("methods", {}).get("kcat", [])
        method_ids = sorted(
            {
                str(entry.get("id")).strip()
                for entry in kcat_entries
                if isinstance(entry, dict) and str(entry.get("id", "")).strip()
            }
        )
        if not method_ids:
            raise CommandError(
                "No kcat-capable methods were returned by the API /methods endpoint."
            )
        return method_ids

    def _extract_error_message(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            text = response.text.strip()
            return text[:500] if text else "No error details provided."

        if isinstance(payload, dict):
            error = payload.get("error")
            if error:
                return str(error)
        return str(payload)[:500]

    def _to_optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _format_method_summary(self, result: MethodBenchmarkResult) -> str:
        uncached = result.uncached
        cached = result.cached

        uncached_time = self._format_compute_seconds(uncached.compute_seconds)
        cached_time = self._format_compute_seconds(cached.compute_seconds)
        base = (
            f"{result.method_key} - "
            f"{uncached_time} compute (not cached), "
            f"{cached_time} compute (cached)"
        )

        if uncached.status == "Completed" and cached.status == "Completed":
            return base

        details: list[str] = []
        if uncached.status != "Completed":
            msg = uncached.error_message or "Unknown failure"
            details.append(f"uncached failed: {msg}")
        if cached.status != "Completed":
            msg = cached.error_message or "Unknown failure"
            details.append(f"cached failed: {msg}")

        return base + " | " + " | ".join(details)

    def _format_compute_seconds(self, seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        return f"{seconds:.2f}s"
