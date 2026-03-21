"""
Benchmark kcat inference latency for all registered kcat-capable methods.

For each method, this command:
1) Generates 1,000 reactions from 100 unique proteins (defaults), where
   proteins have average length 400 and max length 1,000.
2) Runs one prediction job with fresh proteins (uncached embedding path).
3) Runs a second prediction job with the exact same proteins (cached path).
4) Prints a readable per-method summary:
   DLKcat - 100.00s (not cached), 10.00s (cached)

The command uses the same backend execution path as production jobs by calling
`api.tasks.run_prediction` directly.
"""

from __future__ import annotations

import hashlib
import random
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from api.methods.registry import all_methods
from api.models import Job
from api.tasks import run_prediction

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


@dataclass
class SingleRunResult:
    seconds: float
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
            "--keep-jobs",
            action="store_true",
            help="Keep benchmark Job records and media/jobs/<public_id> artifacts.",
        )

    def handle(self, *args, **options):
        num_reactions = options["num_reactions"]
        num_proteins = options["num_proteins"]
        avg_seq_len = options["avg_seq_len"]
        max_seq_len = options["max_seq_len"]
        methods_filter = options["methods"]
        handle_long_sequences = options["handle_long_sequences"]
        keep_jobs = options["keep_jobs"]

        self._validate_generation_parameters(
            num_reactions=num_reactions,
            num_proteins=num_proteins,
            avg_seq_len=avg_seq_len,
            max_seq_len=max_seq_len,
        )

        registry = all_methods()
        kcat_methods = {
            key: desc
            for key, desc in registry.items()
            if "kcat" in desc.supports
        }

        if not kcat_methods:
            raise CommandError("No kcat-capable methods are registered.")

        if methods_filter:
            requested = set(methods_filter)
            unknown = sorted(requested - set(registry.keys()))
            if unknown:
                raise CommandError(
                    "Unknown method key(s): " + ", ".join(unknown)
                )
            non_kcat = sorted(
                key for key in requested if key not in kcat_methods
            )
            if non_kcat:
                raise CommandError(
                    "Method(s) do not support kcat: " + ", ".join(non_kcat)
                )
            selected_keys = sorted(requested)
        else:
            selected_keys = sorted(kcat_methods.keys())

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
        self.stdout.write(
            f"Seed: {run_seed} "
            f"({'fixed' if options['seed'] is not None else 'auto-generated'})"
        )
        self.stdout.write(
            "Uncached run uses fresh proteins per method; cached run repeats the same proteins."
        )
        self.stdout.write("")

        results: list[MethodBenchmarkResult] = []

        for idx, method_key in enumerate(selected_keys, start=1):
            self.stdout.write(f"[{idx}/{len(selected_keys)}] {method_key}")
            method_seed = self._derive_method_seed(run_seed, method_key)
            rng = random.Random(method_seed)

            proteins = self._generate_unique_proteins(
                rng=rng,
                num_proteins=num_proteins,
                avg_seq_len=avg_seq_len,
                max_seq_len=max_seq_len,
            )
            df_input = self._build_reaction_dataframe(
                proteins=proteins,
                num_reactions=num_reactions,
            )

            uncached_result = self._run_single_benchmark(
                method_key=method_key,
                input_df=df_input,
                handle_long_sequences=handle_long_sequences,
            )
            cached_result = self._run_single_benchmark(
                method_key=method_key,
                input_df=df_input,
                handle_long_sequences=handle_long_sequences,
            )

            if not keep_jobs:
                self._cleanup_job(uncached_result.public_id)
                self._cleanup_job(cached_result.public_id)

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
    ) -> list[str]:
        lengths = self._generate_protein_lengths(
            rng=rng,
            num_proteins=num_proteins,
            avg_seq_len=avg_seq_len,
            max_seq_len=max_seq_len,
        )

        proteins: list[str] = []
        seen: set[str] = set()
        for length in lengths:
            for _ in range(1000):
                seq = "".join(rng.choices(AA_ALPHABET, k=length))
                if seq not in seen:
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
        method_key: str,
        input_df: pd.DataFrame,
        handle_long_sequences: str,
    ) -> SingleRunResult:
        job = Job(
            prediction_type="kcat",
            kcat_method=method_key,
            km_method=None,
            kcat_km_method=None,
            status="Pending",
            handle_long_sequences=handle_long_sequences,
            requested_rows=len(input_df),
            ip_address="",  # empty = quota refund hooks become no-op
        )
        job.save()

        job_dir = Path(settings.MEDIA_ROOT) / "jobs" / str(job.public_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        input_df.to_csv(job_dir / "input.csv", index=False)

        start = time.perf_counter()
        unexpected_error = ""
        try:
            run_prediction(
                public_id=job.public_id,
                method_key=method_key,
                target="kcat",
                experimental_results=[],
            )
        except Exception as exc:  # defensive guard to keep the benchmark loop running
            unexpected_error = str(exc)
        elapsed = time.perf_counter() - start

        job.refresh_from_db()
        status = job.status or "Failed"
        error_message = (job.error_message or "").strip()
        if unexpected_error and not error_message:
            error_message = unexpected_error
            status = "Failed"

        return SingleRunResult(
            seconds=elapsed,
            status=status,
            error_message=error_message,
            public_id=job.public_id,
        )

    def _cleanup_job(self, public_id: str) -> None:
        Job.objects.filter(public_id=public_id).delete()
        job_dir = Path(settings.MEDIA_ROOT) / "jobs" / str(public_id)
        shutil.rmtree(job_dir, ignore_errors=True)

    def _format_method_summary(self, result: MethodBenchmarkResult) -> str:
        uncached = result.uncached
        cached = result.cached

        base = (
            f"{result.method_key} - "
            f"{uncached.seconds:.2f}s (not cached), "
            f"{cached.seconds:.2f}s (cached)"
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
