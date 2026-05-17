"""
nyayaeval.connectors.adaption_client — Adaptive Data SDK Client
=================================================================

Wraps the official ``adaption`` Python SDK for the Adaptive Data platform.
This is the core integration point for the hackathon — the Adaptive Data
Track requires meaningful usage of the platform's Ingest → Adapt → Evaluate
→ Export lifecycle.

The client manages the full dataset lifecycle:
    1. **Ingest**  — Upload legal document datasets (CSV/JSONL)
    2. **Adapt**   — Run multilingual adaptation (242 language support)
    3. **Evaluate** — Retrieve quality metrics (grade before/after)
    4. **Export**   — Download adapted datasets for HuggingFace/Kaggle

Also retains a lightweight ``translate()`` fallback for single-segment
translation when the full dataset workflow isn't needed (e.g., during
the correction loop).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import structlog
from adaption import Adaption

logger = structlog.get_logger(__name__)


class AdaptionAPIError(Exception):
    """Raised when the Adaption SDK returns an error."""


class AdaptiveDataClient:
    """
    Client for the Adaption Adaptive Data platform.

    Wraps the official SDK and provides high-level methods aligned
    with the NyayaEval pipeline stages.
    """

    def __init__(self, api_key: str, timeout: int = 120) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client: Adaption | None = None

    async def connect(self) -> None:
        """Initialize the Adaption SDK client."""
        self._client = Adaption(api_key=self._api_key)
        logger.info("adaption_client.connected")

    async def close(self) -> None:
        """Close the client (SDK is stateless, but matches our pattern)."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("adaption_client.closed")

    async def __aenter__(self) -> AdaptiveDataClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _ensure_connected(self) -> Adaption:
        if self._client is None:
            raise AdaptionAPIError("Client not connected. Call connect() first.")
        return self._client

    # ── Ingest ────────────────────────────────────────────────────────────

    def upload_dataset(
        self, file_path: str | Path, name: str | None = None
    ) -> dict[str, Any]:
        """
        Upload a local CSV/JSONL file as a new dataset.

        Args:
            file_path: Path to the file (.csv, .json, .jsonl, .parquet).
            name: Optional dataset name. Defaults to filename.

        Returns:
            Dict with dataset_id and upload metadata.
        """
        client = self._ensure_connected()
        path = Path(file_path)
        dataset_name = name or path.stem

        try:
            result = client.datasets.upload_file(
                path=str(path),
                name=dataset_name,
            )
            logger.info(
                "adaption_client.uploaded",
                dataset_id=result.id,
                name=dataset_name,
            )
            return {"dataset_id": result.id, "name": dataset_name, "status": result.status}
        except Exception as exc:
            raise AdaptionAPIError(f"Upload failed: {exc}") from exc

    def upload_from_records(
        self, records: list[dict[str, Any]], name: str = "nyayaeval_legal_docs"
    ) -> dict[str, Any]:
        """
        Upload in-memory records as a JSONL dataset.

        Creates a temporary JSONL file and uploads it via the SDK.

        Args:
            records: List of dicts to upload as JSONL rows.
            name: Dataset name on the platform.

        Returns:
            Dict with dataset_id and upload metadata.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_path = f.name

        try:
            return self.upload_dataset(tmp_path, name=name)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Adapt ─────────────────────────────────────────────────────────────

    def run_adaptation(
        self,
        dataset_id: str,
        source_column: str = "source_text",
        target_column: str = "adapted_text",
        estimate: bool = False,
    ) -> dict[str, Any]:
        """
        Run the adaptation pipeline on an uploaded dataset.

        Args:
            dataset_id: ID from the upload step.
            source_column: Column containing source text.
            target_column: Column for adapted output.
            estimate: If True, returns cost estimate without running.

        Returns:
            Dict with job metadata (status, estimated cost, etc.).
        """
        client = self._ensure_connected()

        try:
            result = client.datasets.run(
                dataset_id,
                column_mapping={"prompt": source_column},
                estimate=estimate,
            )
            logger.info(
                "adaption_client.run_started",
                dataset_id=dataset_id,
                estimate=estimate,
            )
            return {
                "dataset_id": dataset_id,
                "status": getattr(result, "status", "running"),
                "estimated_credits": getattr(result, "estimated_credits", None),
            }
        except Exception as exc:
            raise AdaptionAPIError(f"Adaptation run failed: {exc}") from exc

    # ── Wait & Status ─────────────────────────────────────────────────────

    def wait_for_completion(
        self, dataset_id: str, timeout: int = 3600, poll_interval: int = 10
    ) -> dict[str, Any]:
        """
        Poll until the dataset adaptation job completes.

        Args:
            dataset_id: Dataset to monitor.
            timeout: Max wait time in seconds.
            poll_interval: Seconds between status checks.

        Returns:
            Final dataset status dict.
        """
        client = self._ensure_connected()
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            try:
                status = client.datasets.get_status(dataset_id)
                current = getattr(status, "status", "unknown")
                logger.debug(
                    "adaption_client.polling",
                    dataset_id=dataset_id,
                    status=current,
                    elapsed_s=round(time.monotonic() - start),
                )
                if current in ("ready", "failed", "completed"):
                    return {"dataset_id": dataset_id, "status": current}
                time.sleep(poll_interval)
            except Exception as exc:
                logger.warning("adaption_client.poll_error", error=str(exc))
                time.sleep(poll_interval)

        raise AdaptionAPIError(
            f"Dataset {dataset_id} did not complete within {timeout}s"
        )

    # ── Evaluate ──────────────────────────────────────────────────────────

    def get_evaluation(self, dataset_id: str) -> dict[str, Any]:
        """
        Retrieve quality evaluation metrics from Adaption.

        Returns grade_before, grade_after, and per-metric breakdowns.

        Args:
            dataset_id: Dataset to evaluate.

        Returns:
            Evaluation results dict.
        """
        client = self._ensure_connected()

        try:
            result = client.datasets.get_evaluation(dataset_id)
            eval_data = {
                "dataset_id": dataset_id,
                "evaluation": result.model_dump() if hasattr(result, "model_dump") else str(result),
            }
            logger.info("adaption_client.evaluation_retrieved", dataset_id=dataset_id)
            return eval_data
        except Exception as exc:
            logger.warning("adaption_client.evaluation_failed", error=str(exc))
            return {"dataset_id": dataset_id, "evaluation": None, "error": str(exc)}

    # ── Export / Download ─────────────────────────────────────────────────

    def download_dataset(
        self,
        dataset_id: str,
        output_path: str | Path,
        file_format: str = "jsonl",
    ) -> Path:
        """
        Download the adapted dataset to a local file.

        Args:
            dataset_id: Dataset to download.
            output_path: Local path to save the file.
            file_format: Output format (jsonl, csv, parquet).

        Returns:
            Path to the downloaded file.
        """
        client = self._ensure_connected()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = client.datasets.download(dataset_id, file_format=file_format)
            # The SDK returns a streaming response — write to file
            with open(out, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
            logger.info(
                "adaption_client.downloaded",
                dataset_id=dataset_id,
                path=str(out),
            )
            return out
        except Exception as exc:
            raise AdaptionAPIError(f"Download failed: {exc}") from exc

    # ── Publish to HuggingFace/Kaggle ─────────────────────────────────────

    def publish_to_huggingface(
        self, dataset_id: str, repo_name: str
    ) -> dict[str, Any]:
        """
        Publish the adapted dataset to HuggingFace Hub.

        Note: This endpoint may return 501 (not yet implemented in SDK 0.3.1).
        Falls back to manual upload instructions if so.

        Args:
            dataset_id: Dataset to publish.
            repo_name: HuggingFace repository name.

        Returns:
            Publication result or fallback instructions.
        """
        client = self._ensure_connected()

        try:
            result = client.datasets.publish(
                dataset_id,
                target="huggingface",
                target_spec={"repo": repo_name},
            )
            return {"status": "published", "target": "huggingface", "repo": repo_name}
        except Exception as exc:
            logger.warning(
                "adaption_client.publish_not_available",
                error=str(exc),
                fallback="Manual upload required",
            )
            return {
                "status": "manual_required",
                "target": "huggingface",
                "instructions": (
                    f"Download the dataset and manually upload to HuggingFace: "
                    f"huggingface-cli upload {repo_name} <downloaded_file>"
                ),
            }

    # ── Health Check ──────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Verify Adaption API connectivity."""
        if self._client is None:
            return {"status": "disconnected"}
        try:
            datasets = self._client.datasets.list()
            return {
                "status": "healthy",
                "datasets_count": len(getattr(datasets, "data", [])),
            }
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}
