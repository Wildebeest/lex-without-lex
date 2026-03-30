from __future__ import annotations

from pathlib import Path

from b2sdk.v2 import B2Api, InMemoryAccountInfo


class B2Storage:
    def __init__(self, key_id: str, app_key: str, bucket_name: str):
        self._bucket_name = bucket_name
        info = InMemoryAccountInfo()
        self._api = B2Api(info)
        self._api.authorize_account("production", key_id, app_key)
        self._bucket = self._api.get_bucket_by_name(bucket_name)

    def upload_episode(self, local_path: Path, episode_guid: str) -> tuple[str, str]:
        """Upload file to B2, return (download_url, file_name)."""
        safe_name = episode_guid.replace("/", "_").replace(":", "_")
        file_name = f"episodes/{safe_name}.mp3"

        self._bucket.upload_local_file(
            local_file=str(local_path),
            file_name=file_name,
        )
        return self._bucket.get_download_url(file_name), file_name

    def get_download_auth_url(self, file_name: str, duration_seconds: int = 604800) -> str:
        """Generate an authorized download URL valid for the given duration (default 7 days)."""
        # file_name_prefix must match the file — use the directory prefix
        prefix = "/".join(file_name.split("/")[:-1]) + "/"
        auth_token = self._bucket.get_download_authorization(
            file_name_prefix=prefix,
            valid_duration_in_seconds=duration_seconds,
        )
        base_url = self._bucket.get_download_url(file_name)
        return f"{base_url}?Authorization={auth_token}"

    def file_exists(self, episode_guid: str) -> bool:
        """Check if episode already uploaded."""
        safe_name = episode_guid.replace("/", "_").replace(":", "_")
        file_name = f"episodes/{safe_name}.mp3"
        try:
            self._bucket.get_file_info_by_name(file_name)
            return True
        except Exception:
            return False
