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

    def upload_episode(self, local_path: Path, episode_guid: str) -> str:
        """Upload file to B2, return public URL."""
        safe_name = episode_guid.replace("/", "_").replace(":", "_")
        file_name = f"episodes/{safe_name}.mp3"

        file_version = self._bucket.upload_local_file(
            local_file=str(local_path),
            file_name=file_name,
        )
        return self._bucket.get_download_url(file_name)

    def file_exists(self, episode_guid: str) -> bool:
        """Check if episode already uploaded."""
        safe_name = episode_guid.replace("/", "_").replace(":", "_")
        file_name = f"episodes/{safe_name}.mp3"
        try:
            self._bucket.get_file_info_by_name(file_name)
            return True
        except Exception:
            return False
