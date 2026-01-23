"""Local data server for Altair charts."""

import http.server
import os
import threading
import hashlib
import pandas as pd
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import streamlit as st

    logger.debug("streamlit imported in data_server")
except ImportError:
    st = None
    logger.debug("streamlit NOT imported in data_server")

# Dummy decorator if streamlit is missing
if st is None:

    def cache_resource(**kwargs: Any) -> Any:  # noqa: ANN401
        """Dummy cache_resource decorator."""

        def decorator(f: Any) -> Any:  # noqa: ANN401
            return f

        return decorator

    # Create a dummy object to hold the decorator if st is None
    class DummySt:
        """Dummy Streamlit module for when Streamlit is not installed."""

        cache_resource = cache_resource

    st_module = DummySt()
else:
    st_module = st


class LocalDataServer:
    """Server for hosting local data files for Altair charts."""

    # Remove class-level singleton in favor of external cache mechanisms
    # but keep a fallback for non-streamlit usage
    _fallback_instance: Optional["LocalDataServer"] = None
    _lock = threading.Lock()

    def __init__(self, port: int = 8001, data_dir: str = ".app_data/served_data") -> None:
        """Initialize the data server."""
        self.port = port
        self.data_dir = data_dir
        self.base_url = f"http://localhost:{port}"
        self._server = None
        self._thread = None

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    @st_module.cache_resource(show_spinner=False)
    def _get_cached_instance(port: int) -> "LocalDataServer":
        """Cached instance creator."""
        instance = LocalDataServer(port=port)
        try:
            instance.start()
        except OSError as e:
            logger.warning(f"Could not bind to port {port}: {e}. Assuming existing server is active.")
            pass
        return instance

    @classmethod
    def get_instance(cls, port: int = 8001) -> "LocalDataServer":
        """Singleton accessor using Streamlit cache if available, else class variable."""
        if st:
            logger.debug(f"Using st.cache_resource for port {port}")
            return cls._get_cached_instance(port)

        else:
            logger.debug(f"Using fallback instance for port {port}")
            # Fallback for tests / non-streamlit usage
            if cls._fallback_instance is None:
                with cls._lock:
                    if cls._fallback_instance is None:
                        instance = cls(port=port)
                        try:
                            instance.start()
                        except OSError as e:
                            logger.warning(f"Could not bind to port {port} in fallback: {e}")
                            pass
                        cls._fallback_instance = instance
            return cls._fallback_instance

    def start(self) -> None:
        """Start the HTTP server in a daemon thread."""
        if self._server:
            return

        class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                super().end_headers()

        handler = lambda *args, **kwargs: CORSRequestHandler(*args, directory=self.data_dir, **kwargs)

        # We catch the error in __init__/_create_instance usually, but here we can be explicit
        self._server = http.server.ThreadingHTTPServer(("localhost", self.port), handler)
        logger.info(f"Starting local data server at {self.base_url} serving {self.data_dir}")

        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True
        self._thread.start()

    def save_data(self, df: pd.DataFrame, name_prefix: str = "data") -> str:
        """
        Save DataFrame as JSON and return the local URL.
        Uses hash of content to avoid duplicates and unnecessary writes.
        """
        # Create a stable JSON representation
        data_json = df.to_json(orient="records", date_format="iso")
        if data_json is None:
            return ""

        # Hash content to create unique filename
        content_hash = hashlib.md5(data_json.encode("utf-8")).hexdigest()
        filename = f"{name_prefix}_{content_hash}.json"
        comp_path = os.path.join(self.data_dir, filename)

        # Write only if doesn't exist
        if not os.path.exists(comp_path):
            with open(comp_path, "w", encoding="utf-8") as f:
                f.write(data_json)

        return f"{self.base_url}/{filename}"


def get_data_server() -> LocalDataServer:
    """Helper to get the singleton instance."""
    return LocalDataServer.get_instance()
