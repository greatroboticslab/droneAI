import os
import sys
import subprocess
import importlib.util
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DB_PATH = REPO_DIR / "db" / "droneai.sqlite"


def is_package_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def install_package(package_name: str):
    """
    Tries to install a package into the current Python environment.
    Returns (ok: bool, message: str)
    """
    try:
        cmd = [sys.executable, "-m", "pip", "install", package_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode == 0:
            return True, f"Installed {package_name} successfully."
        else:
            err = (result.stderr or result.stdout or "").strip()
            return False, f"Install failed for {package_name}: {err}"
    except Exception as e:
        return False, f"Install exception for {package_name}: {e}"


def db_exists() -> bool:
    return DB_PATH.exists()


def db_size_bytes() -> int:
    if DB_PATH.exists():
        return DB_PATH.stat().st_size
    return 0


def mqtt_package_status():
    ok = is_package_installed("paho")
    return {
        "installed": ok,
        "module": "paho",
        "package_name": "paho-mqtt",
    }


def overall_agent_status(mqtt_mgr=None):
    mqtt_pkg = mqtt_package_status()

    return {
        "mqtt_package_installed": mqtt_pkg["installed"],
        "db_exists": db_exists(),
        "db_size_bytes": db_size_bytes(),
        "mqtt_enabled": bool(getattr(mqtt_mgr, "enabled", False)) if mqtt_mgr else False,
        "mqtt_connected": bool(getattr(mqtt_mgr, "connected", False)) if mqtt_mgr else False,
        "mqtt_host": getattr(mqtt_mgr, "host", "") if mqtt_mgr else "",
        "mqtt_port": getattr(mqtt_mgr, "port", "") if mqtt_mgr else "",
        "mqtt_topic_prefix": getattr(mqtt_mgr, "topic_prefix", "") if mqtt_mgr else "",
    }
