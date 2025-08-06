# ---------- Imports ----------
import uvicorn
import os
import sys
import datetime
import platform
import hashlib
import functools
import subprocess
import time
import shutil  # ✅ New: for checking binary availability

# Ensure all print statements flush immediately
print = functools.partial(print, flush=True)

# ---------- Startup Metadata ----------
def log_startup_metadata():
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] 🚀 Starting MedAI backend container")

    print("🔍 Environment:")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Working Directory: {os.getcwd()}")

    lockfile_path = os.path.join(os.getcwd(), "requirements.lock.txt")
    if os.path.exists(lockfile_path):
        with open(lockfile_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        print(f"📦 requirements.lock.txt SHA256: {sha256}")
    else:
        print("⚠️ requirements.lock.txt not found")

# ---------- Banner ----------
def show_startup_banner():
    print(r"""
   __  __      _    ____ ___    _    ____  
  |  \/  | ___| |__/ ___|_ _|  / \  |  _ \ 
  | |\/| |/ _ \ '_ \___ \| |  / _ \ | | | |
  | |  | |  __/ |_) |__) | | / ___ \| |_| |
  |_|  |_|\___|_.__/____/___/_/   \_\____/ 
    """)

# ---------- Service Readiness ----------
def wait_for_service(name, command, retries=10, delay=2):
    print(f"🔄 Checking {name} readiness...")

    if not shutil.which(command[0]):
        print(f"⚠️ {name} check skipped: binary '{command[0]}' not found in PATH")
        return

    for attempt in range(retries):
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {name} is ready")
            return
        except subprocess.CalledProcessError:
            print(f"⏳ {name} not ready (attempt {attempt + 1}/{retries})")
            time.sleep(delay)
    raise RuntimeError(f"❌ {name} failed to respond after {retries} attempts")

# ---------- Agent Diagnostics ----------
def check_agent_readiness():
    print("🤖 Agent readiness:")
    # TODO: Replace with actual diagnostic once model loading is modularized
    print("  - SkinAgent: ✅ (mocked)")
    print("  - RandomForestAgent: ✅ (mocked)")

# ---------- Entry ----------
if __name__ == "__main__":
    log_startup_metadata()
    show_startup_banner()

    wait_for_service("Redis", ["redis-cli", "ping"])
    wait_for_service("Postgres", ["pg_isready", "-h", "postgres", "-p", "5432"])

    check_agent_readiness()

    print("🌐 Launching FastAPI on 0.0.0.0:8000")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
