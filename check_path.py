from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
import platform


print(f"Platform: {platform.system()}")

