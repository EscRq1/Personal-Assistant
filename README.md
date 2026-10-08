INSTALL REQUIRED PYTHON DEPENDENCIES
=
pip install -r requirements.txt

DOWNLOAD ESSENTIAL PACKAGE WHEELS FOR THE SANDBOX MODULE
=
```bash
set -euo pipefail

catalog="./catalog.json"
destination="./downloaded-wheels"

requirements="$(mktemp)"
trap 'rm -f "$requirements"' EXIT

python3 - "$catalog" "$requirements" <<'PY'
import json
import sys
from pathlib import Path

catalog = json.loads(Path(sys.argv[1]).read_text())
packages = sorted({
    (wheel["name"], wheel["version"])
    for wheel in catalog["wheels"]
})

Path(sys.argv[2]).write_text(
    "".join(f"{name}=={version}\n" for name, version in packages)
)

print(f"Downloading {len(packages)} packages.")
PY

mkdir -p "$destination"

python3 -m pip download \
    --index-url https://pypi.org/simple \
    --only-binary=:all: \
    --no-deps \
    --requirement "$requirements" \
    --dest "$destination"

echo "Wheels downloaded to $destination"
```
