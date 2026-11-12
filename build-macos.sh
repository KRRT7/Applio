#!/bin/sh
# Build Applio as a single macOS executable with Nuitka.
#
#   ./build-macos.sh            -> build/nuitka/Applio
#
# The executable unpacks once into ~/Library/Caches/Applio/<version> and reuses
# that folder on later launches (cached onefile mode), so only the first start
# pays for unpacking. User data lives in ~/Library/Application Support/Applio;
# the voice models listed in BUNDLED_MODELS are used read-only from the program.
set -eu
cd "$(dirname "$0")"

VERSION=$(python3 -c "import json; print(json.load(open('assets/config_template.json'))['version'])")

for f in rvc/models/embedders/contentvec/pytorch_model.bin rvc/models/embedders/contentvec/config.json \
         rvc/models/predictors/rmvpe.pt rvc/models/predictors/fcpe.pt rvc/models/predictors/swift.onnx; do
    if [ ! -s "$f" ]; then
        echo "Missing $f; downloading the bundled models first."
        uv run core.py prerequisites --models --no-exe
        break
    fi
done

# Voice models shipped inside the build, listed read-only next to the user's own.
# Their .npy training features are not used by Realtime and are left out (see below).
BUNDLED_MODELS="angelika dona martu"
MODEL_ARGS=""
for m in $BUNDLED_MODELS; do
    if ! ls "logs/$m"/*.pth >/dev/null 2>&1; then
        echo "Voice model logs/$m (with a .pth file) is missing; cannot bundle it." >&2
        exit 1
    fi
    MODEL_ARGS="$MODEL_ARGS --include-data-dir=logs/$m=bundled/logs/$m"
done

uv sync --locked --group build

exec uv run --group build python -m nuitka app.py \
    --mode=onefile \
    --output-dir=build/nuitka \
    --output-filename=Applio \
    --product-name=Applio \
    --company-name=Applio \
    --product-version="$VERSION" \
    --file-version="$VERSION" \
    --onefile-tempdir-spec="{CACHE_DIR}/Applio/{VERSION}" \
    --onefile-cache-mode=cached \
    --module-parameter=torch-disable-jit=yes \
    --module-parameter=numba-disable-jit=yes \
    --noinclude-pytest-mode=nofollow \
    --noinclude-setuptools-mode=nofollow \
    --noinclude-IPython-mode=nofollow \
    --nofollow-import-to=matplotlib \
    --include-package=assets \
    --include-package=tabs \
    --include-package=rvc \
    --nofollow-import-to=assets.build_update \
    --nofollow-import-to=assets.i18n.scan \
    --include-package-data=gradio_client \
    --include-package-data=safehttpx \
    --include-package-data=groovy \
    --include-package-data=torchfcpe \
    --include-data-files=assets/config_template.json=assets/config_template.json \
    --include-data-files=assets/ecosystem.json=assets/ecosystem.json \
    --include-data-files=assets/update_policy.json=assets/update_policy.json \
    --include-data-files=assets/ICON.ico=assets/ICON.ico \
    --include-data-files=assets/applio-logo.svg=assets/applio-logo.svg \
    --include-data-dir=assets/i18n/languages=assets/i18n/languages \
    --include-data-dir=assets/ui=assets/ui \
    --include-data-files=assets/themes/theme_list.json=assets/themes/theme_list.json \
    --include-data-files=tabs/realtime/main.js=tabs/realtime/main.js \
    --include-data-dir=rvc/models/embedders/contentvec=rvc/models/embedders/contentvec \
    --include-data-dir=rvc/models/predictors=rvc/models/predictors \
    $MODEL_ARGS \
    --noinclude-data-files="bundled/logs/*/*.npy" \
    --assume-yes-for-downloads \
    --report=build/nuitka/compilation-report.xml \
    "$@"
