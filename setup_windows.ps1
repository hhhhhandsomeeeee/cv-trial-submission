$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

$ProjectPython = ".venv\Scripts\python.exe"

# tensorflow-hub 0.16 depends on tf-keras, whose package metadata asks pip for
# the full `tensorflow` distribution even when tensorflow-cpu already provides
# the importable runtime. Install the compatible wrappers without dependencies
# to avoid downloading a second ~370 MB TensorFlow wheel on Windows.
& $ProjectPython -m pip install `
    tensorflow-cpu==2.18.0 `
    opencv-python==4.10.0.84 `
    numpy==1.26.4 `
    matplotlib==3.9.2 `
    pytest==8.3.5 `
    setuptools==75.8.0

& $ProjectPython -m pip install `
    tensorflow-hub==0.16.1 `
    tf-keras==2.18.0 `
    --no-deps

& $ProjectPython -c "import tensorflow as tf, tensorflow_hub as hub, tf_keras; print('TensorFlow', tf.__version__, 'TF-Hub', hub.__version__, 'tf-keras', tf_keras.__version__)"
