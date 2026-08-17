# OpenVINO 2026.3 预编译包环境（等效 setupvars.ps1）
OVROOT="C:/Users/Mcsof/Application/openvino_2026.3.0"
export INTEL_OPENVINO_DIR="$OVROOT"
export OPENVINO_DIR="$OVROOT/runtime/cmake"
export OPENVINO_LIB_PATHS="$OVROOT/runtime/bin/intel64/Release;$OVROOT/runtime/bin/intel64/Debug"
for p in "$OVROOT/runtime/3rdparty/tbb/redist/intel64/vc14" "$OVROOT/runtime/3rdparty/tbb/bin/intel64/vc14" "$OVROOT/runtime/3rdparty/tbb/bin"; do
  [ -d "$p" ] && export OPENVINO_LIB_PATHS="$p;$OPENVINO_LIB_PATHS" && break
done
export PATH="$OPENVINO_LIB_PATHS:$PATH"
export PYTHONPATH="$OVROOT/python"
