# syntax=docker/dockerfile:1
# AllRunner image for the Intel Arc Pro B70 engine (launchers/arc-b70.sh).
#
# Built from a clean checkout on 2026-11-11 (about 5 min; image 32.3 GB): the GPU runtime packages and the three kernel
# libraries (_C.so, gdnchunk.so, ng.so) are byte-identical to the production image's. Follows echowarp/exl3xpu's
# docker/Dockerfile.arc and adds the vLLM XPU branch and llmt_kv.
#
#   docker build -f docker/Dockerfile.arc -t allrunner-arc .                (from the AllRunner checkout)
#
# Base: 0xSero's exl3xpu image, built on Intel's llm-scaler vLLM 0.26.0 XPU image (intel/llm-scaler-vllm:0.26.0-b2:
# oneAPI DPC++ 2025.3.3, oneDNN 3.9.1, PyTorch 2.12.0+xpu, Intel's vLLM patch set). Build needs network, no GPU.
#
# Every source repo is pinned to a commit (*_COMMIT); *_REF is only the branch it is fetched from. Docker caches a
# RUN step by its text, so an unpinned branch clone silently reuses an old checkout. Override both to build
# something else, e.g. --build-arg EXLLAMAV3_COMMIT=$(git -C ../exllamav3 rev-parse HEAD).

ARG EXL3XPU_BASE=ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8

# ---- llmt_kv (AllBroker's Python client: Rust/PyO3, abi3 wheel); same stage as in Dockerfile.rtx5090 -----------------
# TODO(release): pin the manylinux image by digest.
FROM quay.io/pypa/manylinux_2_28_x86_64 AS llmt-kv
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG ALLBROKER_REPO=https://github.com/echowarp/AllBroker.git
ARG ALLBROKER_REF=main
ARG ALLBROKER_COMMIT=e005fdd60a02f0f7a5951f508147dcfb1ada1c80
ARG RUST_VERSION=1.93.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
ENV PATH=/root/.cargo/bin:/opt/python/cp312-cp312/bin:$PATH
RUN pip install --no-cache-dir "maturin>=1.8,<2"
RUN git clone --filter=blob:none --single-branch --branch "${ALLBROKER_REF}" "${ALLBROKER_REPO}" /src/allbroker \
 && git -C /src/allbroker checkout -q --detach "${ALLBROKER_COMMIT}"
WORKDIR /src/allbroker/crates/py
RUN maturin build --release --compatibility manylinux_2_28 --out /wheels && ls -l /wheels

# ---- engine image ----------------------------------------------------------------------------------------------------
FROM ${EXL3XPU_BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG EXL3XPU_REPO=https://github.com/echowarp/exl3xpu.git
ARG EXL3XPU_REF=main
ARG EXL3XPU_COMMIT=eb06f13d2571f3363a2a2224be061e0f7ed4928a
ARG VLLM_REPO=https://github.com/echowarp/vllm.git
ARG VLLM_REF=xpu
ARG VLLM_COMMIT=f67d984af72673f4cc0c6044d938d1725745c9df
# The XPU branch starts with one commit that imports Intel's llm-scaler patch set (= the vLLM installed in the base
# image); our commits sit on top of it. Empty = find that commit by its subject.
ARG VLLM_BASE=

# GPU runtime the kernels were tuned on and production runs: compute-runtime (NEO) 26.31.39395.14 / IGC 2.40.13 /
# Level Zero 1.32 from the kobuk-team/intel-graphics PPA. The PPA has moved on (it now serves 26.40), so the exact
# packages are fetched from Launchpad's archive of superseded builds and checked by sha256. Launchpad often answers 503;
# the download retries. These are the files installed in the production image (dpkg md5sums identical).
ARG NEO_PPA=https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics/+files
# hadolint ignore=DL3008
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends git curl ca-certificates; \
    mkdir /tmp/neo; cd /tmp/neo; \
    printf '%s  %s\n' \
      6e79028431b03f98a77d0f0ba56750fcef48cdabbe2b0cd8361d9f03cafb9433 intel-opencl-icd_26.31.39395.14-1~24.04~ppa1_amd64.deb \
      c57e4ca8c780dc85a309e35b4d3249fc866251fea7bf3ebe68a5ab5ef5f71362 libigc2_2.40.13+ds1-1~24.04_amd64.deb \
      e71f41b4dd54c9cda7024680528034204cd911a1c2f644ec2bc3c06e968f5afe libigdfcl2_2.40.13+ds1-1~24.04_amd64.deb \
      2c68fcb150cba1b679063266cdf0054f26f18af90de71d6d3b614eaf95c98f80 libigdgmm12_22.10.1-1~24.04~ppa1_amd64.deb \
      d4001dc474d3cb74ded50a449b9f646cfdfbb31e69e70b8af38f2d18273dfce1 libze1_1.32.0-1~24.04~ppa1_amd64.deb \
      ff2de371d8f1489adf6b71980c929cbbc5ebf08f5a4e64a7572d12ddf264b3d9 libze-dev_1.32.0-1~24.04~ppa1_amd64.deb \
      39f10740e40cf0de1015997d0de094a3c3dc6371c3b9ea1dcd15771d64559c14 libze-intel-gpu1_26.31.39395.14-1~24.04~ppa1_amd64.deb \
      4808c4d96b3938b268371c4fc1336dcad7e3c56b71bd82913f98430067b1e346 libze-intel-gpu-dev_26.31.39395.14-1~24.04~ppa1_amd64.deb \
      > SHA256SUMS; \
    while read -r _ f; do \
      for i in 1 2 3 4 5 6 7 8; do \
        curl -fsSL --max-time 300 -o "$f" "${NEO_PPA}/$(echo "$f" | sed 's/+/%2B/g')" && break; \
        rm -f "$f"; sleep $((i * 10)); \
      done; \
    done < SHA256SUMS; \
    sha256sum -c SHA256SUMS; \
    apt-get install -y --no-install-recommends --allow-downgrades ./*.deb; \
    dpkg -l | grep -E "libze|intel-opencl|libigc"; \
    dpkg-query -W -f '${Version}' libze-intel-gpu1 | grep -qx '26.31.39395.14-1~24.04~ppa1'; \
    cd /; rm -rf /tmp/neo /var/lib/apt/lists/*

# 1) vLLM: copy our commits' changed Python files over Intel's installed vLLM (docker/overlay_vllm.sh: every replaced
#    file must equal the branch's import commit, i.e. the Intel files the branch was made from).
COPY docker/overlay_vllm.sh /usr/local/bin/overlay_vllm.sh
RUN set -eux; \
    git clone --filter=blob:none --single-branch --branch "${VLLM_REF}" "${VLLM_REPO}" /tmp/vllm; \
    git -C /tmp/vllm checkout -q --detach "${VLLM_COMMIT}"; \
    base="${VLLM_BASE}"; \
    if [ -z "$base" ]; then \
      base=$(git -C /tmp/vllm log --format=%H -1 --grep='^Import Intel llm-scaler XPU patch set'); \
    fi; \
    test -n "$base"; \
    bash /usr/local/bin/overlay_vllm.sh /tmp/vllm "$base"; \
    rm -rf /tmp/vllm

# 2) exl3xpu fork: replaces the base image's upstream exl3xpu at the same path (/opt/exl3xpu, editable install), builds
#    _C.so, gdnchunk.so and ng.so with the production options (scripts/build_arc.sh) and installs the OOR guard
#    (tools/oor/llmt_memprobe.py + .pth, read TORCH_CAP_MIB / MEMPROBE_S at run time) into site-packages.
RUN set -eux; \
    cd /; rm -rf /opt/exl3xpu; \
    git clone --filter=blob:none --single-branch --branch "${EXL3XPU_REF}" "${EXL3XPU_REPO}" /opt/exl3xpu; \
    git -C /opt/exl3xpu checkout -q --detach "${EXL3XPU_COMMIT}"; \
    cd /opt/exl3xpu; \
    bash scripts/build_arc.sh all; \
    pip install --no-cache-dir --no-deps --no-build-isolation -e .; \
    sp=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    cp tools/oor/llmt_memprobe.py tools/oor/llmt_memprobe.pth "$sp"/; \
    test -f exl3xpu/_C.so && test -f exl3xpu/gdnchunk.so && test -f exl3xpu/ng.so

# 3) llmt_kv: the RAM-tier connector (llmt_kv.xpu_tier) and the mmap-free weight loader
COPY --from=llmt-kv /wheels /tmp/wheels
RUN pip install --no-cache-dir --no-deps /tmp/wheels/llmt_kv-*.whl && rm -rf /tmp/wheels \
 && python3 -c "import importlib.util as u; assert u.find_spec('llmt_kv') and u.find_spec('llmt_kv.xpu_tier')"

# 4) whx.so: Level Zero dma-buf export / import of the weight keeper's XPU backend (llmt_kv.whx_loader, EXL3_WHX_LIB;
#    without it the keeper export fails and the engine runs without a keeper). Built from AllBroker's weights/xpu/whx_ops.sycl.
COPY --from=llmt-kv /src/allbroker/weights/xpu/whx_ops.sycl /tmp/whx_ops.sycl
RUN set -eux; \
    command -v icpx >/dev/null 2>&1 || { set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u; }; \
    T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))"); \
    icpx -fsycl -fsycl-targets=spir64 -O3 -fPIC -std=c++17 -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
      -I"$T/include" -I"$T/include/torch/csrc/api/include" -x c++ /tmp/whx_ops.sycl -o /opt/exl3xpu/whx.so \
      -L"$T/lib" -Wl,-rpath,"$T/lib" -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu -lze_loader -Wno-deprecated-declarations; \
    rm /tmp/whx_ops.sycl; test -f /opt/exl3xpu/whx.so

WORKDIR /opt/exl3xpu
ENV VLLM_XPU_USE_SAMPLER_KERNEL=0
ENV MODELS_DIR=/models
LABEL org.opencontainers.image.title="allrunner-arc" \
      org.opencontainers.image.description="vLLM 0.26 XPU (Intel llm-scaler) + exl3xpu (EXL3) + llmt_kv for Qwen3.8-27B on an Arc Pro B70" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"
ENTRYPOINT ["python3", "scripts/serve.py"]
CMD ["models/qwen3.8-27b-exl3-4.00bpw", "--gpu", "0"]
