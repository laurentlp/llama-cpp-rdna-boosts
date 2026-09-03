# Patched llama-swap:rocm image with the rdna-boosts set (blocks 01-13).
#
# Stage 1 (builder): clean llama.cpp at the recorded fork point + `git am`
# the patch set, HIP build for gfx1201 (R9700) only.
# Stage 2 (runtime): stock llama-swap:rocm with the patched llama-server
# and ggml/llama shared libs overlaid on /app (llama-swap proxy untouched).
#
# Build args:
#   LLAMA_FORK_POINT  llama.cpp SHA the patches apply to (see patches/README.md)
#   GPU_TARGETS       HIP arch list (single-arch = much faster compile)
ARG LLAMA_FORK_POINT=9cffdcc80
ARG GPU_TARGETS=gfx1201

FROM rocm/dev-ubuntu-24.04:7.2 AS builder
ARG LLAMA_FORK_POINT
ARG GPU_TARGETS
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build ccache curl libcurl4-openssl-dev \
      rocblas-dev hipblas-dev hipsparse-dev hipsolver-dev \
    && rm -rf /var/lib/apt/lists/*

# Patches + apply script from this repo (own layer: reruns only when they change).
COPY patches/ /rdna/patches/
COPY scripts/apply-all.sh /rdna/scripts/apply-all.sh

# Partial clone (blobs on demand) then pin to the fork point. Cached unless
# LLAMA_FORK_POINT changes.
WORKDIR /src
RUN git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp llama.cpp \
 && cd llama.cpp \
 && git checkout "${LLAMA_FORK_POINT}" \
 && git config user.email "ci@localhost" && git config user.name "ci" \
 && git config --global --add safe.directory /src/llama.cpp

RUN cd /src/llama.cpp && bash /rdna/scripts/apply-all.sh /src/llama.cpp /rdna

# HIP build (GGML_HIP_GRAPHS + NATIVE kept for parity with the tuned ref build).
# NOTE: no trailing `|| true` — a failed configure/build must fail the job.
RUN --mount=type=cache,target=/root/.ccache \
    cd /src/llama.cpp \
 && cmake -B build -G Ninja \
      -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=1 \
      -DGPU_TARGETS="${GPU_TARGETS}" -DGGML_CCACHE=ON \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)"

# Collect every built .so (preserving versioned symlinks) + server/cli tools.
# Binary dir is located, not assumed: must exist or the job fails here loudly.
RUN mkdir -p /stage \
 && cd /src/llama.cpp/build \
 && find . -name 'lib*.so*' -exec cp -P {} /stage/ \; \
 && SRV="$(find . -name llama-server -type f | head -1)" \
 && CLI="$(find . -name llama-cli -type f | head -1)" \
 && test -n "$SRV" -a -n "$CLI" \
 && cp "$SRV" "$CLI" /stage/ \
 && ls /stage/ | head -n 30

# Smoke test: no missing shared-lib deps (runs without a GPU).
RUN ldd /stage/llama-server | grep -i "not found" && exit 1 || echo "ldd clean" \
 && /stage/llama-server --version

FROM ghcr.io/mostlygeek/llama-swap:rocm

# Drop the stock backend wholesale (keep llama-swap proxy, config, sd-server).
RUN rm -f /app/lib*.so* /app/llama-server /app/llama-cli
COPY --from=builder /stage/ /app/
