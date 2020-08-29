#!/bin/bash

# cross-compile to arm
# requires nightly; arm intrinsics not stablised yet
cargo +nightly build --release --target=aarch64-unknown-linux-gnu --all-targets
