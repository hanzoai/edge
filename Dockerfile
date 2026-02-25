FROM rust:1.82-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p edge-cli

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/hanzo-edge /usr/bin/hanzo-edge
ENTRYPOINT ["/usr/bin/hanzo-edge"]
