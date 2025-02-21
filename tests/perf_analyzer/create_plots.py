import matplotlib.pyplot as plt

# --------------------------
# 1) Concurrency vs. Throughput & Latency
# --------------------------

concurrency = [1, 2, 3, 4]
throughput = [0.0333326, 0.0333327, 0.0333326, 0.0333325]  # infer/sec
avg_latency_usec = [31904479, 63590510, 95933221, 126937324]

# Convert latencies to seconds
avg_latency_s = [val / 1e6 for val in avg_latency_usec]

plt.figure(figsize=(10, 4))

# --- Subplot 1: Throughput vs Concurrency ---
plt.subplot(1, 2, 1)
plt.plot(concurrency, throughput, marker="o")
plt.title("Throughput vs Concurrency")
plt.xlabel("Concurrency")
plt.ylabel("Throughput (infer/sec)")

# --- Subplot 2: Latency vs Concurrency ---
plt.subplot(1, 2, 2)
plt.plot(concurrency, avg_latency_s, marker="o", color="orange")
plt.title("Latency vs Concurrency")
plt.xlabel("Concurrency")
plt.ylabel("Avg Latency (s)")

plt.tight_layout()
plt.savefig("latency_vs_concurrency.png")


# --------------------------
# 2) Request Size Variation
# --------------------------

tokens = [50, 300, 600]
throughput2 = [1.22218, 0.205551, 0.099998]
avg_latency_usec2 = [813593, 4760297, 9511368]
avg_latency_s2 = [val / 1e6 for val in avg_latency_usec2]

plt.figure(figsize=(10, 4))

# --- Subplot 1: Throughput vs. max_tokens ---
plt.subplot(1, 2, 1)
plt.plot(tokens, throughput2, marker="o")
plt.title("Throughput vs. max_tokens")
plt.xlabel("max_tokens")
plt.ylabel("Throughput (infer/sec)")

# --- Subplot 2: Latency vs. max_tokens ---
plt.subplot(1, 2, 2)
plt.plot(tokens, avg_latency_s2, marker="o", color="orange")
plt.title("Latency vs. max_tokens")
plt.xlabel("max_tokens")
plt.ylabel("Avg Latency (s)")

plt.tight_layout()
plt.savefig("latency_vs_max_tokens.png")
