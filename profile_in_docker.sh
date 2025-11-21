#!/bin/bash
# Script to run inside the Docker container for profiling
# All outputs go to /output which is mounted to D:\temp\neural_net_profiling

echo "=== Rust Profiling with Perf + Inferno ==="
echo ""
echo "Output directory: /output (mounted to D:\temp\neural_net_profiling)"
echo ""

# Ensure output directory exists
mkdir -p /output

# Fix perf_event_paranoid if needed
if [ -f /proc/sys/kernel/perf_event_paranoid ]; then
    CURRENT=$(cat /proc/sys/kernel/perf_event_paranoid)
    if [ "$CURRENT" -gt -1 ]; then
        echo "Setting perf_event_paranoid to -1 for full profiling..."
        echo '-1' | tee /proc/sys/kernel/perf_event_paranoid > /dev/null
    fi
fi

# Build with debug symbols and frame pointers for better stack traces
echo "Building release binary with debug symbols and frame pointers..."
RUSTFLAGS="-C force-frame-pointers=yes -C debuginfo=2" \
    cargo build --release --example profile_genetic_instrumented

echo ""
echo "Running perf profiler..."
echo "This will collect call stack data with symbols"
echo ""

# Run perf record with call-graph support
perf record -F 997 --call-graph dwarf -o /output/perf.data \
    ./target/release/examples/profile_genetic_instrumented

echo ""
echo "Generating Speedscope profile (interactive viewer)..."
# Convert perf.data to speedscope format
perf script -i /output/perf.data -F +pid > /output/perf_script.txt
inferno-collapse-perf /output/perf_script.txt > /output/stacks_collapsed.txt
# Create a simple speedscope JSON format
python3 -c "
import json
import sys

# Read collapsed stacks
stacks = {}
total_samples = 0
with open('/output/stacks_collapsed.txt', 'r') as f:
    for line in f:
        parts = line.strip().rsplit(' ', 1)
        if len(parts) == 2:
            stack, count = parts
            count = int(count)
            stacks[stack] = count
            total_samples += count

# Build speedscope format
frames = []
frame_map = {}

def get_frame_id(name):
    if name not in frame_map:
        frame_map[name] = len(frames)
        frames.append({'name': name})
    return frame_map[name]

samples = []
weights = []

for stack_str, count in stacks.items():
    stack_frames = [get_frame_id(f) for f in stack_str.split(';')]
    samples.append(stack_frames)
    weights.append(count)

profile = {
    '\$schema': 'https://www.speedscope.app/file-format-schema.json',
    'shared': {'frames': frames},
    'profiles': [{
        'type': 'sampled',
        'name': 'perf profile',
        'unit': 'samples',
        'startValue': 0,
        'endValue': total_samples,
        'samples': samples,
        'weights': weights
    }]
}

with open('/output/profile.speedscope.json', 'w') as f:
    json.dump(profile, f, indent=2)

print(f'Generated Speedscope profile with {len(frames)} frames and {len(samples)} unique stacks')
" 2>/dev/null || echo "Speedscope generation skipped (requires Python 3)"



echo ""
echo "=== Profiling Complete ==="
echo ""
echo "Profile saved to: D:\temp\neural_net_profiling\profile.speedscope.json"
echo ""
echo "To view:"
echo "  1. Open https://www.speedscope.app in Firefox (recommended) or Chrome"
echo "  2. Drag and drop profile.speedscope.json"
echo "  3. Use search (Ctrl+F) to find functions like 'neural_network_scratch' or 'hill_descent'"
echo ""
echo "NOTE: Speedscope was developed for Chrome/Firefox."
echo "      Firefox tends to have better rendering. Try Firefox if Chrome has formatting issues."
