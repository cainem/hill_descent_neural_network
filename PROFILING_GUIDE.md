# Profiling with Docker and Flamegraph

This guide shows how to generate flamegraph profiles on Linux using Docker.

## Important: Docker Storage Configuration

All Docker data and output files will be stored on D: drive to save C: drive space.

### One-Time Setup: Move Docker Data Root (Optional but Recommended)

To move Docker's data root from C: to D: permanently:

1. Stop Docker Desktop
2. Create `D:\docker` directory
3. Edit Docker Desktop settings:
   - Open Docker Desktop
   - Settings → Docker Engine
   - Add to the JSON config:
     ```json
     {
       "data-root": "D:\\docker"
     }
     ```
4. Restart Docker Desktop

## Quick Start (Easiest Method)

Just run the PowerShell script:

```powershell
.\run_profiling.ps1
```

This will:
- Create `D:\temp\neural_net_profiling` directory
- Build the Docker image
- Start the container with correct volume mounts
- When you're done, automatically check for output files
- Offer to open the flamegraph

Inside the container, run:
```bash
./profile_in_docker.sh
```

## Manual Method

If you prefer to run commands manually:

1. **Create output directory:**
   ```powershell
   New-Item -ItemType Directory -Force -Path D:\temp\neural_net_profiling
   ```

2. **Build the Docker image:**
   ```powershell
   docker build -f Dockerfile.profiling -t neural-net-profiler .
   ```

3. **Run the container (outputs to D:\temp):**
   ```powershell
   docker run --privileged `
     -v ${PWD}:/workspace `
     -v D:\temp\neural_net_profiling:/output `
     -it neural-net-profiler
   ```

4. **Inside the container, run profiling:**
   ```bash
   ./profile_in_docker.sh
   ```

5. **View the flamegraph:**
   - File location: `D:\temp\neural_net_profiling\flamegraph.svg`
   - Open in any browser (Chrome, Firefox, Edge)
   - Interactive: click to zoom, search for function names
   - All perf data and logs saved to `D:\temp\neural_net_profiling\`

## Manual Profiling Commands

If you want more control, run these commands inside the container:

### Basic flamegraph:
```bash
cargo flamegraph --example profile_genetic_instrumented --release
```

### With custom duration (for longer runs):
```bash
cargo flamegraph --example profile_genetic_instrumented --release -- --duration 120
```

### Generate collapsed stacks (text format):
```bash
cargo flamegraph --example profile_genetic_instrumented --release --output flamegraph.svg
```

## Filtering the Output

To see only specific crates in the flamegraph:

### Option 1: Post-process perf data
```bash
# Generate perf data
perf record -F 99 -g target/release/examples/profile_genetic_instrumented

# Collapse stacks and filter
perf script | stackcollapse-perf.pl | grep 'neural_network\|hill_descent' | flamegraph.pl > filtered.svg
```

### Option 2: Use regex filtering in flamegraph
The flamegraph SVG is interactive - use the search box to highlight specific modules.

## Understanding the Output

- **Width** = % of time spent in that function
- **Height** = call stack depth
- **Color** = different modules (automatic)
- **Click** = zoom into that section
- **Search** = highlight matching functions

## What to Look For

1. **Wide bars at the top** = hot spots (most time spent)
2. **Compare widths** between `neural_network_scratch::` and `hill_descent_lib::`
3. **Deep stacks** = potential optimization targets

## Troubleshooting

If you get "perf not found" errors:
- Make sure you ran with `--privileged` flag
- The container needs kernel access for perf profiling

If cargo flamegraph fails:
- Try: `apt-get update && apt-get install linux-tools-generic`
