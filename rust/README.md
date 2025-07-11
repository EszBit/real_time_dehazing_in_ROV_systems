## Profiling
The following command runs the profiling, saves it in the profile directory, and opens the Firefox profiler for visualization
```
cargo build --release && samply record -o profiles/profile.json.gz ./target/release/rust
```