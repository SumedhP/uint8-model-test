from line_profiler import load_stats, show_text


# Load the .lprof file
stats = load_stats("bench.py.lprof")

show_text(stats.timings, unit=1e-6, rich=True, output_unit=1e-5)
