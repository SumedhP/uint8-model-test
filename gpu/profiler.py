import cProfile
import pstats
import label_images

# cProfile.run('label_images.main()', 'profiler_output')

p = pstats.Stats('profiler_output')
# Sort by cum time
p.strip_dirs().sort_stats('cumulative').print_stats()
