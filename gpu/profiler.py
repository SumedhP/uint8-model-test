import cProfile
import pstats
import label_images

cProfile.run('label_images.main()', 'profiler_output')

def f8_alt(x):
    return "%1.9f" % x
  
pstats.f8 = f8_alt

p = pstats.Stats('profiler_output')
# Sort by cum time
p.strip_dirs().sort_stats('cumulative').print_stats(10,10)
