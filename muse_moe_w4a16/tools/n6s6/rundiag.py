import sys, importlib.util, time
sys.path.insert(0, '.')
import reference
sp = importlib.util.spec_from_file_location("s", sys.argv[1]); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
t = time.time(); st = m.prepare(reference.generate_weights(reference.WEIGHTS_SEED)); print("prepare", time.time() - t)
