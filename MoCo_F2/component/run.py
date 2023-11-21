import time
import traceback
import CaseGenerator as cC


def go(net_name):
    time.sleep(1.0)
    cC.concrete.new_experiment()
    start_time = time.time()
    try:
        cC.goFuzzing(net_name)
        cC.generateBoundary(cC.concrete.get_experiment_id())
        end_time = time.time()
    except Exception:
        print(traceback.format_exc())
        end_time = time.time()
    print("total time cost: " + str(end_time - start_time))
