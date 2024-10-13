from best_tile import BestTile
from fast_hyper import HyperThread

input_folder = ""
output_folder = ""
hyper_thread = 0.5
hyper_batch_size = 16
hyper_move_folder = None # ""
bt = BestTile(input_folder, output_folder, scale=1)
bt.run()

hyper = HyperThread(output_folder,hyper_batch_size,hyper_thread,hyper_move_folder)
hyper.run()
