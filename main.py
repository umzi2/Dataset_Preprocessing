from src.scripts.best_tile import BestTile

input_folder = "/run/media/umzi/H/mahoudb/"
output_folder = "/run/media/umzi/H/mahoudb33/"
tile_size = 512
process_type = "thread"
scale = 1
tile_thread = False

bt = BestTile(input_folder, output_folder, tile_size, process_type, scale)
bt.run()

if tile_thread:
    from src.enum import ThreadAlg
    thread = 0.5
    batch_size = 16
    move_folder = None # ""
    thread_in_folder = output_folder
    thread_type = ThreadAlg.HIPERIQA

    alg = thread_type.value(output_folder, batch_size, thread, move_folder)
    alg.run()
