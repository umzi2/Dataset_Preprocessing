from src.enum import ProcessType
from src.scripts.utils.best_tile import BestTile

input_folder = ""
output_folder = ""
tile_size = 256
process_type = ProcessType.THREAD
scale = 4
dynamic_number_tile = True
median_blur = 5
laplacian_thread = 0.01
image_gray = False

best_tile = BestTile(
    input_folder,
    output_folder,
    tile_size,
    process_type,
    scale,
    dynamic_number_tile,
    median_blur,
    laplacian_thread,
    image_gray,
)
best_tile.run()

tile_thread = True
#
# if tile_thread:
#     from src.torch_enum import ThreadAlg
#
#     thread = 0.5
#     median_thread = 0
#     batch_size = 25
#     move_folder = None
#     thread_in_folder = output_folder
#     thread_type = ThreadAlg.HIPERIQA
#
#     alg = thread_type.value(output_folder, batch_size, thread, median_thread, move_folder)
#     alg()
