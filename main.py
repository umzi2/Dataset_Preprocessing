from pepedp.scripts.utils.deduplicate import (
    create_embedd,
    filtered_pairs,
    move_duplicate_files,
)

embedded = create_embedd("/run/media/umzi/H/nahuy_pixiv/new/visualization/1/")
paired = filtered_pairs(embedded)
move_duplicate_files(
    paired,
    "/run/media/umzi/H/nahuy_pixiv/new/visualization/1/",
    "/run/media/umzi/H/nahuy_pixiv/new/visualization/2/",
)
