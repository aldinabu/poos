import split_folders


src_dir = 'dataset/input'
dst_dir = 'dataset/output'

split_folders.ratio(src_dir, dst_dir, seed=1337, ratio=(.835, .165))
