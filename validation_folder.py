import split_folders


src_dir = 'data/input'
dst_dir = 'data/output'

split_folders.ratio(src_dir, dst_dir, seed=1337, ratio=(.8, .1, .1))
