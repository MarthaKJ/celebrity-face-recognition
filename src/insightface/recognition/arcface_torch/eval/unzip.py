import pickle

with open('/workspace/datasets/manually-annotated/lfw.bin', 'rb') as f:
    bins, issame_list = pickle.load(f)

print(f"# of binary images: {len(bins)}")
print(f"# of issame pairs: {len(issame_list)}")

# Check the first bin
sample = bins[0]
print(f"First image bin size: {len(sample)} bytes")
