import csv

input_file = "large_twitch_edges.csv"
output_file = "large_twitch_edges.txt"

edges = []
nodes = set()

# Read CSV
with open(input_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header

    for row in reader:
        u = int(row[0])
        v = int(row[1])
        edges.append((u, v))
        nodes.add(u)
        nodes.add(v)

# Map old IDs to new 0-based IDs
node_list = sorted(nodes)
id_map = {node: idx for idx, node in enumerate(node_list)}

num_nodes = len(node_list)
num_edges = len(edges)

# Write output
with open(output_file, "w") as f:
    # Header
    f.write(f"{num_nodes} {num_nodes} {num_edges}\n")

    # Edges
    for u, v in edges:
        f.write(f"{id_map[u]}\t{id_map[v]}\n")

print("Conversion complete.")