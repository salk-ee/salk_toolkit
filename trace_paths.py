import sys
import re
import networkx as nx

def parse_dot_final(dot_path):
    G = nx.DiGraph()
    node_metadata = {}
    
    # Matches node_id [attrs] ;
    node_re = re.compile(r'^\s*(?P<id>[^\s\[]+)\s+\[(?P<attrs>.*)\]\s*;')
    attr_re = re.compile(r'(?P<key>\w+)="(?P<val>[^"]+)"')
    # Matches "src" -> "dst" [attrs] ; (handles optional quotes)
    edge_re = re.compile(r'^\s*("?(?P<src>[^"\s\->]+)"?)\s*->\s*("?(?P<dst>[^"\s\[;]+)"?)')

    with open(dot_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 1. Parse Nodes
            n_match = node_re.search(line)
            if n_match:
                # Strip quotes if they exist in the ID
                node_id = n_match.group('id').strip('"')
                attrs = dict(attr_re.findall(n_match.group('attrs')))
                G.add_node(node_id)
                node_metadata[node_id] = attrs
                continue # Move to next line
            
            # 2. Parse Edges
            e_match = edge_re.search(line)
            if e_match:
                src = e_match.group('src').strip('"')
                dst = e_match.group('dst').strip('"')
                G.add_edge(src, dst)
                
    return G, node_metadata

def trace_all_chains(dot_path, start_query, end_query):
    G, metadata = parse_dot_final(dot_path)
    
    # Internal matcher
    def matches(query, node_id):
        m = metadata.get(node_id, {})
        return (query in m.get('name', '') or query in m.get('label', ''))

    start_ids = [nid for nid in G.nodes if matches(start_query, nid)]
    end_ids = [nid for nid in G.nodes if matches(end_query, nid)]

    print(f"--- Debug ---")
    print(f"Nodes in Graph: {G.number_of_nodes()}, Edges in Graph: {G.number_of_edges()}")
    print(f"Matches for '{start_query}': {start_ids}")
    print(f"Matches for '{end_query}': {end_ids}")
    print(f"--------------\n")

    if not start_ids or not end_ids:
        return

    found = False
    for s in start_ids:
        for e in end_ids:
            try:
                # nx.all_simple_paths is a generator
                paths = list(nx.all_simple_paths(G, source=s, target=e))
                for i, path in enumerate(paths, 1):
                    found = True
                    readable = [metadata[n].get('name', metadata[n].get('label', n)) for n in path]
                    print(f"Path {i}: {' -> '.join(readable)}")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    
    if not found:
        print(f"No functional path exists from {start_query} to {end_query}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python trace_paths.py <file.dot> <start_func> <end_func>")
    else:
        trace_all_chains(sys.argv[1], sys.argv[2], sys.argv[3])
