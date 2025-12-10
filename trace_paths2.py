import sys
import re
import networkx as nx

def parse_dot_final(dot_path):
    G = nx.DiGraph()
    node_metadata = {}
    node_re = re.compile(r'^\s*(?P<id>[^\s\[]+)\s+\[(?P<attrs>.*)\]\s*;')
    attr_re = re.compile(r'(?P<key>\w+)="(?P<val>[^"]+)"')
    edge_re = re.compile(r'^\s*("?(?P<src>[^"\s\->]+)"?)\s*->\s*("?(?P<dst>[^"\s\[;]+)"?)')

    with open(dot_path, 'r') as f:
        for line in f:
            line = line.strip()
            n_match = node_re.search(line)
            if n_match:
                node_id = n_match.group('id').strip('"')
                attrs = dict(attr_re.findall(n_match.group('attrs')))
                G.add_node(node_id)
                node_metadata[node_id] = attrs
                continue
            e_match = edge_re.search(line)
            if e_match:
                src, dst = e_match.group('src').strip('"'), e_match.group('dst').strip('"')
                G.add_edge(src, dst)
    return G, node_metadata

def get_name(metadata, nid):
    return metadata.get(nid, {}).get('name', metadata.get(nid, {}).get('label', nid))

def find_relations(dot_path, start_query, end_query):
    G, metadata = parse_dot_final(dot_path)
    
    def matches(query, node_id):
        m = metadata.get(node_id, {})
        return (query in m.get('name', '') or query in m.get('label', ''))

    starts = [nid for nid in G.nodes if matches(start_query, nid)]
    ends = [nid for nid in G.nodes if matches(end_query, nid)]

    if not starts or not ends:
        print("Nodes not found.")
        return

    for s in starts:
        for e in ends:
            print(f"--- Analyzing relation: {get_name(metadata, s)} <-> {get_name(metadata, e)} ---")
            
            # 1. Check Directed Path (Lineage)
            try:
                paths = list(nx.all_simple_paths(G, source=s, target=e))
                for p in paths:
                    print(f"[Lineage] {' -> '.join([get_name(metadata, n) for n in p])}")
            except nx.NetworkXNoPath:
                pass

            # 2. Common Descendants (Common Child)
            s_desc = set(nx.descendants(G, s))
            e_desc = set(nx.descendants(G, e))
            common_children = s_desc.intersection(e_desc)
            for child in common_children:
                # Find shortest path to that common child from both
                p1 = nx.shortest_path(G, s, child)
                p2 = nx.shortest_path(G, e, child)
                print(f"[Common Child: {get_name(metadata, child)}]")
                print(f"  Path A: {' -> '.join([get_name(metadata, n) for n in p1])}")
                print(f"  Path B: {' -> '.join([get_name(metadata, n) for n in p2])}")

            # 3. Common Ancestors (Common Parent)
            # Use G.reverse() to find ancestors via descendants logic
            s_anc = set(nx.ancestors(G, s))
            e_anc = set(nx.ancestors(G, e))
            common_parents = s_anc.intersection(e_anc)
            for parent in common_parents:
                p1 = nx.shortest_path(G, parent, s)
                p2 = nx.shortest_path(G, parent, e)
                print(f"[Common Parent: {get_name(metadata, parent)}]")
                print(f"  Path A: {' -> '.join([get_name(metadata, n) for n in p1])}")
                print(f"  Path B: {' -> '.join([get_name(metadata, n) for n in p2])}")

if __name__ == "__main__":
    find_relations(sys.argv[1], sys.argv[2], sys.argv[3])
