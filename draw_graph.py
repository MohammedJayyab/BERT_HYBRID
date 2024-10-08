import pandas as pd
import networkx as nx
from pyvis.network import Network

def draw_graph(products_file, transactions_file, output_file='graph.html'):
    # Load the data
    products_df = pd.read_csv(products_file)
    transactions_df = pd.read_csv(transactions_file)

    # Initialize the graph
    G = nx.Graph()

    # Add product nodes to the graph
    for _, row in products_df.iterrows():
        product_node = f"product_{row['product_id']}"
        G.add_node(product_node, label=row['product_name'], title=f"Category: {row['category_name']}", color='skyblue', shape='dot')

    # Add user-product interactions to the graph
    for _, row in transactions_df.iterrows():
        user_node = f"user_{row['customer_id']}"
        product_node = f"product_{row['product_id']}"
        interaction_type = row['interaction_type']
        weight = {'clicked': 1, 'added': 2, 'purchased': 3}[interaction_type]

        # Determine edge color based on interaction type
        edge_color = 'black' if interaction_type == 'clicked' else 'gray' if interaction_type == 'added' else 'green'
        edge_width = 1 if interaction_type == 'clicked' else 2 if interaction_type == 'added' else 3

        if user_node not in G:
            G.add_node(user_node, label=f"User {row['customer_id']}", title="User Node", color='lightgreen', shape='dot')

        # Add an edge with the interaction type, color, and weight
        G.add_edge(user_node, product_node, color=edge_color, width=edge_width, title=f"{interaction_type} (weight: {weight})")

    # Create a pyvis network
    net = Network(height='1000px', width='100%', bgcolor='#222222', font_color='white')
    net.from_nx(G)

    # Adjust node sizes based on the number of edges (degree)
    for node in net.nodes:
        degree = G.degree(node['id'])
        node['size'] = 15 + (degree * 3)
        node['borderWidth'] = 2 + (degree * 0.5)

    # Customize the network options
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidthSelected": 4,
        "shadow": true,
        "shape": "dot"
      },
      "edges": {
        "color": {
          "inherit": false
        },
        "smooth": {
          "type": "dynamic"
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": true,
        "dragView": true,
        "zoomView": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75,
        "solver": "barnesHut"
      }
    }
    """)

    # Save the network graph to an HTML file
    net.write_html(output_file)
    print(f"Graph saved to {output_file}")

# Example usage
if __name__ == "__main__":
    draw_graph('data/products.csv', 'data/transactions.csv')
