# Visualization Module

#!/usr/bin/env python3
"""
sep_visualization.py

Visualization module for the SEP Knowledge Graph
Generates interactive visualizations using multiple backends
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import altair as alt

# For 3D visualizations
try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class SEPVisualizer:
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualization module
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set color schemes
        self.color_schemes = {
            'camps': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#dfe6e9'],
            'philosophers': '#74b9ff',
            'concepts': '#a29bfe',
            'articles': '#fd79a8',
            'schools': '#e17055',
            'default': '#95a5a6'
        }
        
        # Layout algorithms
        self.layouts = {
            'force': self._force_directed_layout,
            'circular': self._circular_layout,
            'hierarchical': self._hierarchical_layout,
            'radial': self._radial_layout,
            'spring': self._spring_layout
        }
    
    def create_debate_visualization(self, debate_data: Dict, 
                                  output_format: str = 'interactive') -> str:
        """
        Create visualization for philosophical debates
        
        Args:
            debate_data: Debate structure with camps, arguments, etc.
            output_format: 'interactive', 'static', or 'both'
        
        Returns:
            Path to saved visualization
        """
        if output_format in ['interactive', 'both']:
            interactive_path = self._create_interactive_debate_viz(debate_data)
        
        if output_format in ['static', 'both']:
            static_path = self._create_static_debate_viz(debate_data)
        
        return interactive_path if output_format == 'interactive' else static_path
    
    def _create_interactive_debate_viz(self, debate_data: Dict) -> str:
        """Create interactive debate visualization using Pyvis"""
        
        net = Network(height="750px", width="100%", bgcolor="#222222", 
                     font_color="white", directed=True)
        
        # Configure physics
        net.barnes_hut(gravity=-80000, central_gravity=0.3, 
                       spring_length=250, spring_strength=0.001)
        
        # Add camp nodes (larger, central)
        camps = debate_data.get('camps', {})
        for i, (camp_name, members) in enumerate(camps.items()):
            net.add_node(
                camp_name,
                label=camp_name,
                color=self.color_schemes['camps'][i % len(self.color_schemes['camps'])],
                size=30 + len(members) * 5,
                shape='box',
                font={'size': 20}
            )
        
        # Add philosopher nodes
        philosopher_positions = {}
        for camp_name, members in camps.items():
            for member in members:
                if member not in philosopher_positions:
                    net.add_node(
                        member,
                        label=member,
                        color=self.color_schemes['philosophers'],
                        size=20,
                        title=f"Philosopher in {camp_name} camp"
                    )
                    philosopher_positions[member] = []
                
                philosopher_positions[member].append(camp_name)
                net.add_edge(member, camp_name, 
                           color='gray', width=2)
        
        # Add argument nodes
        arguments = debate_data.get('key_arguments', {})
        for camp_name, camp_arguments in arguments.items():
            for i, argument in enumerate(camp_arguments[:5]):  # Limit to 5 arguments
                arg_id = f"{camp_name}_arg_{i}"
                net.add_node(
                    arg_id,
                    label=argument[:50] + "..." if len(argument) > 50 else argument,
                    color='#ffeaa7',
                    size=15,
                    shape='ellipse',
                    title=argument  # Full argument on hover
                )
                net.add_edge(camp_name, arg_id, 
                           color='orange', width=1, 
                           label='argues')
        
        # Add relationships between camps
        for rel in debate_data.get('relationships', []):
            if rel['source'] in camps and rel['target'] in camps:
                net.add_edge(
                    rel['source'], 
                    rel['target'],
                    label=rel.get('type', 'relates to'),
                    color='red' if 'CRITIQUE' in rel.get('type', '') else 'green',
                    width=3
                )
        
        # Save visualization
        output_path = self.output_dir / f"debate_{debate_data.get('topic', 'unnamed').replace(' ', '_')}.html"
        net.save_graph(str(output_path))
        
        # Add custom CSS and JS
        self._enhance_interactive_viz(output_path)
        
        return str(output_path)
    
    def _create_static_debate_viz(self, debate_data: Dict) -> str:
        """Create static debate visualization using matplotlib"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Main network plot
        ax1 = plt.subplot(221)
        G = self._debate_to_networkx(debate_data)
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes by type
        camps = debate_data.get('camps', {})
        camp_nodes = list(camps.keys())
        philosopher_nodes = [n for n in G.nodes() if n not in camp_nodes and '_arg_' not in n]
        argument_nodes = [n for n in G.nodes() if '_arg_' in n]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=camp_nodes, 
                              node_color=self.color_schemes['camps'][:len(camp_nodes)],
                              node_size=3000, ax=ax1)
        nx.draw_networkx_nodes(G, pos, nodelist=philosopher_nodes,
                              node_color=self.color_schemes['philosophers'],
                              node_size=1000, ax=ax1)
        nx.draw_networkx_nodes(G, pos, nodelist=argument_nodes,
                              node_color='#ffeaa7',
                              node_size=500, ax=ax1)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.5)
        
        # Draw labels
        labels = {n: n[:15] + '...' if len(n) > 15 else n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax1)
        
        ax1.set_title(f"Debate Network: {debate_data.get('topic', 'Unknown')}", fontsize=16)
        ax1.axis('off')
        
        # Camp distribution pie chart
        ax2 = plt.subplot(222)
        camp_sizes = [len(members) for members in camps.values()]
        ax2.pie(camp_sizes, labels=list(camps.keys()), autopct='%1.1f%%',
                colors=self.color_schemes['camps'][:len(camps)])
        ax2.set_title("Distribution of Philosophers by Camp")
        
        # Timeline plot
        ax3 = plt.subplot(223)
        timeline = debate_data.get('historical_development', [])
        if timeline:
            periods = [item.get('period', 'Unknown') for item in timeline[:10]]
            y_pos = np.arange(len(periods))
            
            ax3.barh(y_pos, np.arange(len(periods)), 
                    color=plt.cm.viridis(np.linspace(0, 1, len(periods))))
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(periods)
            ax3.set_xlabel('Development Stages')
            ax3.set_title('Historical Development')
        
        # Key texts list
        ax4 = plt.subplot(224)
        ax4.axis('off')
        texts = debate_data.get('central_texts', [])[:10]
        text_str = "Key Texts:\n\n" + "\n".join([f"• {text}" for text in texts])
        ax4.text(0.1, 0.9, text_str, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"debate_{debate_data.get('topic', 'unnamed').replace(' ', '_')}_static.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_concept_network_viz(self, concept: str, nodes: List[Dict], 
                                 relationships: List[Dict], 
                                 layout: str = 'force',
                                 dimensions: str = '2d') -> str:
        """
        Create concept network visualization
        
        Args:
            concept: Central concept
            nodes: List of node dictionaries
            relationships: List of relationship dictionaries
            layout: Layout algorithm to use
            dimensions: '2d' or '3d'
        
        Returns:
            Path to saved visualization
        """
        if dimensions == '3d' and SKLEARN_AVAILABLE:
            return self._create_3d_concept_network(concept, nodes, relationships)
        else:
            return self._create_2d_concept_network(concept, nodes, relationships, layout)
    
    def _create_2d_concept_network(self, concept: str, nodes: List[Dict], 
                                  relationships: List[Dict], layout: str) -> str:
        """Create 2D concept network using Plotly"""
        
        # Build graph
        G = nx.Graph()
        
        node_types = {}
        node_distances = {}
        
        for node in nodes:
            node_id = node['id']
            G.add_node(node_id)
            node_types[node_id] = node.get('type', 'Unknown')
            node_distances[node_id] = node.get('distance', 1)
        
        for rel in relationships:
            G.add_edge(rel['source'], rel['target'])
        
        # Calculate layout
        if layout in self.layouts:
            pos = self.layouts[layout](G, center_node=concept)
        else:
            pos = nx.spring_layout(G)
        
        # Create Plotly figure
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none'
                )
            )
        
        # Node traces by type
        node_traces = {}
        for node_type in set(node_types.values()):
            node_traces[node_type] = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers+text',
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    size=[],
                    color=self.color_schemes.get(node_type.lower(), self.color_schemes['default']),
                    line=dict(width=2, color='white')
                ),
                name=node_type
            )
        
        # Add nodes to traces
        for node in G.nodes():
            x, y = pos[node]
            node_type = node_types[node]
            distance = node_distances.get(node, 1)
            
            node_traces[node_type]['x'] += tuple([x])
            node_traces[node_type]['y'] += tuple([y])
            node_traces[node_type]['text'] += tuple([node])
            
            # Size based on centrality and distance
            centrality = nx.degree_centrality(G)[node]
            size = 20 + (30 * centrality) - (5 * distance)
            node_traces[node_type]['marker']['size'] += tuple([size])
        
        # Create figure
        fig = go.Figure(
            data=edge_trace + list(node_traces.values()),
            layout=go.Layout(
                title=f"Concept Network: {concept}",
                titlefont_size=20,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0.05)'
            )
        )
        
        # Save
        output_path = self.output_dir / f"concept_network_{concept.replace(' ', '_')}.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def _create_3d_concept_network(self, concept: str, nodes: List[Dict], 
                                  relationships: List[Dict]) -> str:
        """Create 3D concept network visualization"""
        
        # Build graph and get embeddings
        G = nx.Graph()
        node_features = []
        node_labels = []
        
        for node in nodes:
            G.add_node(node['id'])
            node_labels.append(node['id'])
            
            # Create feature vector
            features = [
                node.get('distance', 0),
                len(node.get('id', '')),
                1 if node.get('type') == 'Concept' else 0,
                1 if node.get('central', False) else 0
            ]
            node_features.append(features)
        
        for rel in relationships:
            if rel['source'] in G and rel['target'] in G:
                G.add_edge(rel['source'], rel['target'])
        
        # Add graph-based features
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        for i, node in enumerate(node_labels):
            node_features[i].extend([
                centrality.get(node, 0),
                betweenness.get(node, 0)
            ])
        
        # Dimensionality reduction
        features_array = np.array(node_features)
        
        if len(features_array) > 3:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(nodes)-1))
            coords_3d = tsne.fit_transform(features_scaled)
        else:
            # If too few nodes, use simple layout
            coords_3d = np.random.rand(len(nodes), 3)
        
        # Create 3D plot
        fig = go.Figure()
        
        # Add edges
        edge_trace = []
        for edge in G.edges():
            i = node_labels.index(edge[0])
            j = node_labels.index(edge[1])
            
            edge_trace.append(
                go.Scatter3d(
                    x=[coords_3d[i, 0], coords_3d[j, 0], None],
                    y=[coords_3d[i, 1], coords_3d[j, 1], None],
                    z=[coords_3d[i, 2], coords_3d[j, 2], None],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='none'
                )
            )
        
        # Add nodes
        node_trace = go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=[10 + 20 * centrality.get(node, 0) for node in node_labels],
                color=[nodes[i].get('distance', 0) for i in range(len(nodes))],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Distance from center")
            ),
            text=node_labels,
            textposition='top center',
            hovertext=[f"{node}<br>Centrality: {centrality.get(node, 0):.3f}" 
                      for node in node_labels],
            hoverinfo='text'
        )
        
        # Layout
        layout = go.Layout(
            title=f"3D Concept Network: {concept}",
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, showticklabels=False, title=''),
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor='rgba(0,0,0,0.05)',
            showlegend=False
        )
        
        fig = go.Figure(data=edge_trace + [node_trace], layout=layout)
        
        # Save
        output_path = self.output_dir / f"concept_network_3d_{concept.replace(' ', '_')}.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def create_influence_visualization(self, philosopher: str, 
                                     influence_data: Dict,
                                     viz_type: str = 'tree') -> str:
        """
        Create influence chain visualization
        
        Args:
            philosopher: Central philosopher
            influence_data: Influence tree structure
            viz_type: 'tree', 'radial', or 'timeline'
        
        Returns:
            Path to saved visualization
        """
        if viz_type == 'tree':
            return self._create_influence_tree(philosopher, influence_data)
        elif viz_type == 'radial':
            return self._create_influence_radial(philosopher, influence_data)
        elif viz_type == 'timeline':
            return self._create_influence_timeline(philosopher, influence_data)
    
    def _create_influence_tree(self, philosopher: str, influence_data: Dict) -> str:
        """Create hierarchical tree visualization of influences"""
        
        # Build hierarchical structure
        G = nx.DiGraph()
        
        def add_influences(node_name, influences, depth=0):
            if depth > 3:  # Limit depth
                return
            
            for inf in influences:
                inf_name = inf.get('philosopher', {}).get('name', 'Unknown')
                if inf_name != 'Unknown':
                    G.add_edge(node_name, inf_name)
                    G.nodes[inf_name]['distance'] = inf.get('distance', depth + 1)
                    
                    # Recursively add sub-influences
                    if 'influences' in inf:
                        add_influences(inf_name, inf['influences'], depth + 1)
        
        # Add root
        G.add_node(philosopher, distance=0)
        
        # Add influences
        influences = influence_data.get('nodes', [])
        for node in influences:
            if node.get('type') == 'philosopher':
                G.add_node(node['id'], **node)
        
        for rel in influence_data.get('relationships', []):
            G.add_edge(rel['source'], rel['target'])
        
        # Create hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if nx.nx_agraph else nx.spring_layout(G)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='none'
                )
            )
        
        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            distance = G.nodes[node].get('distance', 0)
            
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=30 - (distance * 5),
                        color=self.color_schemes['philosophers'],
                        line=dict(color='white', width=2)
                    ),
                    text=[node],
                    textposition='top center',
                    showlegend=False,
                    hovertext=f"{node}<br>Distance: {distance}",
                    hoverinfo='text'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Influence Tree: {philosopher}",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        # Save
        output_path = self.output_dir / f"influence_tree_{philosopher.replace(' ', '_')}.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def create_schools_visualization(self, schools_data: Dict) -> str:
        """Create visualization of philosophical schools and their relationships"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('School Sizes', 'School Relationships', 
                          'School Clusters', 'Member Distribution'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'pie'}]]
        )
        
        # 1. School sizes bar chart
        schools = schools_data.get('schools', [])
        school_names = [s['school'] for s in schools[:20]]
        member_counts = [s['member_count'] for s in schools[:20]]
        
        fig.add_trace(
            go.Bar(x=school_names, y=member_counts, 
                  marker_color=self.color_schemes['schools']),
            row=1, col=1
        )
        
        # 2. School relationships network
        relationships = schools_data.get('relationships', [])
        if relationships:
            G = nx.Graph()
            for rel in relationships:
                G.add_edge(rel['school1'], rel['school2'])
            
            pos = nx.spring_layout(G)
            
            # Add edges
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                fig.add_trace(
                    go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                             mode='lines', line=dict(color='gray'),
                             showlegend=False),
                    row=1, col=2
                )
            
            # Add nodes
            x_nodes = [pos[node][0] for node in G.nodes()]
            y_nodes = [pos[node][1] for node in G.nodes()]
            
            fig.add_trace(
                go.Scatter(x=x_nodes, y=y_nodes,
                         mode='markers+text',
                         marker=dict(size=20, color=self.color_schemes['schools']),
                         text=list(G.nodes()),
                         textposition='top center'),
                row=1, col=2
            )
        
        # 3. School clusters
        clusters = schools_data.get('clusters', {})
        if clusters:
            # Use t-SNE for cluster visualization
            cluster_data = []
            cluster_labels = []
            
            for cluster_id, schools_in_cluster in clusters.items():
                for school in schools_in_cluster:
                    cluster_data.append([cluster_id, len(schools_in_cluster)])
                    cluster_labels.append(school)
            
            if cluster_data and SKLEARN_AVAILABLE:
                tsne = TSNE(n_components=2, random_state=42)
                coords = tsne.fit_transform(cluster_data)
                
                fig.add_trace(
                    go.Scatter(x=coords[:, 0], y=coords[:, 1],
                             mode='markers+text',
                             marker=dict(size=15, 
                                       color=[d[0] for d in cluster_data],
                                       colorscale='Viridis'),
                             text=cluster_labels,
                             textposition='top center'),
                    row=2, col=1
                )
        
        # 4. Member distribution pie
        top_schools = schools[:10]
        fig.add_trace(
            go.Pie(labels=[s['school'] for s in top_schools],
                  values=[s['member_count'] for s in top_schools]),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Philosophical Schools Analysis",
            showlegend=False,
            height=1000
        )
        
        # Save
        output_path = self.output_dir / "schools_analysis.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def create_heatmap_visualization(self, data: pd.DataFrame, 
                                   title: str = "Concept Relationships") -> str:
        """Create heatmap visualization for concept relationships"""
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(data, cmap='YlOrRd', cbar=True, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"heatmap_{title.replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_timeline_visualization(self, timeline_data: List[Dict]) -> str:
        """Create timeline visualization for historical development"""
        
        # Prepare data
        df = pd.DataFrame(timeline_data)
        
        # Create Altair timeline
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('start:T', title='Time Period'),
            x2='end:T',
            y=alt.Y('category:N', title='Category'),
            color=alt.Color('importance:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['event', 'description', 'philosophers']
        ).properties(
            width=800,
            height=400,
            title='Philosophical Development Timeline'
        ).interactive()
        
        # Save
        output_path = self.output_dir / "timeline.html"
        chart.save(str(output_path))
        
        return str(output_path)
    
    def create_dashboard(self, graph_data: Dict, 
                        title: str = "SEP Knowledge Graph Dashboard") -> str:
        """Create comprehensive dashboard with multiple visualizations"""
        
        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .dashboard {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-top: 20px;
                }}
                .viz-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #333;
                }}
                .stats {{
                    display: flex;
                    justify-content: space-around;
                    margin-bottom: 30px;
                }}
                .stat-box {{
                    text-align: center;
                    padding: 15px;
                    background: #e3f2fd;
                    border-radius: 8px;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #1976d2;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{total_nodes}</div>
                    <div>Total Nodes</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{total_edges}</div>
                    <div>Total Relationships</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{num_philosophers}</div>
                    <div>Philosophers</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{num_concepts}</div>
                    <div>Concepts</div>
                </div>
            </div>
            
            <div class="dashboard">
                <div class="viz-container">
                    <h2>Node Distribution</h2>
                    <div id="nodeDistribution"></div>
                </div>
                
                <div class="viz-container">
                    <h2>Relationship Types</h2>
                    <div id="relationshipTypes"></div>
                </div>
                
                <div class="viz-container">
                    <h2>Most Connected Nodes</h2>
                    <div id="mostConnected"></div>
                </div>
                
                <div class="viz-container">
                    <h2>Time Period Distribution</h2>
                    <div id="timePeriods"></div>
                </div>
            </div>
            
            <script>
                {plot_scripts}
            </script>
        </body>
        </html>
        """
        
        # Generate plot scripts
        plot_scripts = self._generate_dashboard_plots(graph_data)
        
        # Fill template
        html_content = html_template.format(
            title=title,
            total_nodes=graph_data.get('stats', {}).get('total_nodes', 0),
            total_edges=graph_data.get('stats', {}).get('total_relationships', 0),
            num_philosophers=graph_data.get('stats', {}).get('philosophers', 0),
            num_concepts=graph_data.get('stats', {}).get('concepts', 0),
            plot_scripts=plot_scripts
        )
        
        # Save
        output_path = self.output_dir / "dashboard.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _debate_to_networkx(self, debate_data: Dict) -> nx.Graph:
        """Convert debate data to NetworkX graph"""
        G = nx.Graph()
        
        # Add camp nodes
        camps = debate_data.get('camps', {})
        for camp_name, members in camps.items():
            G.add_node(camp_name, node_type='camp')
            for member in members:
                G.add_node(member, node_type='philosopher')
                G.add_edge(member, camp_name)
        
        # Add argument nodes
        arguments = debate_data.get('key_arguments', {})
        for camp_name, camp_arguments in arguments.items():
            for i, argument in enumerate(camp_arguments[:5]):
                arg_id = f"{camp_name}_arg_{i}"
                G.add_node(arg_id, node_type='argument', text=argument)
                G.add_edge(camp_name, arg_id)
        
        return G
    
    def _force_directed_layout(self, G: nx.Graph, center_node: Optional[str] = None) -> Dict:
        """Force-directed layout with optional center"""
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        
        if center_node and center_node in G:
            # Center the specified node
            center_pos = pos[center_node]
            for node in pos:
                pos[node] = (pos[node][0] - center_pos[0], 
                           pos[node][1] - center_pos[1])
        
        return pos
    
    def _circular_layout(self, G: nx.Graph, center_node: Optional[str] = None) -> Dict:
        """Circular layout"""
        return nx.circular_layout(G)
    
    def _hierarchical_layout(self, G: nx.Graph, center_node: Optional[str] = None) -> Dict:
        """Hierarchical layout"""
        if nx.is_directed(G):
            # For directed graphs, use topological ordering
            try:
                return nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                return nx.spring_layout(G)
        else:
            # For undirected, create BFS tree from center
            if center_node and center_node in G:
                bfs_tree = nx.bfs_tree(G, center_node)
                return nx.nx_agraph.graphviz_layout(bfs_tree, prog='dot')
            return nx.spring_layout(G)
    
    def _radial_layout(self, G: nx.Graph, center_node: Optional[str] = None) -> Dict:
        """Radial layout with center node"""
        if not center_node or center_node not in G:
            center_node = max(G.degree(), key=lambda x: x[1])[0]
        
        # BFS from center
        distances = nx.single_source_shortest_path_length(G, center_node)
        
        # Group by distance
        layers = defaultdict(list)
        for node, dist in distances.items():
            layers[dist].append(node)
        
        pos = {center_node: (0, 0)}
        
        # Place nodes in concentric circles
        for distance, nodes in layers.items():
            if distance == 0:
                continue
            
            radius = distance * 0.5
            angle_step = 2 * np.pi / len(nodes)
            
            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pos[node] = (x, y)
        
        return pos
    
    def _spring_layout(self, G: nx.Graph, center_node: Optional[str] = None) -> Dict:
        """Spring layout"""
        return nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50)
    
    def _enhance_interactive_viz(self, html_path: Path):
        """Add custom enhancements to interactive visualizations"""
        
        # Read the HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add custom CSS and controls
        custom_css = """
        <style>
            .vis-network {
                border: 2px solid #ddd;
                border-radius: 8px;
            }
            .controls {
                margin: 20px 0;
                padding: 15px;
                background: #f5f5f5;
                border-radius: 8px;
            }
            .control-btn {
                margin: 5px;
                padding: 8px 15px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .control-btn:hover {
                background: #45a049;
            }
        </style>
        """
        
        custom_js = """
        <div class="controls">
            <button class="control-btn" onclick="network.fit()">Fit to Screen</button>
            <button class="control-btn" onclick="downloadImage()">Download Image</button>
            <button class="control-btn" onclick="togglePhysics()">Toggle Physics</button>
        </div>
        
        <script>
            function downloadImage() {
                const canvas = document.querySelector('canvas');
                const link = document.createElement('a');
                link.download = 'network.png';
                link.href = canvas.toDataURL();
                link.click();
            }
            
            let physicsEnabled = true;
            function togglePhysics() {
                physicsEnabled = !physicsEnabled;
                network.setOptions({physics: {enabled: physicsEnabled}});
            }
        </script>
        """
        
        # Insert custom elements
        content = content.replace('<body>', f'<body>\n{custom_css}')
        content = content.replace('<div id="mynetwork"', f'{custom_js}\n<div id="mynetwork"')
        
        # Write back
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_dashboard_plots(self, graph_data: Dict) -> str:
        """Generate Plotly plots for dashboard"""
        
        scripts = []
        
        # Node distribution pie chart
        node_types = graph_data.get('node_type_distribution', {})
        if node_types:
            labels = list(node_types.keys())
            values = list(node_types.values())
            
            scripts.append(f"""
            var nodeData = [{{
                values: {values},
                labels: {labels},
                type: 'pie',
                marker: {{
                    colors: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
                }}
            }}];
            
            Plotly.newPlot('nodeDistribution', nodeData);
            """)
        
        # Relationship types bar chart
        rel_types = graph_data.get('relationship_types', {})
        if rel_types:
            scripts.append(f"""
            var relData = [{{
                x: {list(rel_types.keys())},
                y: {list(rel_types.values())},
                type: 'bar',
                marker: {{
                    color: '#4ecdc4'
                }}
            }}];
            
            Plotly.newPlot('relationshipTypes', relData);
            """)
        
        # Most connected nodes
        most_connected = graph_data.get('most_connected', [])[:10]
        if most_connected:
            names = [item.get('name', 'Unknown') for item in most_connected]
            degrees = [item.get('degree', 0) for item in most_connected]
            
            scripts.append(f"""
            var connectedData = [{{
                x: {degrees},
                y: {names},
                type: 'bar',
                orientation: 'h',
                marker: {{
                    color: '#45b7d1'
                }}
            }}];
            
            var layout = {{
                yaxis: {{
                    automargin: true
                }}
            }};
            
            Plotly.newPlot('mostConnected', connectedData, layout);
            """)
        
        return '\n'.join(scripts)

# Utility functions for common visualizations
def create_quick_network_viz(nodes: List[str], edges: List[Tuple[str, str]], 
                           output_path: str = "quick_network.html"):
    """Quick function to create a basic network visualization"""
    
    net = Network(height="750px", width="100%")
    
    for node in nodes:
        net.add_node(node)
    
    for edge in edges:
        net.add_edge(edge[0], edge[1])
    
    net.save_graph(output_path)
    return output_path

def export_graph_to_gephi(G: nx.Graph, output_path: str = "graph.gexf"):
    """Export NetworkX graph to Gephi format"""
    nx.write_gexf(G, output_path)
    return output_path