"""
Graph Builder Module
Creates and manages knowledge graph representations using NetworkX
"""
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
import json

import networkx as nx
from pyvis.network import Network
from loguru import logger


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    node_id: str
    node_type: str  # 'section', 'table', 'formula', 'concept', 'rule'
    label: str
    properties: Dict[str, any]
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'id': self.node_id,
            'type': self.node_type,
            'label': self.label,
            'properties': self.properties
        }


@dataclass 
class GraphEdge:
    """Represents an edge in the knowledge graph"""
    source: str
    target: str
    edge_type: str  # 'references', 'contains', 'depends_on', 'related_to'
    properties: Dict[str, any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
            
    def to_dict(self) -> Dict[str, any]:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.edge_type,
            'properties': self.properties
        }


class GraphBuilder:
    """Builds and manages document knowledge graphs"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_index = {}  # Quick lookup by type
        
    def add_document(self, doc_id: str, doc_data: Dict[str, any]):
        """
        Add a document's structure to the graph
        
        Args:
            doc_id: Document identifier
            doc_data: Document data including sections, tables, formulas
        """
        # Add document root node
        doc_node = GraphNode(
            node_id=f"doc_{doc_id}",
            node_type="document",
            label=doc_data.get('title', 'Untitled'),
            properties={
                'doc_id': doc_id,
                'total_pages': doc_data.get('total_pages', 0),
                'doc_type': doc_data.get('doc_type', 'technical_standard')
            }
        )
        self._add_node(doc_node)
        
        # Add sections
        if 'sections' in doc_data:
            self._add_sections(doc_id, doc_data['sections'], f"doc_{doc_id}")
            
        # Add tables
        if 'tables' in doc_data:
            self._add_tables(doc_id, doc_data['tables'])
            
        # Add formulas
        if 'formulas' in doc_data:
            self._add_formulas(doc_id, doc_data['formulas'])
            
        # Add cross-references
        if 'cross_references' in doc_data:
            self._add_cross_references(doc_data['cross_references'])
            
        logger.info(f"Added document {doc_id} to graph. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        
    def _add_node(self, node: GraphNode):
        """Add a node to the graph"""
        self.graph.add_node(
            node.node_id,
            type=node.node_type,
            label=node.label,
            **node.properties
        )
        
        # Update index
        if node.node_type not in self.node_index:
            self.node_index[node.node_type] = set()
        self.node_index[node.node_type].add(node.node_id)
        
    def _add_edge(self, edge: GraphEdge):
        """Add an edge to the graph"""
        self.graph.add_edge(
            edge.source,
            edge.target,
            type=edge.edge_type,
            **edge.properties
        )
        
    def _add_sections(self, doc_id: str, sections: List[Dict], parent_id: str):
        """Add document sections to the graph"""
        for section in sections:
            # Create section node
            section_node = GraphNode(
                node_id=f"section_{doc_id}_{section['number']}",
                node_type="section",
                label=f"{section['number']} {section['title']}",
                properties={
                    'section_number': section['number'],
                    'level': section['level'],
                    'page_number': section.get('page_number', 0),
                    'content_preview': section.get('content', '')[:200]
                }
            )
            self._add_node(section_node)
            
            # Link to parent
            edge = GraphEdge(
                source=parent_id,
                target=section_node.node_id,
                edge_type="contains"
            )
            self._add_edge(edge)
            
            # Process subsections recursively
            if 'subsections' in section and section['subsections']:
                self._add_sections(doc_id, section['subsections'], section_node.node_id)
                
    def _add_tables(self, doc_id: str, tables: List[Dict]):
        """Add tables to the graph"""
        for table in tables:
            # Create table node
            table_node = GraphNode(
                node_id=f"table_{doc_id}_{table['table_id']}",
                node_type="table",
                label=table.get('title', f"Table {table['table_id']}"),
                properties={
                    'table_type': table.get('table_type', 'data'),
                    'page_number': table.get('page_number', 0),
                    'rows': len(table.get('data', [])),
                    'columns': len(table.get('headers', []))
                }
            )
            self._add_node(table_node)
            
            # Link to document
            edge = GraphEdge(
                source=f"doc_{doc_id}",
                target=table_node.node_id,
                edge_type="contains"
            )
            self._add_edge(edge)
            
            # Extract and link concepts from table
            self._extract_table_concepts(table_node.node_id, table)
            
    def _add_formulas(self, doc_id: str, formulas: List[Dict]):
        """Add formulas to the graph"""
        for formula in formulas:
            # Create formula node
            formula_node = GraphNode(
                node_id=f"formula_{doc_id}_{formula['formula_id']}",
                node_type="formula",
                label=formula['normalized_text'],
                properties={
                    'formula_type': formula.get('formula_type', 'equation'),
                    'variables': formula.get('variables', []),
                    'raw_text': formula.get('raw_text', '')
                }
            )
            self._add_node(formula_node)
            
            # Link to source section if available
            if 'source_location' in formula and 'section_number' in formula['source_location']:
                section_id = f"section_{doc_id}_{formula['source_location']['section_number']}"
                if self.graph.has_node(section_id):
                    edge = GraphEdge(
                        source=section_id,
                        target=formula_node.node_id,
                        edge_type="contains"
                    )
                    self._add_edge(edge)
                    
            # Create variable nodes and relationships
            for var in formula.get('variables', []):
                var_node_id = f"variable_{var}"
                if not self.graph.has_node(var_node_id):
                    var_node = GraphNode(
                        node_id=var_node_id,
                        node_type="variable",
                        label=var,
                        properties={'symbol': var}
                    )
                    self._add_node(var_node)
                    
                # Link formula to variable
                edge = GraphEdge(
                    source=formula_node.node_id,
                    target=var_node_id,
                    edge_type="uses_variable"
                )
                self._add_edge(edge)
                
    def _add_cross_references(self, cross_refs: List[Dict]):
        """Add cross-references as edges"""
        for ref in cross_refs:
            source = ref.get('source')
            target = ref.get('target')
            
            if source and target:
                edge = GraphEdge(
                    source=source,
                    target=target,
                    edge_type="references",
                    properties={
                        'reference_type': ref.get('type', 'section'),
                        'reference_text': ref.get('text', '')
                    }
                )
                self._add_edge(edge)
                
    def _extract_table_concepts(self, table_node_id: str, table: Dict):
        """Extract concepts from table data"""
        # Extract key concepts from headers and data
        concepts = set()
        
        # From headers
        for header in table.get('headers', []):
            if len(header) > 2 and header.lower() not in ['id', 'no', 'ref']:
                concepts.add(header)
                
        # From first column (often contains key terms)
        data = table.get('data', [])
        if data and len(data[0]) > 0:
            for row in data[:10]:  # Sample first 10 rows
                first_cell = str(row[0]) if row else ""
                if len(first_cell) > 3:
                    concepts.add(first_cell)
                    
        # Create concept nodes and link to table
        for concept in list(concepts)[:5]:  # Limit to 5 concepts per table
            concept_id = f"concept_{concept.lower().replace(' ', '_')}"
            
            if not self.graph.has_node(concept_id):
                concept_node = GraphNode(
                    node_id=concept_id,
                    node_type="concept",
                    label=concept,
                    properties={'term': concept}
                )
                self._add_node(concept_node)
                
            # Link table to concept
            edge = GraphEdge(
                source=table_node_id,
                target=concept_id,
                edge_type="related_to"
            )
            self._add_edge(edge)
            
    def find_related_nodes(self, node_id: str, max_distance: int = 2) -> List[str]:
        """Find nodes related to a given node within max_distance"""
        if not self.graph.has_node(node_id):
            return []
            
        # Use BFS to find related nodes
        related = set()
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            
            if current in visited or distance > max_distance:
                continue
                
            visited.add(current)
            if distance > 0:
                related.add(current)
                
            # Add neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
                    
        return list(related)
        
    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return None
            
    def get_node_context(self, node_id: str) -> Dict[str, any]:
        """Get comprehensive context about a node"""
        if not self.graph.has_node(node_id):
            return {}
            
        node_data = self.graph.nodes[node_id]
        
        # Get connected nodes
        predecessors = list(self.graph.predecessors(node_id))
        successors = list(self.graph.successors(node_id))
        
        # Get edge information
        in_edges = []
        for pred in predecessors:
            edge_data = self.graph.get_edge_data(pred, node_id)
            if edge_data:
                for key, data in edge_data.items():
                    in_edges.append({
                        'source': pred,
                        'type': data.get('type', 'unknown'),
                        'properties': {k: v for k, v in data.items() if k != 'type'}
                    })
                    
        out_edges = []
        for succ in successors:
            edge_data = self.graph.get_edge_data(node_id, succ)
            if edge_data:
                for key, data in edge_data.items():
                    out_edges.append({
                        'target': succ,
                        'type': data.get('type', 'unknown'),
                        'properties': {k: v for k, v in data.items() if k != 'type'}
                    })
                    
        return {
            'node_id': node_id,
            'node_data': node_data,
            'in_edges': in_edges,
            'out_edges': out_edges,
            'predecessors': predecessors,
            'successors': successors
        }
        
    def search_nodes(self, query: str, node_type: Optional[str] = None) -> List[str]:
        """Search for nodes matching query"""
        results = []
        query_lower = query.lower()
        
        # Filter by type if specified
        if node_type and node_type in self.node_index:
            nodes_to_search = self.node_index[node_type]
        else:
            nodes_to_search = self.graph.nodes()
            
        for node_id in nodes_to_search:
            node_data = self.graph.nodes[node_id]
            
            # Search in label
            if query_lower in node_data.get('label', '').lower():
                results.append(node_id)
                continue
                
            # Search in properties
            for key, value in node_data.items():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(node_id)
                    break
                elif isinstance(value, list) and any(query_lower in str(v).lower() for v in value):
                    results.append(node_id)
                    break
                    
        return results
        
    def visualize_graph(self, output_file: str = "knowledge_graph.html", 
                       subset_nodes: Optional[List[str]] = None):
        """Create interactive visualization of the graph"""
        net = Network(height="750px", width="100%", directed=True)
        
        # Configure physics
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
        
        # Determine nodes to visualize
        if subset_nodes:
            subgraph = self.graph.subgraph(subset_nodes)
        else:
            subgraph = self.graph
            
        # Add nodes with styling based on type
        node_colors = {
            'document': '#FF6B6B',
            'section': '#4ECDC4',
            'table': '#45B7D1',
            'formula': '#96CEB4',
            'concept': '#DDA0DD',
            'variable': '#FFD93D',
            'rule': '#FF8C94'
        }
        
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            node_type = node_data.get('type', 'unknown')
            
            net.add_node(
                node_id,
                label=node_data.get('label', node_id),
                color=node_colors.get(node_type, '#CCCCCC'),
                title=f"Type: {node_type}\n" + "\n".join(
                    f"{k}: {v}" for k, v in node_data.items() 
                    if k not in ['label', 'type'] and not k.startswith('_')
                )
            )
            
        # Add edges with styling based on type
        edge_colors = {
            'contains': '#666666',
            'references': '#FF6B6B',
            'depends_on': '#4ECDC4',
            'related_to': '#45B7D1',
            'uses_variable': '#96CEB4'
        }
        
        for source, target, data in subgraph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            net.add_edge(
                source,
                target,
                color=edge_colors.get(edge_type, '#999999'),
                title=edge_type,
                arrows='to'
            )
            
        # Save visualization
        net.save_graph(output_file)
        logger.info(f"Graph visualization saved to {output_file}")
        
    def export_graph(self, format: str = "json") -> str:
        """Export graph in various formats"""
        if format == "json":
            # Convert to node-link format
            data = nx.node_link_data(self.graph)
            return json.dumps(data, indent=2)
        elif format == "gexf":
            # Return GEXF XML string
            import io
            buffer = io.StringIO()
            nx.write_gexf(self.graph, buffer)
            return buffer.getvalue()
        elif format == "graphml":
            # Return GraphML XML string
            import io
            buffer = io.StringIO()
            nx.write_graphml(self.graph, buffer)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def add_document_node(self, doc_id: str, title: str):
        """Add a document root node"""
        doc_node = GraphNode(
            node_id=doc_id,
            node_type="document", 
            label=title,
            properties={'doc_id': doc_id}
        )
        self._add_node(doc_node)
        
    def add_section_node(self, section_id: str, title: str, properties: Dict[str, any]):
        """Add a section node"""
        section_node = GraphNode(
            node_id=section_id,
            node_type="section",
            label=title,
            properties=properties
        )
        self._add_node(section_node)
        
    def add_table_node(self, table_id: str, title: str, properties: Dict[str, any]):
        """Add a table node"""
        table_node = GraphNode(
            node_id=table_id,
            node_type="table",
            label=title,
            properties=properties
        )
        self._add_node(table_node)
        
    def add_formula_node(self, formula_id: str, formula_text: str, properties: Dict[str, any]):
        """Add a formula node"""
        formula_node = GraphNode(
            node_id=formula_id,
            node_type="formula",
            label=formula_text,
            properties=properties
        )
        self._add_node(formula_node)
        
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, properties: Dict[str, any] = None):
        """Add a relationship between nodes"""
        if properties is None:
            properties = {}
        edge = GraphEdge(
            source=source_id,
            target=target_id,
            edge_type=relationship_type,
            properties=properties
        )
        self._add_edge(edge)
        
    def get_related_nodes(self, node_id: str, relationship_types: List[str] = None, max_relationships: int = 10):
        """Get nodes related to the given node"""
        related_nodes = []
        
        if not self.graph.has_node(node_id):
            return related_nodes
            
        # Get outgoing edges
        for _, target, edge_data in self.graph.out_edges(node_id, data=True):
            if relationship_types is None or edge_data.get('type') in relationship_types:
                if len(related_nodes) >= max_relationships:
                    break
                target_data = self.graph.nodes[target]
                related_node = GraphNode(
                    node_id=target,
                    node_type=target_data.get('type', 'unknown'),
                    label=target_data.get('label', target),
                    properties={k: v for k, v in target_data.items() if k not in ['type', 'label']}
                )
                related_nodes.append(related_node)
                
        # Get incoming edges
        for source, _, edge_data in self.graph.in_edges(node_id, data=True):
            if relationship_types is None or edge_data.get('type') in relationship_types:
                if len(related_nodes) >= max_relationships:
                    break
                source_data = self.graph.nodes[source]
                related_node = GraphNode(
                    node_id=source,
                    node_type=source_data.get('type', 'unknown'),
                    label=source_data.get('label', source),
                    properties={k: v for k, v in source_data.items() if k not in ['type', 'label']}
                )
                if related_node not in related_nodes:  # Avoid duplicates
                    related_nodes.append(related_node)
                    
        return related_nodes
        
    def clear_graph(self):
        """Clear the entire graph"""
        self.graph.clear()
        self.node_index = {}
        logger.info("Knowledge graph cleared")

    def get_statistics(self) -> Dict[str, any]:
        """Get graph statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'edge_types': {},
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'density': nx.density(self.graph)
        }
        
        # Count node types
        for node_type, node_ids in self.node_index.items():
            stats['node_types'][node_type] = len(node_ids)
            
        # Count edge types
        edge_type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        stats['edge_types'] = edge_type_counts
        
        # Find most connected nodes
        degree_centrality = nx.degree_centrality(self.graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        stats['most_connected_nodes'] = [
            {'node_id': node_id, 'centrality': centrality}
            for node_id, centrality in top_nodes
        ]
        
        return stats