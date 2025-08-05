"""
Hierarchy Builder Module
Builds document structure hierarchy and maintains cross-references
"""
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re

from loguru import logger


@dataclass
class CrossReference:
    """Represents a cross-reference within the document"""
    source_section: str
    target_section: str
    reference_text: str
    reference_type: str  # 'section', 'table', 'figure', 'equation'
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'source': self.source_section,
            'target': self.target_section,
            'text': self.reference_text,
            'type': self.reference_type
        }


@dataclass
class DocumentNode:
    """Node in the document hierarchy tree"""
    section_number: str
    title: str
    level: int
    content: str
    page_number: int
    parent: Optional['DocumentNode'] = None
    children: List['DocumentNode'] = field(default_factory=list)
    cross_references: List[CrossReference] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def get_full_path(self) -> str:
        """Get full hierarchical path to this node"""
        path_parts = []
        current = self
        while current:
            path_parts.append(f"{current.section_number} {current.title}")
            current = current.parent
        return " > ".join(reversed(path_parts))
        
    def to_dict(self) -> Dict[str, any]:
        """Convert node to dictionary representation"""
        return {
            'section_number': self.section_number,
            'title': self.title,
            'level': self.level,
            'content': self.content[:200] + '...' if len(self.content) > 200 else self.content,
            'page_number': self.page_number,
            'full_path': self.get_full_path(),
            'children': [child.to_dict() for child in self.children],
            'cross_references': [ref.to_dict() for ref in self.cross_references],
            'metadata': self.metadata
        }


class HierarchyBuilder:
    """Builds and manages document hierarchy structure"""
    
    def __init__(self):
        # Cross-reference patterns
        self.ref_patterns = [
            # "see Section 3.2.1"
            (r'see\s+[Ss]ection\s+(\d+(?:\.\d+)*)', 'section'),
            
            # "refer to Table 3-1"
            (r'refer(?:s)?\s+to\s+[Tt]able\s+([\d\-\.]+)', 'table'),
            
            # "as shown in Figure 4.2"
            (r'(?:as\s+)?shown\s+in\s+[Ff]igure\s+([\d\-\.]+)', 'figure'),
            
            # "according to Equation (3.1)"
            (r'according\s+to\s+[Ee]quation\s*\(?([\d\.]+)\)?', 'equation'),
            
            # "per Section 3.2.1"
            (r'per\s+[Ss]ection\s+(\d+(?:\.\d+)*)', 'section'),
            
            # "in accordance with 3.2.1"
            (r'in\s+accordance\s+with\s+(\d+(?:\.\d+)*)', 'section'),
            
            # "(see 3.2.1)"
            (r'\(\s*see\s+(\d+(?:\.\d+)*)\s*\)', 'section'),
            
            # "Section 3.2.1 provides"
            (r'[Ss]ection\s+(\d+(?:\.\d+)*)\s+(?:provides|describes|specifies)', 'section')
        ]
        
    def build_hierarchy(self, sections: List[Dict[str, any]]) -> Tuple[List[DocumentNode], Dict[str, DocumentNode]]:
        """
        Build document hierarchy from flat section list
        
        Args:
            sections: List of section dictionaries with keys:
                     - number, title, level, content, page_number
                     
        Returns:
            Tuple of (root_nodes, section_lookup)
        """
        if not sections:
            return [], {}
            
        # Create nodes
        nodes = []
        section_lookup = {}
        
        for section in sections:
            node = DocumentNode(
                section_number=section['number'],
                title=section['title'],
                level=section['level'],
                content=section.get('content', ''),
                page_number=section.get('page_number', 1)
            )
            nodes.append(node)
            section_lookup[section['number']] = node
            
        # Build parent-child relationships
        root_nodes = self._build_tree_structure(nodes, section_lookup)
        
        # Extract cross-references
        self._extract_cross_references(nodes, section_lookup)
        
        # Validate hierarchy
        self._validate_hierarchy(root_nodes)
        
        logger.info(f"Built hierarchy with {len(root_nodes)} root sections and {len(nodes)} total sections")
        
        return root_nodes, section_lookup
        
    def _build_tree_structure(self, nodes: List[DocumentNode], 
                            section_lookup: Dict[str, DocumentNode]) -> List[DocumentNode]:
        """Build parent-child relationships between nodes"""
        root_nodes = []
        
        # Sort nodes by section number to ensure parents come before children
        sorted_nodes = sorted(nodes, key=lambda n: self._parse_section_number(n.section_number))
        
        for node in sorted_nodes:
            # Find parent based on section numbering
            parent = self._find_parent_node(node, section_lookup)
            
            if parent:
                node.parent = parent
                parent.children.append(node)
            else:
                root_nodes.append(node)
                
        return root_nodes
        
    def _parse_section_number(self, section_num: str) -> Tuple[int, ...]:
        """Parse section number into tuple for sorting"""
        # Handle various formats
        if 'Chapter' in section_num:
            match = re.search(r'Chapter\s+(\d+)', section_num)
            if match:
                return (int(match.group(1)),)
                
        # Standard dotted notation
        parts = re.findall(r'\d+', section_num)
        return tuple(int(p) for p in parts)
        
    def _find_parent_node(self, node: DocumentNode, 
                         section_lookup: Dict[str, DocumentNode]) -> Optional[DocumentNode]:
        """Find parent node based on section numbering"""
        # For standard numbering (e.g., 3.2.1)
        if '.' in node.section_number:
            parts = node.section_number.split('.')
            # Try progressively shorter prefixes
            for i in range(len(parts) - 1, 0, -1):
                parent_num = '.'.join(parts[:i])
                if parent_num in section_lookup:
                    return section_lookup[parent_num]
                    
        # For chapter-based numbering
        if node.level > 1:
            # Look for any node with level = current level - 1
            for section_num, potential_parent in section_lookup.items():
                if (potential_parent.level == node.level - 1 and 
                    self._is_ancestor(potential_parent.section_number, node.section_number)):
                    return potential_parent
                    
        return None
        
    def _is_ancestor(self, ancestor_num: str, descendant_num: str) -> bool:
        """Check if one section number is ancestor of another"""
        # Simple prefix check for dotted notation
        if '.' in ancestor_num and '.' in descendant_num:
            return descendant_num.startswith(ancestor_num + '.')
            
        # Chapter check
        if 'Chapter' in ancestor_num and ancestor_num in descendant_num:
            return True
            
        return False
        
    def _extract_cross_references(self, nodes: List[DocumentNode], 
                                section_lookup: Dict[str, DocumentNode]):
        """Extract all cross-references from node content"""
        for node in nodes:
            if not node.content:
                continue
                
            # Search for each reference pattern
            for pattern, ref_type in self.ref_patterns:
                matches = re.finditer(pattern, node.content, re.IGNORECASE)
                
                for match in matches:
                    target_ref = match.group(1)
                    
                    # Try to resolve the reference
                    target_node = self._resolve_reference(target_ref, ref_type, section_lookup)
                    
                    if target_node:
                        ref = CrossReference(
                            source_section=node.section_number,
                            target_section=target_node.section_number,
                            reference_text=match.group(0),
                            reference_type=ref_type
                        )
                        node.cross_references.append(ref)
                        
    def _resolve_reference(self, ref_text: str, ref_type: str, 
                         section_lookup: Dict[str, DocumentNode]) -> Optional[DocumentNode]:
        """Resolve a reference to its target node"""
        # Direct lookup for section references
        if ref_type == 'section' and ref_text in section_lookup:
            return section_lookup[ref_text]
            
        # Fuzzy matching for other types
        # Could be extended to handle table/figure references
        
        return None
        
    def _validate_hierarchy(self, root_nodes: List[DocumentNode]):
        """Validate the constructed hierarchy"""
        issues = []
        
        # Check for orphaned nodes
        all_nodes = set()
        
        def collect_nodes(node: DocumentNode):
            all_nodes.add(node.section_number)
            for child in node.children:
                collect_nodes(child)
                
        for root in root_nodes:
            collect_nodes(root)
            
        # Check for circular references
        def has_cycle(node: DocumentNode, visited: Set[str], stack: Set[str]) -> bool:
            visited.add(node.section_number)
            stack.add(node.section_number)
            
            for child in node.children:
                if child.section_number not in visited:
                    if has_cycle(child, visited, stack):
                        return True
                elif child.section_number in stack:
                    return True
                    
            stack.remove(node.section_number)
            return False
            
        for root in root_nodes:
            if has_cycle(root, set(), set()):
                issues.append(f"Circular reference detected starting from {root.section_number}")
                
        if issues:
            for issue in issues:
                logger.warning(f"Hierarchy validation issue: {issue}")
                
    def find_node(self, section_number: str, root_nodes: List[DocumentNode]) -> Optional[DocumentNode]:
        """Find a node by section number"""
        def search_recursive(node: DocumentNode) -> Optional[DocumentNode]:
            if node.section_number == section_number:
                return node
                
            for child in node.children:
                result = search_recursive(child)
                if result:
                    return result
                    
            return None
            
        for root in root_nodes:
            result = search_recursive(root)
            if result:
                return result
                
        return None
        
    def get_node_context(self, node: DocumentNode, include_ancestors: bool = True,
                        include_siblings: bool = True) -> Dict[str, any]:
        """Get contextual information about a node"""
        context = {
            'node': node.to_dict(),
            'ancestors': [],
            'siblings': [],
            'children': [child.to_dict() for child in node.children]
        }
        
        # Get ancestors
        if include_ancestors:
            current = node.parent
            while current:
                context['ancestors'].append({
                    'section_number': current.section_number,
                    'title': current.title
                })
                current = current.parent
                
        # Get siblings
        if include_siblings and node.parent:
            for sibling in node.parent.children:
                if sibling.section_number != node.section_number:
                    context['siblings'].append({
                        'section_number': sibling.section_number,
                        'title': sibling.title
                    })
                    
        return context
        
    def export_hierarchy_graph(self, root_nodes: List[DocumentNode]) -> Dict[str, any]:
        """Export hierarchy as a graph structure"""
        nodes = []
        edges = []
        
        def process_node(node: DocumentNode):
            # Add node
            nodes.append({
                'id': node.section_number,
                'label': f"{node.section_number} {node.title}",
                'level': node.level,
                'page': node.page_number
            })
            
            # Add edges to children
            for child in node.children:
                edges.append({
                    'source': node.section_number,
                    'target': child.section_number,
                    'type': 'hierarchy'
                })
                process_node(child)
                
            # Add cross-reference edges
            for ref in node.cross_references:
                edges.append({
                    'source': ref.source_section,
                    'target': ref.target_section,
                    'type': 'reference',
                    'ref_type': ref.reference_type
                })
                
        for root in root_nodes:
            process_node(root)
            
        return {
            'nodes': nodes,
            'edges': edges
        }