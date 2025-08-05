"""
Query Processor Module
Handles query understanding, planning, and response generation using GPT-4
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from openai import OpenAI
from loguru import logger


class QueryType(Enum):
    """Types of queries the system can handle"""
    LOOKUP = "lookup"  # Simple information retrieval
    CALCULATION = "calculation"  # Requires mathematical computation
    COMPARISON = "comparison"  # Compare multiple items
    EXPLANATION = "explanation"  # Explain a concept or rule
    VALIDATION = "validation"  # Check if something meets requirements
    MULTI_STEP = "multi_step"  # Complex query requiring multiple operations


@dataclass
class QueryIntent:
    """Parsed query intent and extracted information"""
    query_type: QueryType
    entities: Dict[str, any] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)
    required_sections: List[str] = field(default_factory=list)
    required_calculations: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'query_type': self.query_type.value,
            'entities': self.entities,
            'parameters': self.parameters,
            'required_sections': self.required_sections,
            'required_calculations': self.required_calculations,
            'constraints': self.constraints
        }


@dataclass
class QueryResponse:
    """Structured response to a query"""
    answer: str
    supporting_evidence: List[Dict[str, any]] = field(default_factory=list)
    calculations: List[Dict[str, any]] = field(default_factory=list)
    citations: List[Dict[str, any]] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'answer': self.answer,
            'supporting_evidence': self.supporting_evidence,
            'calculations': self.calculations,
            'citations': self.citations,
            'confidence': self.confidence,
            'reasoning_steps': self.reasoning_steps
        }


class QueryProcessor:
    """Processes natural language queries using GPT-4"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        
        # Query understanding prompts
        self.intent_prompt = """You are analyzing a technical query about building codes and standards.
        
Query: {query}

Extract the following information:
1. Query Type: Is this a lookup, calculation, comparison, explanation, validation, or multi_step query?
2. Entities: What specific things are being asked about? (e.g., "6-car carpark", "ventilation rate")
3. Parameters: What numerical values are mentioned? Extract as key-value pairs.
4. Required Sections: What document sections might contain the answer?
5. Required Calculations: What calculations need to be performed?
6. Constraints: What conditions or requirements are mentioned?

Respond in JSON format:
{{
    "query_type": "calculation",
    "entities": {{"subject": "6-car carpark", "requirement": "ventilation"}},
    "parameters": {{"num_cars": 6}},
    "required_sections": ["ventilation requirements", "carpark ventilation"],
    "required_calculations": ["total_ventilation = num_cars * rate_per_car"],
    "constraints": ["minimum ventilation rate"]
}}"""

        self.response_prompt = """Generate a clear, professional response to this technical query.

Query: {query}
Intent: {intent}
Context: {context}
Calculations: {calculations}

Requirements:
1. Provide a direct answer to the question
2. Include specific values and calculations
3. Cite relevant sections and standards
4. Use clear, technical language
5. Include units in all measurements

Format the response professionally with:
- Direct answer first
- Supporting calculations
- Source citations in format "Section X.Y.Z"
- Any relevant notes or conditions"""

    def parse_query(self, query: str) -> QueryIntent:
        """
        Parse user query to extract intent and entities
        
        Args:
            query: Natural language query
            
        Returns:
            QueryIntent object
        """
        try:
            # Use GPT-4 to parse query
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a technical document analysis assistant."},
                    {"role": "user", "content": self.intent_prompt.format(query=query)}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            intent_data = json.loads(response.choices[0].message.content)
            
            # Create QueryIntent object
            intent = QueryIntent(
                query_type=QueryType(intent_data.get('query_type', 'lookup')),
                entities=intent_data.get('entities', {}),
                parameters=intent_data.get('parameters', {}),
                required_sections=intent_data.get('required_sections', []),
                required_calculations=intent_data.get('required_calculations', []),
                constraints=intent_data.get('constraints', [])
            )
            
            logger.info(f"Parsed query intent: {intent.query_type.value}")
            return intent
            
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            # Return default intent
            return QueryIntent(query_type=QueryType.LOOKUP)
            
    def plan_query_execution(self, intent: QueryIntent) -> List[str]:
        """
        Plan the steps needed to answer the query
        
        Args:
            intent: Parsed query intent
            
        Returns:
            List of execution steps
        """
        steps = []
        
        # Retrieval steps
        if intent.required_sections:
            steps.append(f"Search for sections: {', '.join(intent.required_sections)}")
            
        # Table lookup steps
        if any(keyword in str(intent.entities) for keyword in ['table', 'rate', 'value']):
            steps.append("Search for relevant tables")
            
        # Calculation steps
        if intent.query_type == QueryType.CALCULATION:
            steps.append("Extract calculation formulas")
            for calc in intent.required_calculations:
                steps.append(f"Perform calculation: {calc}")
                
        # Validation steps
        if intent.query_type == QueryType.VALIDATION:
            steps.append("Retrieve validation criteria")
            steps.append("Check against requirements")
            
        # Comparison steps
        if intent.query_type == QueryType.COMPARISON:
            steps.append("Retrieve items to compare")
            steps.append("Extract comparison criteria")
            
        return steps
        
    def generate_response(self, 
                         query: str,
                         intent: QueryIntent,
                         context: List[Dict[str, any]],
                         calculations: Optional[List[Dict[str, any]]] = None) -> QueryResponse:
        """
        Generate final response to user query
        
        Args:
            query: Original user query
            intent: Parsed query intent
            context: Retrieved context documents
            calculations: Performed calculations
            
        Returns:
            QueryResponse object
        """
        try:
            # Prepare context string
            context_str = self._format_context(context)
            
            # Prepare calculations string
            calc_str = self._format_calculations(calculations) if calculations else "No calculations performed"
            
            # Generate response using GPT-4
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a technical expert on building codes and standards."},
                    {"role": "user", "content": self.response_prompt.format(
                        query=query,
                        intent=json.dumps(intent.to_dict()),
                        context=context_str,
                        calculations=calc_str
                    )}
                ],
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            
            # Extract structured information from response
            answer, citations = self._extract_response_components(response_text, context)
            
            # Create response object
            query_response = QueryResponse(
                answer=answer,
                supporting_evidence=context[:3],  # Top 3 context items
                calculations=calculations or [],
                citations=citations,
                confidence=self._calculate_confidence(intent, context, calculations),
                reasoning_steps=self.plan_query_execution(intent)
            )
            
            return query_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return QueryResponse(
                answer=f"I encountered an error processing your query: {str(e)}",
                confidence=0.0
            )
            
    def _format_context(self, context: List[Dict[str, any]]) -> str:
        """Format context documents for prompt"""
        if not context:
            return "No relevant context found."
            
        formatted = []
        for i, ctx in enumerate(context[:5]):  # Limit to top 5
            section = f"[{i+1}] Section {ctx.get('section_number', 'Unknown')}: {ctx.get('title', 'Untitled')}\n"
            content = ctx.get('content', '')[:500]  # Limit content length
            formatted.append(section + content)
            
        return "\n\n".join(formatted)
        
    def _format_calculations(self, calculations: List[Dict[str, any]]) -> str:
        """Format calculations for prompt"""
        if not calculations:
            return "No calculations"
            
        formatted = []
        for calc in calculations:
            expr = calc.get('expression', '')
            result = calc.get('numeric_result', '')
            units = calc.get('units', '')
            
            if units:
                formatted.append(f"{expr} = {result} {units}")
            else:
                formatted.append(f"{expr} = {result}")
                
        return "\n".join(formatted)
        
    def _extract_response_components(self, 
                                   response_text: str, 
                                   context: List[Dict[str, any]]) -> Tuple[str, List[Dict[str, any]]]:
        """Extract answer and citations from response text"""
        # For now, return full response as answer
        answer = response_text
        
        # Extract citations (look for "Section X.Y.Z" patterns)
        citation_pattern = r'Section\s+(\d+(?:\.\d+)*)'
        matches = re.findall(citation_pattern, response_text)
        
        citations = []
        for match in matches:
            # Find matching context
            for ctx in context:
                if ctx.get('section_number') == match:
                    citations.append({
                        'section': match,
                        'title': ctx.get('title', ''),
                        'page': ctx.get('page_number', 0)
                    })
                    break
                    
        return answer, citations
        
    def _calculate_confidence(self, 
                            intent: QueryIntent,
                            context: List[Dict[str, any]],
                            calculations: Optional[List[Dict[str, any]]]) -> float:
        """Calculate confidence score for response"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on evidence
        if context:
            confidence += min(0.3, len(context) * 0.1)
            
        # Increase confidence if calculations successful
        if calculations and intent.query_type == QueryType.CALCULATION:
            if all(calc.get('error') is None for calc in calculations):
                confidence += 0.2
                
        # Decrease confidence for complex queries
        if intent.query_type == QueryType.MULTI_STEP:
            confidence -= 0.1
            
        return min(1.0, max(0.0, confidence))
        
    def format_response_markdown(self, response: QueryResponse) -> str:
        """Format response as markdown for display"""
        md = f"## Answer\n\n{response.answer}\n\n"
        
        if response.calculations:
            md += "### Calculations\n\n"
            for calc in response.calculations:
                expr = calc.get('expression', '')
                result = calc.get('numeric_result', '')
                units = calc.get('units', '')
                
                if units:
                    md += f"- `{expr} = {result} {units}`\n"
                else:
                    md += f"- `{expr} = {result}`\n"
            md += "\n"
            
        if response.citations:
            md += "### Sources\n\n"
            for cite in response.citations:
                section = cite.get('section', '')
                title = cite.get('title', '')
                page = cite.get('page', '')
                
                md += f"- Section {section}: {title}"
                if page:
                    md += f" (Page {page})"
                md += "\n"
            md += "\n"
            
        if response.reasoning_steps:
            md += "### Reasoning Steps\n\n"
            for step in response.reasoning_steps:
                md += f"1. {step}\n"
            md += "\n"
            
        md += f"*Confidence: {response.confidence:.0%}*\n"
        
        return md