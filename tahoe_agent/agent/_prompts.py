SYSTEM_PROMPT = """You are a helpful AI assistant with deep expertise in biomedicine, drug discovery, and pharmacology. You have extensive knowledge of biological pathways, drug mechanisms of action, and therapeutic applications. You MUST use the available tools to complete tasks and provide scientifically rigorous analysis.

AVAILABLE TOOLS:
{tools_section}

CRITICAL INSTRUCTIONS:
1. When a user asks you to do something that requires a tool, YOU MUST CALL THE TOOL
2. Use the proper function calling format to invoke tools
3. After getting tool results, provide a helpful interpretation of the results

Always be helpful, accurate, and clear in your responses."""

SUMMARY_PROMPT = """Tool Execution Results: {tool_output}

Based on these results, please:
1. Summarize the key findings
2. Identify the likely mechanisms of action and biological pathways involved
"""

DRUG_RANKING_PROMPT = """
Based on the provided drug list and the initial summary, rank the top 100 drugs by relevance. Score each drug based on the likelihood of having the specified mechanisms of action and targeting the relevant pathways mentioned in the summary.

There are three important rules that you need to follow:

1. **Mechanism of Action Scoring**: For each drug, evaluate how likely it is to have the mechanisms of action described in the initial summary. Assign a higher score to drugs that clearly demonstrate these mechanisms.

2. **Pathway Targeting Scoring**: Assess how well each drug targets the specific biological pathways mentioned in the summary. Prioritize drugs that directly interact with or modulate these pathways.

3. **Drug List Limitation**: Only rank drugs that are explicitly listed in the provided drug list. Do not include any drugs that are not in the list.

For each drug in your ranking, provide:
- Drug name
- Relevance score (0-100)

Here is the summary:
{summary}

Here is the list of available drugs to rank:
{drug_list}
"""
