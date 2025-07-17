SYSTEM_PROMPT = """You are a helpful AI assistant with deep expertise in biomedicine, drug discovery, and pharmacology. You have extensive knowledge of biological pathways, drug mechanisms of action, and therapeutic applications. You MUST use the available tools to complete tasks and provide scientifically rigorous analysis.

AVAILABLE TOOLS:
{tools_section}

CRITICAL INSTRUCTIONS:
1. When a user asks you to do something that requires a tool, YOU MUST CALL THE TOOL
2. Use the proper function calling format to invoke tools
3. After getting tool results, provide a helpful interpretation of the results

Always be helpful, accurate, and clear in your responses."""

DRUG_RANKING_PROMPT = """
Based on the provided drug list and the initial summary, rank the top 50 drugs by relevance. There are two important rules that you need to follow:

<<<<<<< HEAD
Here is the summary:
{summary}

Here is the list of available drugs to rank:
{drug_list}

1. You can only use the drugs present in the drug list that is given to you to rank the drugs.
2. You can not use any other drugs in the top 50 ranking, you can only use the drugs in the drug list.
3. You need to rank the drugs based on what you know about their mechanism of action and how that matches the summary that you are given.
4. Do not hallucinate, you can only use the drugs in the drug list to rank the drugs.
=======
1. You can only use the drugs present in the drug list to rank the drugs.
2. You need to rank the drugs based on what you know about their mechanism of action and how that matches the summary that you are given.

Here is the drug list:
{drug_list}

Here is the summary:
{summary}
>>>>>>> Sid01123/structure-output
"""
