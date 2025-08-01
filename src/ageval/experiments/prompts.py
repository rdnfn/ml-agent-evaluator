"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Configuration of prompts varied during experiments.

For simplicity, other prompts are directly included in the implementations.
Note that with structured output, prompts and other language model inputs
are not just defined in string form, but also as Python classes or dicts
(see agent evaluator for example).
"""

from dataclasses import dataclass, field


@dataclass
class NoHarmAgentConfig:
    will_websearch_help_description: str = (
        "Confidence that a websearch will help "
        "correctly select the better response. "
        "Integer between 0 (won't help) and 5 "
        "(will with absolute certainty help), 3 "
        "would mean 'may help'."
        "Consider whether there are facts present in "
        "either response, and if (!) consider whether "
        "these facts can be checked in a websearch. "
        "For example a word count task can't be checked "
        "with a websearch, but the birthday of a celebrity "
        "may be checked. "
        "Remember that websearches do not help on maths problems."
    )


# Tool prompts


@dataclass
class FactCheckToolConfig:
    contains_facts_desc: str = (
        "Whether the text contains any facts that may be checked using a web search."
    )
    is_like_wiki_desc: str = "Whether the response text could be from a wiki page."
    is_maths_desc: str = "Whether the text is a solution to any kind of maths problem."
    is_word_count_desc: str = "Whether the text is providing a word count."
    confidence_web_helps_desc: str = (
        "Confidence that a websearch will help "
        "correctly select the better response. "
        "Integer between 0 (won't help) and 5 "
        "(will with absolute certainty help), 3 "
        "would mean 'may help'."
        "Consider whether there are facts present in "
        "either response, and if (!) consider whether "
        "these facts can be checked in a websearch. "
        "For example a word count task can't be checked "
        "with a websearch, but the birthday of a celebrity "
        "may be checked. "
        "Remember that websearches do not help on maths problems."
    )


@dataclass
class ToolConfig:
    fact_check: FactCheckToolConfig = field(default_factory=FactCheckToolConfig)


@dataclass
class PromptConfig:
    no_harm_agent: NoHarmAgentConfig = field(default_factory=NoHarmAgentConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
