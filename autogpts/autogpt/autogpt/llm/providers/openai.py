from __future__ import annotations

import logging
from typing import Iterable

from autogpt.core.resource.model_providers import CompletionModelFunction
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command import Command

logger = logging.getLogger(__name__)


def get_openai_command_specs(
    commands: Iterable[Command],
) -> list[CompletionModelFunction]:
    """Get OpenAI-consumable function specs for the agent's available commands."""
    return [
        CompletionModelFunction(
            name=command.name,
            description=command.description,
            parameters={param.name: param.spec for param in command.parameters},
        )
        for command in commands
    ]


def format_openai_function_for_prompt(func: CompletionModelFunction) -> str:
    """Returns the function formatted similarly to the way OpenAI does it internally."""

    def param_signature(name: str, spec: JSONSchema) -> str:
        type_dec = (
            spec.type if not spec.enum else " | ".join(repr(e) for e in spec.enum)
        )
        return (
            f"// {spec.description}\n" if spec.description else ""
        ) + f"{name}{'' if spec.required else '?'}: {type_dec},"

    return "\n".join(
        [
            f"// {func.description}",
            f"type {func.name} = (_ :{{",
            *[param_signature(name, p) for name, p in func.parameters.items()],
            "}) => any;",
        ]
    )


def format_function_specs_as_typescript_ns(
    functions: list[CompletionModelFunction],
) -> str:
    """Returns a function signature block in the TypeScript namespace format."""
    return (
        "namespace functions {\n\n"
        + "\n\n".join(format_openai_function_for_prompt(f) for f in functions)
        + "\n\n} // namespace functions"
    )


def count_openai_functions_tokens(
    functions: list[CompletionModelFunction], for_model: str
) -> int:
    """Returns the number of tokens taken up by a set of function definitions."""
    from autogpt.llm.utils import count_string_tokens

    return count_string_tokens(
        f"# Tools\n\n## functions\n\n{format_function_specs_as_typescript_ns(functions)}",
        for_model,
    )
