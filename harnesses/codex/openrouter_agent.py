"""Harbor's codex agent for models served through OpenRouter (run.sh openrouter).

Harbor passes Codex only the last segment of the model name (`--model gpt-6.1-sol` for `openai/gpt-6.1-sol`), but
OpenRouter needs the full slug (`anthropic/claude-opus-5.5`). Run with

    harbor run ... -a openrouter_agent:OpenRouterCodex -m openrouter/<key>      (PYTHONPATH=harnesses/codex)

where <key> is the model's name in the results; SLUGS maps it to the OpenRouter slug. Everything else is
Harbor's codex agent.
"""
import shlex

from harbor.agents.installed.codex import Codex

SLUGS = {
    "claude-opus-5-5": "anthropic/claude-opus-5.5",
    "gemini-4-neon": "google/gemini-4-neon",
}


class OpenRouterCodex(Codex):
    def _slug(self) -> str:
        provider, _, key = (self.model_name or "").partition("/")
        if provider != "openrouter" or key not in SLUGS:
            raise ValueError(f"model {self.model_name!r}: expected openrouter/<key>, key one of {sorted(SLUGS)}")
        return SLUGS[key]

    async def run(self, instruction, environment, context):
        self._slug()  # fail before the container does any work
        return await super().run(instruction, environment, context)

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        if "codex exec " in command:
            short = f"--model {self.model_name.split('/')[-1]} "
            if command.count(short) != 1:
                raise RuntimeError("unexpected Harbor codex invocation; refusing to run")
            command = command.replace(short, f"--model {shlex.quote(self._slug())} ", 1)
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)
