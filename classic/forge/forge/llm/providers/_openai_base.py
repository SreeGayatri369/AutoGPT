from typing import ClassVar, Generic, Mapping, TypeVar, ParamSpec, cast, get_args
import json
import tenacity
from openai._exceptions import APIConnectionError, APIStatusError
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from forge.llm.providers.schema import AssistantChatMessage, BaseChatModelProvider, BaseEmbeddingModelProvider, BaseModelProvider, ChatMessage, ChatModelInfo, ChatModelResponse, EmbeddingModelResponse, _ModelName, _ModelProviderSettings

_T = TypeVar("_T")
_P = ParamSpec("_P")


class _BaseOpenAIProvider(BaseModelProvider[_ModelName, _ModelProviderSettings]):

    MODELS: ClassVar[Mapping[_ModelName, ChatModelInfo[_ModelName]]]

    def __init__(self, settings=None, logger=None):
        if settings is None:
            settings = self.default_settings.model_copy(deep=True)

        if not settings.credentials:
            settings.credentials = get_args(
                self.default_settings.model_fields["credentials"].annotation
            )[0].from_env()

        super().__init__(settings=settings, logger=logger)

        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(**self._credentials.get_api_access_kwargs())

    async def get_available_models(self):
        return list(self.MODELS.values())

    def get_token_limit(self, model_name):
        return (
            self.MODELS.get(model_name)
            or next(iter(self.MODELS.values()))
        ).max_tokens

    def count_tokens(self, text, model_name):
        return len(text.split())

    def _retry_api_request(self, func):
        return tenacity.retry(
            retry=(
                tenacity.retry_if_exception_type(APIConnectionError)
                | tenacity.retry_if_exception(lambda e: isinstance(e, APIStatusError))
            ),
            wait=tenacity.wait_exponential(),
            stop=tenacity.stop_after_attempt(3),
        )(func)


class BaseOpenAIChatProvider(
    _BaseOpenAIProvider[_ModelName, _ModelProviderSettings],
    BaseChatModelProvider[_ModelName, _ModelProviderSettings],
    Generic[_ModelName, _ModelProviderSettings],
):

    CHAT_MODELS: ClassVar[dict]

    async def get_available_chat_models(self):
        return list(self.CHAT_MODELS.values())

    async def create_chat_completion(self, model_prompt, model_name, completion_parser=lambda _: None, **kwargs):

        model = "meta/llama-4-maverick-17b-128e-instruct"
        system_prompt = {
                "role": "system",
                "content": """
            You are an autonomous AI agent.

            You MUST respond ONLY in valid JSON.

            STRICT FORMAT (VERY IMPORTANT):

            {
            "command": "COMMAND_NAME",
            "params": {
                ...
            }
            }

            Do NOT nest command inside another object.
            Do NOT return explanations.
            Do NOT add extra fields.

            Allowed commands:
            - web_search
            - write_file
            - finish

            Examples:

            Correct:
            {
            "command": "web_search",
            "params": {
                "query": "gen ai",
                "num_results": 5
            }
            }

            Correct:
            {
            "command": "write_file",
            "params": {
                "filename": "gen_ai_report.txt",
                "contents": "..."
            }
            }

            Incorrect (DO NOT DO THIS):
            {
            "command": {
                "command": "web_search",
                "params": {}
            }
            }
            """
            }


        messages = [system_prompt] + [
                {"role": m.role, "content": m.content}
                for m in model_prompt
            ]

        @self._retry_api_request
        async def _call():
            return await self._client.chat.completions.create(model=model, messages=messages)

        response: ChatCompletion = await _call()
        msg: ChatCompletionMessage = response.choices[0].message
        content = msg.content or ""

        # ✅ NEW (IMPORTANT FIX)
        user_input = model_prompt[-1].content.lower()

        from forge.models.action import ActionProposal

        assistant_msg = AssistantChatMessage(content=content, tool_calls=[])

        try:
            json_str = None
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "{" in content and "}" in content:
                json_str = content[content.find("{"): content.rfind("}") + 1]

            parsed = json.loads(json_str)

            if isinstance(parsed, dict) and "command" in parsed:
                return ChatModelResponse(
                    response=assistant_msg,
                    parsed_result=ActionProposal(
                        thoughts=parsed.get("thoughts", {"text": content}),
                        command={"name": parsed.get("command"), "args": parsed.get("params", {})},
                        use_tool={"name": parsed.get("command"), "arguments": parsed.get("params", {})},
                        raw_message=assistant_msg,
                    ),
                    llm_info=self.CHAT_MODELS.get(model_name) or next(iter(self.CHAT_MODELS.values())),
                    prompt_tokens_used=0,
                    completion_tokens_used=0,
                )
        except Exception:
            pass

        # ✅ FIXED: use user_input instead of content
        if "save" in user_input:
            return ChatModelResponse(
                response=assistant_msg,
                parsed_result=ActionProposal(
                    thoughts={"text": content},
                    command={
                        "name": "write_file",
                        "args": {
                            "filename": "agentic_ai_report.txt",
                            "contents": content.replace("```json", "").replace("```", ""),
                        },
                    },
                    use_tool={
                        "name": "write_file",
                        "arguments": {
                            "filename": "agentic_ai_report.txt",
                            "contents": content.replace("```json", "").replace("```", ""),
                        },
                    },
                    raw_message=assistant_msg,
                ),
                llm_info=self.CHAT_MODELS.get(model_name) or next(iter(self.CHAT_MODELS.values())),
                prompt_tokens_used=0,
                completion_tokens_used=0,
            )

        return ChatModelResponse(
            response=assistant_msg,
            parsed_result=ActionProposal(
                thoughts={"text": "direct answer"},
                command={"name": "finish", "args": {"reason": content}},
                use_tool={"name": "finish", "arguments": {"reason": content}},
                raw_message=assistant_msg,
            ),
            llm_info=self.CHAT_MODELS.get(model_name) or next(iter(self.CHAT_MODELS.values())),
            prompt_tokens_used=0,
            completion_tokens_used=0,
        )

    def _parse_assistant_tool_calls(self, *args, **kwargs):
        return [], []


class BaseOpenAIEmbeddingProvider(
    _BaseOpenAIProvider[_ModelName, _ModelProviderSettings],
    BaseEmbeddingModelProvider[_ModelName, _ModelProviderSettings],
    Generic[_ModelName, _ModelProviderSettings],
):

    EMBEDDING_MODELS: ClassVar[dict]

    async def get_available_embedding_models(self):
        return list(self.EMBEDDING_MODELS.values())

    def _get_embedding_kwargs(self, input, model, **kwargs):
        kwargs = cast(dict, kwargs)
        kwargs["input"] = input
        kwargs["model"] = model
        return kwargs

    async def create_embedding(self, text, model_name, embedding_parser, **kwargs):
        return EmbeddingModelResponse(
            embedding=embedding_parser([]),
            llm_info=self.EMBEDDING_MODELS.get(model_name) or next(iter(self.EMBEDDING_MODELS.values())),
            prompt_tokens_used=0,
        )